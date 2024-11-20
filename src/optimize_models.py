import optuna
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb

def objective_xgb(trial, X_train, y_train):
    params = {
        "booster": "gbtree",
        "objective": "multi:softmax",
        "num_class": 5,
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "eta": trial.suggest_float("eta", 0.01, 0.3),
        "lambda": trial.suggest_float("lambda", 1.0, 10.0),
        "alpha": trial.suggest_float("alpha", 0.0, 10.0),
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=1000,
        nfold=5,
        stratified=True,
        metrics="mlogloss",
        early_stopping_rounds=50,
        verbose_eval=False,
        seed=42,
    )
    return cv_results["test-mlogloss-mean"].min()


def objective_lgb(trial, X_train, y_train):
    params = {
        "objective": "multiclass",
        "num_class": 5,
        "boosting_type": "gbdt",
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
    }
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    cv_results = lgb.cv(
        params,
        dtrain,
        num_boost_round=1000,
        nfold=5,
        stratified=True,
        metrics="multi_logloss",
        early_stopping_rounds=50,
        verbose_eval=False,
        seed=42,
    )
    return np.min(cv_results["multi_logloss-mean"])


def objective_cat(trial, X_train, y_train):
    params = {
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "iterations": trial.suggest_int("iterations", 100, 1000),
    }
    
    model = CatBoostClassifier(**params, verbose=0, random_seed=42, loss_function='MultiClass')
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    return 1 - np.mean(scores)


def best_params(X_train, y_train):
    params_dict = {}
    # XGBoost
    study_xgb = optuna.create_study(direction="minimize")
    study_xgb.optimize(lambda trial: objective_xgb(trial, X_train, y_train), n_trials=50)
    params_dict['xgb'] = study_xgb.best_params

    # LightGBM
    study_lgb = optuna.create_study(direction="minimize")
    study_lgb.optimize(lambda trial: objective_lgb(trial, X_train, y_train), n_trials=50)
    params_dict['lgb'] = study_lgb.best_params


    # CatBoost
    study_cat = optuna.create_study(direction="minimize")
    study_cat.optimize(lambda trial: objective_cat(trial, X_train, y_train), n_trials=50)
    params_dict['cat'] = study_cat.best_params

    return params_dict
