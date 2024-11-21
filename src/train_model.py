import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from tqdm import tqdm

def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def threshold_Rounder(oof_non_rounded, thresholds):
    return np.where(oof_non_rounded < thresholds[0], 1,
                    np.where(oof_non_rounded < thresholds[1], 2,
                             np.where(oof_non_rounded < thresholds[2], 3,
                                      np.where(oof_non_rounded < thresholds[3], 4, 5))))

def evaluate_predictions(thresholds, y_true, oof_non_rounded):
    rounded_p = threshold_Rounder(oof_non_rounded, thresholds)
    return RMSE(y_true, rounded_p)

def TrainML_round(train_X, train_y, test_X, model_class, n_splits=10, seed=42):
    SKF = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    train_scores = []
    val_scores = []

    oof_non_rounded = np.zeros(len(train_y), dtype=float)
    oof_rounded = np.zeros(len(train_y), dtype=int)
    test_preds = np.zeros((len(test_X), n_splits))

    for fold, (train_idx, val_idx) in enumerate(tqdm(SKF.split(train_X, train_y), desc="Training Folds")):
        # Split data
        X_train, X_val = train_X.iloc[train_idx], train_X.iloc[val_idx]
        y_train, y_val = train_y.iloc[train_idx], train_y.iloc[val_idx]

        # Clone and train model
        model = clone(model_class)
        model.fit(X_train, y_train)

        # Predict on validation set
        y_val_pred = model.predict(X_val)
        oof_non_rounded[val_idx] = y_val_pred  # Store non-rounded predictions
        oof_rounded[val_idx] = np.round(y_val_pred).astype(int)

        # Calculate RMSE
        train_rmse = RMSE(y_train, np.round(model.predict(X_train)).astype(int))
        val_rmse = RMSE(y_val, oof_rounded[val_idx])

        train_scores.append(train_rmse)
        val_scores.append(val_rmse)

        # Predict on test set
        test_preds[:, fold] = model.predict(test_X)

        print(f"Fold {fold+1} - Train RMSE: {train_rmse:.4f}, Validation RMSE: {val_rmse:.4f}")

    # Overall scores
    print(f"Mean Train RMSE: {np.mean(train_scores):.4f}")
    print(f"Mean Validation RMSE: {np.mean(val_scores):.4f}")

    # Optimize thresholds for rounding
    rmse_optimizer = minimize(
        evaluate_predictions,
        x0=[1.5, 2.5, 3.5, 4.5],
        args=(train_y, oof_non_rounded),
        method='Nelder-Mead'
    )
    assert rmse_optimizer.success, "Optimization did not converge."

    # Tune out-of-fold predictions
    tuned_thresholds = rmse_optimizer.x
    oof_tuned = threshold_Rounder(oof_non_rounded, tuned_thresholds)
    tuned_rmse = RMSE(train_y, oof_tuned)

    print(f"Optimized RMSE Score: {tuned_rmse:.4f}")

    # Tune test predictions
    test_pred_mean = np.mean(test_preds, axis=1)
    test_pred_tuned = threshold_Rounder(test_pred_mean, tuned_thresholds)

    return test_pred_tuned


def TrainML(train_X, train_y, test_X, model_class, n_splits=10, seed=42):
    SKF = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    train_scores = []
    val_scores = []

    oof = np.zeros(len(train_y), dtype=float)
    test_preds = np.zeros((len(test_X), n_splits))

    for fold, (train_idx, val_idx) in enumerate(tqdm(SKF.split(train_X, train_y), desc="Training Folds")):
        # Split data
        X_train, X_val = train_X.iloc[train_idx], train_X.iloc[val_idx]
        y_train, y_val = train_y.iloc[train_idx], train_y.iloc[val_idx]

        # Clone and train model
        model = clone(model_class)
        model.fit(X_train, y_train)

        # Predict on validation set
        y_val_pred = model.predict(X_val)
        oof[val_idx] = y_val_pred  # Store non-rounded predictions

        # Calculate RMSE
        train_rmse = RMSE(y_train, model.predict(X_train))
        val_rmse = RMSE(y_val, oof[val_idx])

        train_scores.append(train_rmse)
        val_scores.append(val_rmse)

        # Predict on test set
        test_preds[:, fold] = model.predict(test_X)

        print(f"Fold {fold+1} - Train RMSE: {train_rmse:.4f}, Validation RMSE: {val_rmse:.4f}")

    # Overall scores
    print(f"Mean Train RMSE: {np.mean(train_scores):.4f}")
    print(f"Mean Validation RMSE: {np.mean(val_scores):.4f}")

    # Tune test predictions
    test_pred_mean = np.mean(test_preds, axis=1)

    return test_pred_mean
