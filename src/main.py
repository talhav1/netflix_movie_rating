import load_data, process_data, optimize_models, train_model
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor
import numpy as np

dir = 'data/'
data_dict = load_data.load_files(dir)

train_X, train_y, test_X = process_data.train_test_split(data=data_dict, imputation_method='autoencoder', use_movie_dates=False)

# Create model instances
Light = LGBMRegressor(
    learning_rate=0.05,       # Lower learning rate
    max_depth=6,             # Limit tree depth
    num_leaves=31,           # Standard number of leaves
    min_child_samples=20,    # Minimum samples per leaf
    lambda_l1=1.0,           # L1 regularization
    lambda_l2=1.0            # L2 regularization
)

XGB_Model = XGBRegressor(
    learning_rate=0.05,       # Lower learning rate
    max_depth=6,             # Limit tree depth
    min_child_weight=5,      # Minimum child weight
    reg_alpha=1.0,           # L1 regularization
    reg_lambda=1.0,          # L2 regularization
    subsample=0.8,           # Use subsampling
    colsample_bytree=0.8     # Use feature subsampling
)

CatBoost_Model = CatBoostRegressor(
    learning_rate=0.05,       # Lower learning rate
    depth=6,                 # Limit tree depth
    l2_leaf_reg=3.0,         # L2 regularization
    random_strength=1.0,     # Add randomness for regularization
    iterations=1000,         # Increase iterations for convergence
    early_stopping_rounds=100,  # Enable early stopping
    verbose=200              # Log progress
)

voting_model = VotingRegressor(estimators=[
    ('lightgbm', Light),
    ('xgboost', XGB_Model),
    ('catboost', CatBoost_Model)
])

test_preds = train_model.trainML(train_X, train_y, test_X, voting_model)
np.savetxt("submission.csv", test_preds, delimiter=",")