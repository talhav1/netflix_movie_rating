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
    learning_rate=0.05,      
    max_depth=6,            
    num_leaves=31,         
    min_child_samples=20,    
    lambda_l1=1.0,          
    lambda_l2=1.0           
)

XGB_Model = XGBRegressor(
    learning_rate=0.05,      
    max_depth=6,            
    min_child_weight=5,      
    reg_alpha=1.0,           
    reg_lambda=1.0,          
    subsample=0.8,          
    colsample_bytree=0.8     
)

CatBoost_Model = CatBoostRegressor(
    learning_rate=0.05,       
    depth=6,                 
    l2_leaf_reg=3.0,         
    random_strength=1.0,     
    iterations=1000,         
    early_stopping_rounds=100,  
    verbose=200             
)

voting_model = VotingRegressor(estimators=[
    ('lightgbm', Light),
    ('xgboost', XGB_Model),
    ('catboost', CatBoost_Model)
])

test_preds = train_model.trainML(train_X, train_y, test_X, voting_model)
np.savetxt("submission.csv", test_preds, delimiter=",")
