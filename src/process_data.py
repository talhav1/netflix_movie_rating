import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer


def impute_missing(method, data, params=None):
    # Map imputation methods to their corresponding classes
    imputers = {'KNN': KNNImputer,
                'iterative': IterativeImputer}
    
    # Get imputer class or raise an error if method is unknown
    imputer_class = imputers.get(method)
    if imputer_class is None:
        raise ValueError(f"Unknown imputation method: {method}")
    
    # Apply imputer with or without parameters
    params = params if params is not None else {}
    return imputer_class(**params).fit_transform(data)

def process_dates(dates):
    # Calculate and return date statistics (mean, std, percentiles, etc.)
    stats = np.array([
        dates.mean(axis=1),
        dates.std(axis=1),
        dates.min(axis=1),
        np.percentile(dates, 25, axis=1),
        np.median(dates, axis=1),
        np.percentile(dates, 75, axis=1),
        dates.max(axis=1)
    ])
    return stats.T  # Transpose to get features as columns

def merge_dates(X_dates, y_date, movie_dates=None):
    # Combine X_dates and y_date, subtract movie_dates if provided
    merged_dates = np.hstack((X_dates, y_date[:, None]))
    if movie_dates is not None:
        merged_dates -= movie_dates
    return merged_dates

def merge_features(ratings, X_dates, y_date, movie_dates=None):
    # Merge ratings with processed date statistics
    merged_dates = merge_dates(X_dates, y_date, movie_dates)
    processed_dates = process_dates(merged_dates)
    return np.hstack((ratings, processed_dates))


def train_test_split(data_dict, imputing_method, use_movie_dates=False, params=None):
    movie_dates = data_dict['movie_dates']
    train_ratings = data_dict['train_ratings_all']
    train_dates = data_dict['train_dates_all']
    train_y_date = data_dict['train_y_date']
    train_y = data_dict['train_y_rating']
    test_ratings = data_dict['test_ratings_all']
    test_dates = data_dict['test_dates_all']
    test_y_date = data_dict['test_y_date']

    train_ratings_imputed = impute_missing(imputing_method, train_ratings, params)
    train_dates_imputed = impute_missing(imputing_method, train_dates, params)
    test_ratings_imputed = impute_missing(imputing_method, test_ratings, params)
    test_dates_imputed = impute_missing(imputing_method, test_dates, params)

    if use_movie_dates == False:
        train_X = merge_features(train_ratings_imputed, train_dates_imputed, train_y_date)
        test_X = merge_features(test_ratings_imputed, test_dates_imputed, test_y_date)
    else:
        train_X = merge_features(train_ratings_imputed, train_dates_imputed, train_y_date, movie_dates)
        test_X = merge_features(test_ratings_imputed, test_dates_imputed, test_y_date, movie_dates)

    return train_X, train_y, test_X