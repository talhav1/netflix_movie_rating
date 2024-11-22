import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
import autoencoder


def impute_missing_simple(method, data, params=None):
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


def train_test_split(data, imputation_method, use_dates=False, use_movie_dates=False, imputation_params=None):
    # Extract data
    movie_dates = data['movie_dates']
    train_ratings, train_dates = data['train_ratings_all'], data['train_dates_all']
    train_y_date, train_y = data['train_y_date'], data['train_y_rating']
    test_ratings, test_dates = data['test_ratings_all'], data['test_dates_all']
    test_y_date = data['test_y_date']

    # Helper function for imputing data
    def impute_data(method, data, params=None):
        if method == 'autoencoder':
            return autoencoder.impute_data(data)
        elif method in ['KNN', 'iterative']:
            return impute_missing_simple(method, data, params)
        else:
            raise ValueError(f"Unsupported imputation method: {method}")

    # Impute missing values
    train_ratings_imputed = impute_data(imputation_method, train_ratings, imputation_params)
    train_dates_imputed = impute_data(imputation_method, train_dates, imputation_params)
    test_ratings_imputed = impute_data(imputation_method, test_ratings, imputation_params)
    test_dates_imputed = impute_data(imputation_method, test_dates, imputation_params)

    if use_dates is False:
        return train_ratings_imputed, train_y, test_ratings_imputed

    # Merge features
    if use_movie_dates:
        train_X = merge_features(train_ratings_imputed, train_dates_imputed, train_y_date, movie_dates)
        test_X = merge_features(test_ratings_imputed, test_dates_imputed, test_y_date, movie_dates)
    else:
        train_X = merge_features(train_ratings_imputed, train_dates_imputed, train_y_date)
        test_X = merge_features(test_ratings_imputed, test_dates_imputed, test_y_date)

    return train_X, train_y, test_X
