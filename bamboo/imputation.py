# bamboo/imputation.py
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import KNNImputer, IterativeImputer
from bamboo.utils import log
from fancyimpute import IterativeSVD

from bamboo.bamboo import Bamboo

@log
def impute_missing(self, strategy='mean', columns=None):
    """
    Impute missing values in the dataset based on the specified strategy for numeric columns,
    and mode for non-numeric columns.

    Parameters:
    - strategy: str, default='mean'
        The strategy to use for imputation of numeric columns. Options are:
        - 'mean': Replace NaN values with the mean of the column (numeric only).
        - 'median': Replace NaN values with the median of the column (numeric only).
        - 'mode': Replace NaN values with the mode of the column (works for both numeric and non-numeric columns).
    - columns: list or None, default=None
        A list of columns to apply the imputation to. If None, all columns will be imputed.

    Returns:
    - Bamboo: The Bamboo instance with imputed data.
    """
    if columns is None:
        columns = self.data.columns

    # impute missing values based on the data type of the cols, non-numeric cols will always be imputed using mode
    numeric_columns = self.data[columns].select_dtypes(include=[np.number]).columns
    non_numeric_columns = self.data[columns].select_dtypes(exclude=[np.number]).columns

    if strategy == 'mean':
        self.data[numeric_columns] = self.data[numeric_columns].fillna(self.data[numeric_columns].mean())
    elif strategy == 'median':
        self.data[numeric_columns] = self.data[numeric_columns].fillna(self.data[numeric_columns].median())
    elif strategy == 'mode':
        self.data[numeric_columns] = self.data[numeric_columns].fillna(self.data[numeric_columns].mode().iloc[0])
    else:
        raise ValueError("Unsupported strategy! Use 'mean', 'median', or 'mode'.")

    for column in non_numeric_columns:
        self.data[column] = self.data[column].fillna(self.data[column].mode().iloc[0])

    self.log_changes(f"Imputed missing values using {strategy} strategy for numeric columns and mode for non-numeric columns.")
    return self

@log
def drop_missing(self, axis=0, how='any', thresh=None, subset=None):
    """
    Drop rows or columns with missing values. To use threshold-based dropping, specify the `thresh` parameter, and specify how as None.

    Parameters:
    - axis: int, default=0
        Whether to drop rows (0) or columns (1).
    - how: str, default='any'
        - 'any': Drop rows or columns that contain any NaN values.
        - 'all': Drop rows or columns that contain all NaN values.
    - thresh: int, optional
        Require that many non-NA values to not drop.
    - subset: array-like, optional
        Labels along other axis to consider for missing value filtering.

    Returns:
    - Bamboo: The Bamboo instance with dropped missing values.
    """
    if thresh is None:
        self.data.dropna(axis=axis, how=how, subset=subset, inplace=True)
    elif how is None:
        self.data.dropna(axis=axis, thresh=thresh, subset=subset, inplace=True)
    else:
        raise ValueError("Specify either 'thresh' or 'how', not both.")
    self.log_changes(f"Dropped missing values with axis={axis}, how={how}, thresh={thresh}.")
    return self

@log
def fill_with_custom(self, custom_function, columns=None):
    """
    Impute missing values using a custom function.

    Parameters:
    - custom_function: callable
        A custom function that will be applied to fill NaN values.
        The function should take a Pandas Series as input and return a value.
    - columns: list or None, default=None
        A list of columns to apply the imputation to. If None, all columns will be imputed.

    Returns:
    - Bamboo: The Bamboo instance with imputed data.
    """
    if columns is None:
        columns = self.data.columns

    for column in columns:
        self.data[column] = self.data[column].apply(lambda x: custom_function(x) if pd.isna(x) else x)

    self.log_changes("Imputed missing values using custom function.")
    return self

@log
def impute_knn(self, n_neighbors=5, columns=None):
    """
    Impute missing values using K-Nearest Neighbors (KNN) imputation.

    Parameters:
    - n_neighbors: int, default=5
        Number of neighbors to use for KNN imputation.
    - columns: list or None, default=None
        A list of columns to apply the imputation to. If None, all numeric columns will be imputed.

    Returns:
    - Bamboo: The Bamboo instance with KNN-imputed data.
    """
    if columns is None:
        columns = self.data.select_dtypes(include=[np.number]).columns

    # Apply KNN Imputation
    imputer = KNNImputer(n_neighbors=n_neighbors)
    self.data[columns] = pd.DataFrame(imputer.fit_transform(self.data[columns]), columns=columns)

    self.log_changes(f"Imputed missing values using KNN with {n_neighbors} neighbors.")
    return self

@log
def interpolate_missing(self, method='linear', axis=0, limit=None, inplace=True):
    """
    Impute missing values by interpolation.

    Parameters:
    - method: str, default='linear'
        Interpolation method to use. Options include:
        - 'linear', 'polynomial', 'nearest', 'spline'
    - axis: int, default=0
        Axis along which to interpolate.
    - limit: int, optional
        Maximum number of consecutive NaNs to fill.
    - inplace: bool, default=True
        Whether to perform operation inplace.

    Returns:
    - Bamboo: The Bamboo instance with interpolated data.
    """
    self.data.interpolate(method=method, axis=axis, limit=limit, inplace=inplace)
    self.log_changes(f"Interpolated missing values using {method} method.")
    return self

@log
def impute_regression(self, target_column, predictor_columns):
    """
    Impute missing values using regression imputation.

    Parameters:
    - target_column: str
        The column with missing values to be imputed.
    - predictor_columns: list of str
        The columns to use as predictors for the regression model.

    Returns:
    - Bamboo: The Bamboo instance with regression-imputed data.
    """
    from sklearn.linear_model import LinearRegression

    # Split data into rows with missing target and without missing target
    missing_data = self.data[self.data[target_column].isna()]
    complete_data = self.data.dropna(subset=[target_column])

    if complete_data.empty:
        raise ValueError("Not enough data to perform regression imputation.")

    # Train the regression model
    model = LinearRegression()
    model.fit(complete_data[predictor_columns], complete_data[target_column])

    # Predict missing values
    predicted_values = model.predict(missing_data[predictor_columns])

    # Fill missing values with predicted values
    self.data.loc[self.data[target_column].isna(), target_column] = predicted_values

    self.log_changes(f"Imputed missing values in {target_column} using regression on {predictor_columns}.")
    return self

@log
def impute_mice(self, columns=None, max_iter=10, tol=1e-3):
    """
    Impute missing values using Multiple Imputation by Chained Equations (MICE).

    Parameters:
    - columns: list or None, default=None
        A list of columns to apply the imputation to. If None, all numeric columns will be imputed.
    - max_iter: int, default=10
        Maximum number of imputation iterations.
    - tol: float, default=1e-3
        Tolerance to declare convergence.

    Returns:
    - Bamboo: The Bamboo instance with MICE-imputed data.
    """
    if columns is None:
        columns = self.data.select_dtypes(include=[np.number]).columns

    imputer = IterativeImputer(max_iter=max_iter, tol=tol, random_state=0)
    self.data[columns] = pd.DataFrame(imputer.fit_transform(self.data[columns]), columns=columns)

    self.log_changes(f"Imputed missing values using MICE with max_iter={max_iter} and tol={tol}.")
    return self

@log
def impute_em(self, columns=None, rank=2, max_iter=100):
    """
    Impute missing values using Expectation-Maximization (EM) method.

    Parameters:
    - columns: list or None, default=None
        A list of columns to apply the imputation to. If None, all numeric columns will be imputed.
    - rank: int, default=2
        Rank for the low-rank approximation in the EM algorithm.
    - max_iter: int, default=100
        Maximum number of iterations for the EM algorithm.

    Returns:
    - Bamboo: The Bamboo instance with EM-imputed data.
    """
    if columns is None:
        columns = self.data.select_dtypes(include=[np.number]).columns

    # Use IterativeSVD from fancyimpute for EM-like imputation
    imputer = IterativeSVD(rank=rank, max_iters=max_iter)
    self.data[columns] = pd.DataFrame(imputer.fit_transform(self.data[columns]), columns=columns)

    self.log_changes(f"Imputed missing values using EM with rank={rank} and max_iter={max_iter}.")
    return self


Bamboo.impute_missing = impute_missing
Bamboo.drop_missing = drop_missing
Bamboo.fill_with_custom = fill_with_custom
Bamboo.impute_knn = impute_knn
Bamboo.interpolate_missing = interpolate_missing
Bamboo.impute_regression = impute_regression
Bamboo.impute_mice = impute_mice
Bamboo.impute_em = impute_em