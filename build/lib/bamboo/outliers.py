# bamboo/outliers.py
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from bamboo.utils import log

from bamboo.bamboo import Bamboo

@log
def detect_outliers_zscore(self, columns=None, threshold=3):
    """
    Detect outliers using the Z-Score method.

    Parameters:
    - columns: list or None, default=None
        A list of columns to detect outliers in. If None, all numeric columns will be used.
    - threshold: float, default=3
        The Z-Score threshold to identify outliers. Values with a Z-Score higher than the threshold are considered outliers.

    Returns:
    - pd.DataFrame: A DataFrame marking outliers with True/False.
    """
    if columns is None:
        columns = self.data.select_dtypes(include=[np.number]).columns

    z_scores = np.abs((self.data[columns] - self.data[columns].mean()) / self.data[columns].std())
    outliers = z_scores > threshold

    # Mark rows with any outliers in specified columns
    outlier_rows = outliers.any(axis=1)
    self.data['outliers'] = outlier_rows

    self.log_changes(f"Detected outliers using Z-Score with threshold={threshold}.")
    return outliers

@log
def detect_outliers_iqr(self, columns=None, multiplier=1.5):
    """
    Detect outliers using the Interquartile Range (IQR) method.

    Parameters:
    - columns: list or None, default=None
        A list of columns to detect outliers in. If None, all numeric columns will be used.
    - multiplier: float, default=1.5
        The multiplier for the IQR to determine the outlier range. 

    Returns:
    - pd.DataFrame: A DataFrame marking outliers with True/False.
    """
    if columns is None:
        columns = self.data.select_dtypes(include=[np.number]).columns

    Q1 = self.data[columns].quantile(0.25)
    Q3 = self.data[columns].quantile(0.75)
    IQR = Q3 - Q1
    outliers = (self.data[columns] < (Q1 - multiplier * IQR)) | (self.data[columns] > (Q3 + multiplier * IQR))

    self.log_changes(f"Detected outliers using IQR with multiplier={multiplier}.")
    return outliers

@log
def detect_outliers_isolation_forest(self, contamination=0.05, random_state=None, columns=None, n_estimators=100):
    """
    Detect outliers using Isolation Forest.

    Parameters:
    - contamination: float, default=0.05
        The proportion of outliers in the dataset. The higher the contamination, the more outliers will be detected.
    - random_state: int or None, default=None
        Controls the randomness of the estimator for reproducibility.
    - columns: list or None, default=None
        A list of columns to apply the outlier detection to. If None, all numeric columns will be used.

    Returns:
    - pd.DataFrame: A DataFrame marking outliers with True/False.
    """
    if columns is None:
        columns = self.data.select_dtypes(include=[np.number]).columns

    # Isolation Forest model
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=n_estimators)
    self.data['outliers'] = iso_forest.fit_predict(self.data[columns])

    # Convert the -1 values (outliers) to True and 1 values (inliers) to False
    self.data['outliers'] = self.data['outliers'].apply(lambda x: True if x == -1 else False)

    self.log_changes(f"Detected outliers using Isolation Forest with contamination={contamination}.")
    return self.data[['outliers']]

@log
def detect_outliers_dbscan(self, eps=0.5, min_samples=5, columns=None):
    """
    Detect outliers using DBSCAN (Density-Based Spatial Clustering).

    Parameters:
    - eps: float, default=0.5
        The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples: int, default=5
        The number of samples in a neighborhood for a point to be considered a core point.
    - columns: list or None, default=None
        A list of columns to apply the outlier detection to. If None, all numeric columns will be used.

    Returns:
    - pd.DataFrame: A DataFrame marking outliers with True/False.
    """
    if columns is None:
        columns = self.data.select_dtypes(include=[np.number]).columns

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(self.data[columns])
    
    # Mark points as outliers where cluster value is -1
    self.data['outliers'] = clusters == -1
    self.log_changes(f"Detected outliers using DBSCAN with eps={eps} and min_samples={min_samples}.")
    return self.data[['outliers']]

@log
def detect_outliers_modified_zscore(self, threshold=3.5, columns=None):
    """
    Detect outliers using the Modified Z-Score method.

    Parameters:
    - threshold: float, default=3.5
        The threshold for identifying outliers. Values with a modified Z-Score higher than this threshold are considered outliers.
    - columns: list or None, default=None
        A list of columns to apply the outlier detection to. If None, all numeric columns will be used.

    Returns:
    - pd.DataFrame: A DataFrame marking outliers with True/False.
    """
    if columns is None:
        columns = self.data.select_dtypes(include=[np.number]).columns

    median = self.data[columns].median()
    mad = np.median(np.abs(self.data[columns] - median), axis=0)  # Median Absolute Deviation (MAD)
    modified_z_scores = 0.67449075947 * (self.data[columns] - median) / mad

    outliers = np.abs(modified_z_scores) > threshold
    self.data['outliers'] = outliers.any(axis=1)
    self.log_changes(f"Detected outliers using Modified Z-Score with threshold={threshold}.")
    return self.data[['outliers']]

@log
def detect_outliers_robust_covariance(self, contamination=0.1, columns=None):
    """
    Detect outliers using robust covariance estimation.

    Parameters:
    - contamination: float, default=0.1
        The proportion of the dataset assumed to be outliers.
    - columns: list or None, default=None
        A list of columns to apply the outlier detection to. If None, all numeric columns will be used.

    Returns:
    - pd.DataFrame: A DataFrame marking outliers with True/False.
    """
    if columns is None:
        columns = self.data.select_dtypes(include=[np.number]).columns

    robust_cov = EllipticEnvelope(contamination=contamination)
    outliers = robust_cov.fit_predict(self.data[columns])

    # Mark points as outliers where the prediction is -1
    self.data['outliers'] = outliers == -1
    self.log_changes(f"Detected outliers using robust covariance with contamination={contamination}.")
    return self.data[['outliers']]

@log
def detect_outliers_lof(self, n_neighbors=20, contamination=0.1, columns=None):
    """
    Detect outliers using Local Outlier Factor (LOF).

    Parameters:
    - n_neighbors: int, default=20
        The number of neighbors to use for calculating the local density.
    - contamination: float, default=0.1
        The proportion of outliers in the dataset.
    - columns: list or None, default=None
        A list of columns to apply the outlier detection to. If None, all numeric columns will be used.

    Returns:
    - pd.DataFrame: A DataFrame marking outliers with True/False.
    """
    if columns is None:
        columns = self.data.select_dtypes(include=[np.number]).columns

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    outliers = lof.fit_predict(self.data[columns])

    # Mark points as outliers where prediction is -1
    self.data['outliers'] = outliers == -1
    self.log_changes(f"Detected outliers using LOF with n_neighbors={n_neighbors} and contamination={contamination}.")
    return self.data[['outliers']]

@log
def remove_outliers(self, method='zscore', columns=None, **kwargs):
    """
    Remove rows that contain outliers in the specified columns.

    Parameters:
    - method: str, default='zscore'
        The method to use for outlier detection. Options: 'zscore', 'iqr'.
    - columns: list or None, default=None
        List of columns to check for outliers. If None, all numeric columns are used.

    Returns:
    - Bamboo: The Bamboo instance with rows containing outliers removed.
    """
    if columns is None:
        columns = self.data.select_dtypes(include=[np.number]).columns

    if method == 'zscore':
        outliers = self.detect_outliers_zscore(columns=columns, **kwargs)
    elif method == 'iqr':
        outliers = self.detect_outliers_iqr(columns=columns, **kwargs)
    else:
        raise ValueError("Unsupported outlier detection method!")

    # Remove rows where outliers are detected
    self.data = self.data[~outliers.any(axis=1)]
    self.log_changes(f"Removed rows with outliers in columns {columns} using {method} method.")
    return self

@log
def remove_outliers_isolation_forest(self, contamination=0.05, random_state=None, columns=None, n_estimators=100):
    """
    Remove outliers detected using Isolation Forest.

    Parameters:
    - contamination: float, default=0.05
        The proportion of outliers in the dataset.
    - random_state: int or None, default=None
        Controls the randomness of the estimator.
    - columns: list or None, default=None
        A list of columns to apply the outlier detection to. If None, all numeric columns will be used.

    Returns:
    - Bamboo: The Bamboo instance with outliers removed.
    """
    outliers = self.detect_outliers_isolation_forest(contamination=contamination, random_state=random_state, columns=columns, n_estimators=n_estimators)
    self.data = self.data[~self.data['outliers']]
    self.data.drop(columns='outliers', inplace=True)

    self.log_changes(f"Removed outliers using Isolation Forest with contamination={contamination}.")
    return self

@log
def remove_outliers_dbscan(self, eps=0.5, min_samples=5, columns=None):
    """
    Remove outliers detected using DBSCAN (Density-Based Spatial Clustering).

    Parameters:
    - eps: float, default=0.5
        The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples: int, default=5
        The number of samples in a neighborhood for a point to be considered a core point.
    - columns: list or None, default=None
        A list of columns to apply the outlier detection to. If None, all numeric columns will be used.

    Returns:
    - Bamboo: The Bamboo instance with outliers removed.
    """
    outliers = self.detect_outliers_dbscan(eps=eps, min_samples=min_samples, columns=columns)
    self.data = self.data[~self.data['outliers']]
    self.data.drop(columns='outliers', inplace=True)

    self.log_changes(f"Removed outliers using DBSCAN with eps={eps} and min_samples={min_samples}.")
    return self

@log
def remove_outliers_modified_zscore(self, threshold=3.5, columns=None):
    """
    Remove outliers detected using the Modified Z-Score method.

    Parameters:
    - threshold: float, default=3.5
        The threshold for identifying outliers.
    - columns: list or None, default=None
        A list of columns to apply the outlier detection to. If None, all numeric columns will be used.

    Returns:
    - Bamboo: The Bamboo instance with outliers removed.
    """
    outliers = self.detect_outliers_modified_zscore(threshold=threshold, columns=columns)
    self.data = self.data[~self.data['outliers']]
    self.data.drop(columns='outliers', inplace=True)

    self.log_changes(f"Removed outliers using Modified Z-Score with threshold={threshold}.")
    return self

@log
def remove_outliers_robust_covariance(self, contamination=0.1, columns=None):
    """
    Remove outliers detected using Robust Covariance Estimation.

    Parameters:
    - contamination: float, default=0.1
        The proportion of the dataset assumed to be outliers.
    - columns: list or None, default=None
        A list of columns to apply the outlier detection to. If None, all numeric columns will be used.

    Returns:
    - Bamboo: The Bamboo instance with outliers removed.
    """
    outliers = self.detect_outliers_robust_covariance(contamination=contamination, columns=columns)
    self.data = self.data[~self.data['outliers']]
    self.data.drop(columns='outliers', inplace=True)

    self.log_changes(f"Removed outliers using Robust Covariance with contamination={contamination}.")
    return self

@log
def remove_outliers_lof(self, n_neighbors=20, contamination=0.1, columns=None):
    """
    Remove outliers detected using Local Outlier Factor (LOF).

    Parameters:
    - n_neighbors: int, default=20
        The number of neighbors to use for calculating the local density.
    - contamination: float, default=0.1
        The proportion of outliers in the dataset.
    - columns: list or None, default=None
        A list of columns to apply the outlier detection to. If None, all numeric columns will be used.

    Returns:
    - Bamboo: The Bamboo instance with outliers removed.
    """
    outliers = self.detect_outliers_lof(n_neighbors=n_neighbors, contamination=contamination, columns=columns)
    self.data = self.data[~self.data['outliers']]
    self.data.drop(columns='outliers', inplace=True)

    self.log_changes(f"Removed outliers using LOF with n_neighbors={n_neighbors} and contamination={contamination}.")
    return self

@log
def clip_outliers(self, method='zscore', clip_value=None, **kwargs):
    """
    Clips outliers and replaces outlier data with clip_value, or mean.

    Parameters:
    - method: str, default='zscore'
        The method to use for outlier detection. Options: 'zscore', 'iqr'.
    - cap_value: float or None, default=None
        The value to cap outliers at. If None, outliers will be capped at the threshold.

    Returns:
    - Bamboo: The Bamboo instance with outliers capped.
    """
    if method == 'zscore':
        outliers = self.detect_outliers_zscore(**kwargs)
    elif method == 'iqr':
        outliers = self.detect_outliers_iqr(**kwargs)
    else:
        raise ValueError("Unsupported outlier detection method!")

    if clip_value is None:
        clip_value = self.data.mean()

    self.data[outliers] = clip_value
    self.log_changes(f"Capped outliers using {method} method.")
    return self

@log
def cap_outliers(self, method='zscore', lower_cap=None, upper_cap=None, columns=None, **kwargs):
    """
    Cap outliers by setting them to a defined upper or lower limit in specific columns.

    Parameters:
    - method: str, default='zscore'
        The method to use for outlier detection. Options: 'zscore', 'iqr'.
    - lower_cap: float or None, default=None
        The value to cap lower-bound outliers at. If None (undefined), the lower threshold will be used.
    - upper_cap: float or None, default=None
        The value to cap upper-bound outliers at. If None (undefined), the upper threshold will be used.
    - columns: list or None, default=None
        List of columns to apply capping to. If None, all numeric columns are used.

    Returns:
    - Bamboo: The Bamboo instance with outliers capped at the specified limits.
    """
    if columns is None:
        columns = self.data.select_dtypes(include=[np.number]).columns

    if method == 'zscore':
        z_scores = (self.data[columns] - self.data[columns].mean()) / self.data[columns].std()
        outliers = np.abs(z_scores) > kwargs.get('threshold', 3)
    elif method == 'iqr':
        Q1 = self.data[columns].quantile(0.25)
        Q3 = self.data[columns].quantile(0.75)
        IQR = Q3 - Q1
        outliers = (self.data[columns] < (Q1 - kwargs.get('multiplier', 1.5) * IQR)) | \
                   (self.data[columns] > (Q3 + kwargs.get('multiplier', 1.5) * IQR))
    else:
        raise ValueError("Unsupported outlier detection method!")

    for col in columns:
        if lower_cap is not None:
            self.data[col] = np.where(outliers[col] & (self.data[col] < lower_cap), lower_cap, self.data[col])
        if upper_cap is not None:
            self.data[col] = np.where(outliers[col] & (self.data[col] > upper_cap), upper_cap, self.data[col])

    self.log_changes(f"Capped outliers in columns {columns} using {method} method with lower_cap={lower_cap}, upper_cap={upper_cap}.")
    return self

Bamboo.detect_outliers_zscore = detect_outliers_zscore
Bamboo.detect_outliers_iqr = detect_outliers_iqr
Bamboo.detect_outliers_isolation_forest = detect_outliers_isolation_forest
Bamboo.detect_outliers_dbscan = detect_outliers_dbscan
Bamboo.detect_outliers_modified_zscore = detect_outliers_modified_zscore
Bamboo.detect_outliers_robust_covariance = detect_outliers_robust_covariance
Bamboo.detect_outliers_lof = detect_outliers_lof
Bamboo.remove_outliers = remove_outliers
Bamboo.remove_outliers_isolation_forest = remove_outliers_isolation_forest
Bamboo.remove_outliers_dbscan = remove_outliers_dbscan
Bamboo.remove_outliers_modified_zscore = remove_outliers_modified_zscore
Bamboo.remove_outliers_robust_covariance = remove_outliers_robust_covariance
Bamboo.remove_outliers_lof = remove_outliers_lof
Bamboo.clip_outliers = clip_outliers
Bamboo.cap_outliers = cap_outliers