# bamboo/profiling.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bamboo.utils import log
from bamboo.bamboo import Bamboo

@log
def basic_summary(self):
    """
    Generate a basic summary report with descriptive statistics for numeric and categorical columns.

    Returns:
    - pd.DataFrame: A DataFrame with descriptive statistics for each column.
    """
    summary = self.data.describe(include='all')
    self.log_changes("Generated basic summary report.")
    return summary

@log
def missing_data_report(self):
    """
    Generate a report of missing values, showing the percentage and count of missing values per column.

    Returns:
    - pd.DataFrame: A DataFrame showing the count and percentage of missing values per column.
    """
    missing_count = self.data.isnull().sum()
    missing_percentage = (missing_count / len(self.data)) * 100
    missing_report = pd.DataFrame({
        'missing_count': missing_count,
        'missing_percentage': missing_percentage
    }).sort_values(by='missing_count', ascending=False)

    self.log_changes("Generated missing data report.")
    return missing_report

@log
def outliers_report(self, columns=None, method='zscore', threshold=3):
    """
    Generate a report of detected outliers using specified methods.

    Parameters:
    - columns: list or None, default=None
        List of columns to include in the outliers report. If None, all columns will be included.
    - method: str, default='zscore'
        The method to use for outlier detection. Options: 'zscore', 'iqr', 'isolation_forest', etc.
    - threshold: float, default=3
        The threshold for identifying outliers in the case of Z-Score detection.

    Returns:
    - pd.DataFrame: A DataFrame indicating which rows contain outliers.
    """
    data = self.data if columns is None else self.data[columns]

    if method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std(ddof=0))
        #    print(z_scores)
        outliers = z_scores > threshold
    elif method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))
    else:
        raise ValueError("Unsupported method for outliers report.")

    outliers_report = outliers.any(axis=1)
    self.log_changes(f"Generated outliers report using {method} method.")
    print(outliers_report)
    return outliers_report

@log
def distribution_report(self, columns=None):
    """
    Generate a report visualizing the distribution of numeric and categorical columns.

    Parameters:
    - columns: list or None, default=None
        List of columns to include in the distribution report. If None, all columns will be included.

    Returns:
    - None: Displays distribution plots.
    """
    if columns is None:
        columns = self.data.columns

    for col in columns:
        plt.figure(figsize=(10, 6))
        if self.data[col].dtype == 'object':
            sns.countplot(x=col, data=self.data)
        else:
            sns.histplot(self.data[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.show()

    self.log_changes(f"Generated distribution report for columns: {columns}.")
    return None

@log
def correlation_report(self):
    """
    Generate a correlation matrix report for numeric columns.

    Returns:
    - pd.DataFrame: A DataFrame containing the correlation matrix.
    """
    # select only numeric columns
    numeric_columns = self.data.select_dtypes(include=np.number)

    corr_matrix = numeric_columns.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.show()

    self.log_changes("Generated correlation report.")
    return corr_matrix

@log
def data_types_overview(self):
    """
    Generate a summary of data types in the dataset.

    Returns:
    - pd.DataFrame: A DataFrame showing the count and percentage of each data type.
    """
    data_type_counts = self.data.dtypes.value_counts()
    data_type_percentage = (data_type_counts / len(self.data.columns)) * 100
    data_type_report = pd.DataFrame({
        'data_type_count': data_type_counts,
        'data_type_percentage': data_type_percentage
    })

    self.log_changes("Generated data types overview.")
    print(data_type_report)
    return data_type_report

@log
def duplicate_report(self):
    """
    Generate a report identifying duplicate rows in the dataset.

    Returns:
    - pd.DataFrame: A DataFrame showing the count of duplicate rows.
    """
    duplicate_count = self.data.duplicated().sum()
    duplicate_report = pd.DataFrame({
        'duplicate_count': [duplicate_count],
        'total_rows': [len(self.data)],
        'duplicate_percentage': [(duplicate_count / len(self.data)) * 100]
    })

    self.log_changes("Generated duplicate report.")
    return duplicate_report


Bamboo.basic_summary = basic_summary
Bamboo.missing_data_report = missing_data_report
Bamboo.outliers_report = outliers_report
Bamboo.distribution_report = distribution_report
Bamboo.correlation_report = correlation_report
Bamboo.data_types_overview = data_types_overview
Bamboo.duplicate_report = duplicate_report
