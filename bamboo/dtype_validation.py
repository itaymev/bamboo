# bamboo/dtype_validation.py

import pandas as pd
import numpy as np
from bamboo.utils import log
from bamboo.bamboo import Bamboo

@log
def check_dtype_consistency(self, columns=None):
    """
    Check if the columns in the DataFrame have consistent data types.

    Parameters:
    - columns: list or None, default=None
        A list of columns to check for type consistency. If None, all columns will be checked.

    Returns:
    - dict: A dictionary indicating whether each column has consistent data types (True/False).
    """
    if columns is None:
        columns = self.data.columns

    consistency = {col: pd.api.types.infer_dtype(self.data[col]) not in ['mixed', 'mixed-integer'] for col in columns}
    self.log_changes(f"Checked data type consistency for columns: {columns}")
    return consistency

@log
def convert_column_types(self, column_type_dict):
    """
    Convert columns to specified data types.

    Parameters:
    - column_type_dict: dict
        A dictionary where keys are column names and values are target data types.

    Returns:
    - Bamboo: The Bamboo instance with updated column data types.
    """
    for column, dtype in column_type_dict.items():
        self.data[column] = pd.to_numeric(self.data[column], errors='coerce')
        self.log_changes(f"Converted column '{column}' to {dtype}")
    return self

@log
def identify_invalid_types(self, columns=None, expected_dtype=None):
    """
    Identify rows with invalid data types in the specified columns.

    Parameters:
    - columns: list or None, default=None
        A list of columns to check. If None, all columns will be checked.
    - expected_dtype: str or type, default=None
        The expected data type for the columns. If None, checks for each column’s current type.

    Returns:
    - pd.DataFrame: A DataFrame marking rows with invalid data types.
    """
    if columns is None:
        columns = self.data.columns

    invalid_rows = {}
    for col in columns:
        col_dtype = self.data[col].dtype if expected_dtype is None else expected_dtype
        invalid_rows[col] = ~self.data[col].apply(lambda x: pd.api.types.is_dtype_equal(type(x), col_dtype))

    invalid_df = pd.DataFrame(invalid_rows)
    self.log_changes(f"Identified invalid types for columns: {columns}")
    return invalid_df

@log
def enforce_column_types(self, column_type_dict):
    """
    Enforce specific data types for the given columns. Invalid values will be replaced with NaN.

    Parameters:
    - column_type_dict: dict
        A dictionary where keys are column names and values are the expected data types.

    Returns:
    - Bamboo: The Bamboo instance with enforced column types.
    """
    for column, dtype in column_type_dict.items():
        if dtype == 'datetime64[ns]':
            self.data[column] = pd.to_datetime(self.data[column], errors='coerce')
        else:
            self.data[column] = pd.to_numeric(self.data[column], errors='coerce')
        self.log_changes(f"Enforced column '{column}' as {dtype}, invalid entries set to NaN")
    return self


@log
def coerce_data_types(self, column_type_dict):
    """
    Coerce data in specified columns into target types, with error handling for invalid conversions.

    Parameters:
    - column_type_dict: dict
        A dictionary where keys are column names and values are target data types.

    Returns:
    - Bamboo: The Bamboo instance with coerced data types.
    """
    for column, dtype in column_type_dict.items():
        if dtype == 'datetime64[ns]':
            self.data[column] = pd.to_datetime(self.data[column], errors='coerce')
        else:
            self.data[column] = pd.to_numeric(self.data[column], errors='coerce')
        self.log_changes(f"Coerced column '{column}' to {dtype} with error handling")
    return self

@log
def detect_categorical_columns(self):
    """
    Detect categorical columns based on unique value counts or predefined types.

    Returns:
    - list: A list of column names detected as categorical.
    """
    categorical_cols = [col for col in self.data.columns if isinstance(self.data[col].dtype, pd.CategoricalDtype) or self.data[col].nunique() < 10]
    self.log_changes("Detected categorical columns")
    return categorical_cols

@log
def detect_numeric_columns(self):
    """
    Detect numeric columns in the dataset.

    Returns:
    - list: A list of column names detected as numeric (int or float).
    """
    numeric_cols = []
    for col in self.data.columns:
        num_numeric = pd.to_numeric(self.data[col], errors='coerce').notna().sum()
        if num_numeric > (len(self.data[col]) / 2):
            numeric_cols.append(col)
    self.log_changes("Detected numeric columns")
    return numeric_cols


Bamboo.check_dtype_consistency = check_dtype_consistency
Bamboo.convert_column_types = convert_column_types
Bamboo.identify_invalid_types = identify_invalid_types
Bamboo.enforce_column_types = enforce_column_types
Bamboo.coerce_data_types = coerce_data_types
Bamboo.detect_categorical_columns = detect_categorical_columns
Bamboo.detect_numeric_columns = detect_numeric_columns