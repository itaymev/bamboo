# bamboo/validation.py
import pandas as pd
from bamboo.utils import log
from bamboo.bamboo import Bamboo

@log
def validate_missing_data(self, columns=None):
    """
    Validate that there are no missing values in the specified columns.

    Parameters:
    - columns: list or None, default=None
        A list of columns to check for missing data. If None, all columns will be checked.

    Returns:
    - bool: True if validation passes, False if there are missing values.
    """
    if columns is None:
        missing_data = self.data.isnull().any()
    else:
        missing_data = self.data[columns].isnull().any()

    if missing_data.any():
        self.log_changes(f"Validation failed: missing data found in {missing_data[missing_data].index.tolist()}.")
        return False
    self.log_changes("Validation passed: no missing data.")
    return True

@log
def validate_data_types(self, expected_types):
    """
    Validate that the columns match the expected data types.

    Parameters:
    - expected_types: dict
        A dictionary mapping column names to expected data types (e.g., {'age': 'int64', 'name': 'object'}).

    Returns:
    - bool: True if validation passes, False if there are mismatches.
    """
    mismatches = {}
    for col, dtype in expected_types.items():
        if col in self.data.columns and self.data[col].dtype != dtype:
            mismatches[col] = self.data[col].dtype

    if mismatches:
        self.log_changes(f"Validation failed: data type mismatches in columns {mismatches}.")
        return False
    self.log_changes("Validation passed: all columns match expected data types.")
    return True

@log
def validate_value_ranges(self, column, min_value=None, max_value=None):
    """
    Validate that the values in the specified column fall within the defined range.

    Parameters:
    - column: str
        The column to validate.
    - min_value: int or float or None, default=None
        The minimum allowable value. If None, no minimum validation is performed.
    - max_value: int or float or None, default=None
        The maximum allowable value. If None, no maximum validation is performed.

    Returns:
    - bool: True if validation passes, False if values fall outside the range.
    """
    if min_value is not None and (self.data[column] < min_value).any():
        self.log_changes(f"Validation failed: values in column '{column}' below {min_value}.")
        return False

    if max_value is not None and (self.data[column] > max_value).any():
        self.log_changes(f"Validation failed: values in column '{column}' above {max_value}.")
        return False

    self.log_changes(f"Validation passed: values in column '{column}' are within range.")
    return True

@log
def validate_unique_values(self, column):
    """
    Validate that the specified column contains unique values.

    Parameters:
    - column: str
        The column to validate.

    Returns:
    - bool: True if validation passes, False if there are duplicate values.
    """
    duplicates = self.data[column].duplicated().any()

    if duplicates:
        self.log_changes(f"Validation failed: duplicate values found in column '{column}'.")
        return False
    self.log_changes(f"Validation passed: column '{column}' contains unique values.")
    return True

@log
def validate_categories(self, column, valid_categories):
    """
    Validate that the categorical column contains only the specified valid categories.

    Parameters:
    - column: str
        The column to validate.
    - valid_categories: list
        The list of valid categories.

    Returns:
    - bool: True if validation passes, False if invalid categories are found.
    """
    invalid_categories = set(self.data[column].unique()) - set(valid_categories)
    # print(invalid_categories)

    if invalid_categories:
        self.log_changes(f"Validation failed: invalid categories {invalid_categories} found in column '{column}'.")
        return False
    self.log_changes(f"Validation passed: all values in column '{column}' are valid categories.")
    return True

@log
def validate_date_range(self, column, start_date=None, end_date=None):
    """
    Validate that the date column falls within the specified date range.

    Parameters:
    - column: str
        The date column to validate.
    - start_date: str or datetime-like or None, default=None
        The minimum allowable date. If None, no minimum validation is performed.
    - end_date: str or datetime-like or None, default=None
        The maximum allowable date. If None, no maximum validation is performed.

    Returns:
    - bool: True if validation passes, False if dates fall outside the range.
    """
    if start_date is not None and (pd.to_datetime(self.data[column]) < pd.to_datetime(start_date)).any():
        self.log_changes(f"Validation failed: dates in column '{column}' before {start_date}.")
        return False

    if end_date is not None and (pd.to_datetime(self.data[column]) > pd.to_datetime(end_date)).any():
        self.log_changes(f"Validation failed: dates in column '{column}' after {end_date}.")
        return False

    self.log_changes(f"Validation passed: dates in column '{column}' are within range.")
    return True

@log
def custom_validation(self, column, validation_function):
    """
    Apply a custom validation rule to a specified column.

    Parameters:
    - column: str
        The column to validate.
    - validation_function: callable
        A function that takes a Pandas Series and returns True if the validation passes, or False otherwise.

    Returns:
    - bool: True if validation passes, False if the validation fails.
    """
    if not validation_function(self.data[column]):
        self.log_changes(f"Validation failed: custom validation rule failed for column '{column}'.")
        return False
    self.log_changes(f"Validation passed: custom validation rule passed for column '{column}'.")
    return True

Bamboo.validate_missing_data = validate_missing_data
Bamboo.validate_data_types = validate_data_types
Bamboo.validate_value_ranges = validate_value_ranges
Bamboo.validate_unique_values = validate_unique_values
Bamboo.validate_categories = validate_categories
Bamboo.validate_date_range = validate_date_range
Bamboo.custom_validation = custom_validation