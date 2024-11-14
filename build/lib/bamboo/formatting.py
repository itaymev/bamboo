# bamboo/formatting.py
import pandas as pd
import re
from bamboo.utils import log
from bamboo.bamboo import Bamboo

@log
def trim_whitespace(self, columns=None):
    """
    Trim leading, trailing, and excessive internal whitespace from string columns.

    Parameters:
    - columns: list or None, default=None
        A list of columns to apply the formatting to. If None, all string columns will be used.

    Returns:
    - Bamboo: The Bamboo instance with trimmed whitespace in string columns.
    """
    if columns is None:
        columns = self.data.select_dtypes(include=['object']).columns

    for col in columns:
        self.data[col] = self.data[col].str.strip().str.replace(r'\s+', ' ', regex=True)
        self.log_changes(f"Trimmed whitespace in column '{col}'")
    return self

@log
def standardize_case(self, case='lower', columns=None):
    """
    Standardize the case of text in string columns.

    Parameters:
    - case: str, default='lower'
        The case to convert to ('lower', 'upper', 'title').
    - columns: list or None, default=None
        A list of columns to apply the formatting to. If None, all string columns will be used.

    Returns:
    - Bamboo: The Bamboo instance with standardized case in string columns.
    """
    if columns is None:
        columns = self.data.select_dtypes(include=['object']).columns

    for col in columns:
        if case == 'lower':
            self.data[col] = self.data[col].str.lower()
        elif case == 'upper':
            self.data[col] = self.data[col].str.upper()
        elif case == 'title':
            self.data[col] = self.data[col].str.title()
        self.log_changes(f"Standardized case to {case} in column '{col}'")
    return self

@log
def format_dates(self, format='%Y-%m-%d', columns=None):
    """
    Standardize the format of date columns.

    Parameters:
    - format: str, default='%Y-%m-%d'
        The date format to convert to.
    - columns: list or None, default=None
        A list of columns to apply the formatting to. If None, all datetime columns will be used.

    Returns:
    - Bamboo: The Bamboo instance with standardized date formats.
    """
    if columns is None:
        columns = self.data.select_dtypes(include=['datetime']).columns
    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        self.data[col] = pd.to_datetime(self.data[col], errors='coerce').dt.strftime(format)
        self.log_changes(f"Standardized date format in column '{col}' to {format}")
    return self

@log
def remove_special_characters(self, columns=None, chars_to_remove=None):
    """
    Remove or replace special characters in string columns.

    Parameters:
    - columns: list or None, default=None
        A list of columns to apply the formatting to. If None, all string columns will be used.
    - chars_to_remove: str or None, default=None
        A string containing the characters to remove. If None, common special characters will be removed.

    Returns:
    - Bamboo: The Bamboo instance with special characters removed from string columns.
    """
    if columns is None:
        columns = self.data.select_dtypes(include=['object']).columns
    if isinstance(columns, str):
        columns = [columns]


    if chars_to_remove is None:
        chars_to_remove = r'[^\w\s]'  # Default is to remove any non-word and non-space characters

    for col in columns:
        self.data[col] = self.data[col].str.replace(chars_to_remove, '', regex=True)
        self.log_changes(f"Removed special characters in column '{col}'")
    return self

@log
def standardize_currency_format(self, columns=None):
    """
    Standardize currency format in the specified columns.

    Parameters:
    - columns: list or None, default=None
        A list of columns to apply the formatting to. If None, all numeric columns will be used.

    Returns:
    - Bamboo: The Bamboo instance with standardized currency format.
    """
    if columns is None:
        columns = self.data.select_dtypes(include=[float, int]).columns
    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        self.data[col] = self.data[col].apply(lambda x: f"${x:,.2f}")
        self.log_changes(f"Standardized currency format in column '{col}'")
    return self

# Append methods to the Bamboo class
Bamboo.trim_whitespace = trim_whitespace
Bamboo.standardize_case = standardize_case
Bamboo.format_dates = format_dates
Bamboo.remove_special_characters = remove_special_characters
Bamboo.standardize_currency_format = standardize_currency_format