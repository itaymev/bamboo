# bamboo/dates.py
import pandas as pd
from bamboo.utils import log
from bamboo.bamboo import Bamboo

@log
def convert_to_datetime(self, columns=None, format=None):
    """
    Convert specified columns to datetime format.

    Parameters:
    - columns: list or None, default=None
        A list of columns to convert. If None, all object-type columns that can be parsed as dates will be converted.
    - format: str or None, default=None
        The format to use for parsing the dates. If None, Pandas will infer the format.

    Returns:
    - Bamboo: The Bamboo instance with converted datetime columns.
    """
    if columns is None:
        columns = self.data.select_dtypes(include=['object']).columns

    for col in columns:
        self.data[col] = pd.to_datetime(self.data[col], format=format, errors='coerce')
        self.log_changes(f"Converted column '{col}' to datetime.")
    return self

@log
def extract_date_parts(self, column, parts=None):
    """
    Extract specific date parts (year, month, day, etc.) from a datetime column.

    Parameters:
    - column: str
        The column to extract date parts from.
    - parts: list of str or None, default=None
        The date parts to extract. Options are 'year', 'month', 'day', 'weekday', 'hour', etc.
        If None, all parts will be extracted.

    Returns:
    - Bamboo: The Bamboo instance with extracted date parts.
    """
    date_parts = ['year', 'month', 'day', 'weekday', 'hour', 'minute', 'second']
    if parts is None:
        parts = date_parts

    for part in parts:
        self.data[f"{column}_{part}"] = getattr(self.data[column].dt, part)
        self.log_changes(f"Extracted '{part}' from column '{column}'.")
    return self

@log
def create_date_range(self, start, end, freq='D', column_name='date_range'):
    """
    Create a new column with a date range.

    Parameters:
    - start: str or datetime-like
        The start date for the range.
    - end: str or datetime-like
        The end date for the range.
    - freq: str, default='D'
        The frequency of the range. Options include 'D' (daily), 'W' (weekly), 'M' (monthly), etc.
    - column_name: str, default='date_range'
        The name of the new column to store the date range.

    Returns:
    - Bamboo: The Bamboo instance with the new date range column.
    """
    self.data[column_name] = pd.date_range(start=start, end=end, freq=freq)
    self.log_changes(f"Created date range from {start} to {end} with frequency '{freq}' as column '{column_name}'.")
    return self

@log
def handle_invalid_dates(self, columns=None, fill_value=None):
    """
    Handle invalid dates in the specified columns by removing them or filling with a default value.

    Parameters:
    - columns: list or None, default=None
        A list of columns to check. If None, all datetime columns will be checked.
    - fill_value: str or datetime-like, default=None
        The value to fill invalid dates with. If None, invalid dates will be removed.

    Returns:
    - Bamboo: The Bamboo instance with invalid dates handled.
    """
    if columns is None:
        columns = self.data.select_dtypes(include=['datetime']).columns

    for col in columns:
        if fill_value is None:
            self.data = self.data[self.data[col].notna()]
            self.log_changes(f"Removed rows with invalid dates in column '{col}'.")
        else:
            self.data[col] = self.data[col].fillna(fill_value)
            self.log_changes(f"Filled invalid dates in column '{col}' with '{fill_value}'.")
    return self

@log
def calculate_date_differences(self, start_column, end_column, unit='days', new_column='date_diff'):
    """
    Calculate the difference between two date columns.

    Parameters:
    - start_column: str
        The name of the start date column.
    - end_column: str
        The name of the end date column.
    - unit: str, default='days'
        The unit to calculate the difference in. Options are 'days', 'weeks', 'months', 'years'.
    - new_column: str, default='date_diff'
        The name of the new column to store the date differences.

    Returns:
    - Bamboo: The Bamboo instance with the calculated date differences.
    """
    if unit == 'days':
        self.data[new_column] = (self.data[end_column] - self.data[start_column]).dt.days
    elif unit == 'weeks':
        self.data[new_column] = (self.data[end_column] - self.data[start_column]).dt.days / 7
    elif unit == 'months':
        self.data[new_column] = (self.data[end_column] - self.data[start_column]).dt.days / 30
    elif unit == 'years':
        self.data[new_column] = (self.data[end_column] - self.data[start_column]).dt.days / 365
    else:
        raise ValueError("Invalid unit! Use 'days', 'weeks', 'months', or 'years'.")
    
    self.log_changes(f"Calculated date differences between '{start_column}' and '{end_column}' in '{unit}'.")
    return self

@log
def shift_dates(self, columns=None, periods=1, freq='D'):
    """
    Shift dates forward or backward by a specified number of periods.

    Parameters:
    - columns: list or None, default=None
        A list of columns to apply the shift to. If None, all datetime columns will be shifted.
    - periods: int, default=1
        The number of periods to shift the dates by. Positive values shift forward, negative values shift backward.
    - freq: str, default='D'
        The frequency to shift by. Options include 'D' (days), 'W' (weeks), 'M' (months), etc.

    Returns:
    - Bamboo: The Bamboo instance with shifted dates.
    """
    if columns is None:
        columns = self.data.select_dtypes(include=['datetime']).columns

    for col in columns:
        self.data[col] = self.data[col] + pd.to_timedelta(periods, unit=freq)
        self.log_changes(f"Shifted column '{col}' by {periods} {freq}.")
    return self

@log
def round_dates(self, columns=None, freq='D'):
    """
    Round dates to the nearest specified frequency (e.g., year, month, day).

    Parameters:
    - columns: list or None, default=None
        A list of columns to apply the rounding to. If None, all datetime columns will be rounded.
    - freq: str, default='D'
        The frequency to round to. Options include 'Y' (year), 'M' (month), 'D' (day), etc.

    Returns:
    - Bamboo: The Bamboo instance with rounded dates.
    """
    if columns is None:
        columns = self.data.select_dtypes(include=['datetime']).columns

    for col in columns:
        self.data[col] = self.data[col].dt.round(freq)
        self.log_changes(f"Rounded dates in column '{col}' to '{freq}'.")
    return self

@log
def detect_time_gaps(self, column, freq='D'):
    """
    Detect gaps or missing intervals in a sequence of dates.

    Parameters:
    - column: str
        The column containing the dates.
    - freq: str, default='D'
        The expected frequency of the dates. Options include 'D' (daily), 'W' (weekly), 'M' (monthly), etc.

    Returns:
    - pd.DataFrame: A DataFrame containing the missing date intervals.
    """
    date_range = pd.date_range(start=self.data[column].min(), end=self.data[column].max(), freq=freq)
    missing_dates = date_range.difference(self.data[column])
    
    self.log_changes(f"Detected missing dates in column '{column}' with frequency '{freq}'.")
    return pd.DataFrame(missing_dates, columns=['missing_dates'])


# Add the date-related methods to the Bamboo class
Bamboo.convert_to_datetime = convert_to_datetime
Bamboo.extract_date_parts = extract_date_parts
Bamboo.create_date_range = create_date_range
Bamboo.handle_invalid_dates = handle_invalid_dates
Bamboo.calculate_date_differences = calculate_date_differences
Bamboo.shift_dates = shift_dates
Bamboo.round_dates = round_dates
Bamboo.detect_time_gaps = detect_time_gaps