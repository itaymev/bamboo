# bamboo/duplicates.py
import pandas as pd
from bamboo.utils import log
from bamboo.bamboo import Bamboo
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

@log
def identify_duplicates(self, subset=None):
    """
    Identify duplicate rows in the dataset.

    Parameters:
    - subset: list or None, default=None
        List of columns to check for duplicates. If None, all columns are used.

    Returns:
    - pd.DataFrame: A DataFrame containing the duplicate rows.
    """
    duplicates = self.data[self.data.duplicated(subset=subset, keep=False)]
    self.log_changes(f"Identified duplicates in columns: {subset if subset else 'all columns'}.")
    return duplicates

@log
def drop_duplicates(self, subset=None, keep='first'):
    """
    Drop duplicate rows from the dataset.

    Parameters:
    - subset: list or None, default=None
        List of columns to check for duplicates. If None, all columns are used.
    - keep: {'first', 'last', False}, default='first'
        Determines which duplicates to keep. 'first' keeps the first occurrence, 'last' keeps the last, and False drops all duplicates.

    Returns:
    - Bamboo: The Bamboo instance with duplicates removed.
    """
    self.data.drop_duplicates(subset=subset, keep=keep, inplace=True)
    self.log_changes(f"Dropped duplicates in columns: {subset if subset else 'all columns'}, keeping {keep}.")
    return self

@log
def mark_duplicates(self, subset=None, keep='first', marker_column='is_duplicate'):
    """
    Mark duplicate rows with a new column without removing them.

    Parameters:
    - subset: list or None, default=None
        List of columns to check for duplicates. If None, all columns are used.
    - keep: {'first', 'last', False}, default='first'
        Determines which duplicates to keep and mark.
    - marker_column: str, default='is_duplicate'
        Name of the column to mark duplicates.

    Returns:
    - Bamboo: The Bamboo instance with duplicates marked.
    """
    self.data[marker_column] = self.data.duplicated(subset=subset, keep=keep)
    self.log_changes(f"Marked duplicates in column '{marker_column}' for subset: {subset if subset else 'all columns'}, keeping {keep}.")
    return self

@log
def subset_duplicates(self, subset):
    """
    Identify duplicates based on a subset of columns.

    Parameters:
    - subset: list
        List of columns to check for duplicates.

    Returns:
    - pd.DataFrame: A DataFrame containing the duplicate rows based on the subset of columns.
    """
    duplicates = self.data[self.data.duplicated(subset=subset, keep=False)]
    self.log_changes(f"Identified duplicates based on subset: {subset}.")
    return duplicates

@log
def merge_duplicates(self, subset=None, keep_method='most_frequent'):
    """
    Merge duplicate rows by keeping the most frequent or most recent values for specified columns.

    Parameters:
    - subset: list or None, default=None
        List of columns to check for duplicates. If None, all columns are used.
    - keep_method: str, default='most_frequent'
        Method to resolve duplicate values. Options are 'most_frequent' or 'most_recent'.

    Returns:
    - Bamboo: The Bamboo instance with duplicates merged.
    """
    if keep_method == 'most_frequent':
        self.data = self.data.groupby(subset).agg(lambda x: x.value_counts().index[0]).reset_index()
    elif keep_method == 'most_recent':
        self.data = self.data.groupby(subset).agg(lambda x: x.iloc[-1]).reset_index()
    else:
        raise ValueError("Unsupported keep_method. Use 'most_frequent' or 'most_recent'.")
    
    self.log_changes(f"Merged duplicates based on {subset if subset else 'all columns'} using '{keep_method}' method.")
    return self

@log
def handle_near_duplicates(self, column, similarity_threshold=0.8):
    """
    Handle near-duplicates based on text similarity for the specified column using fuzzy matching.

    Parameters:
    - column: str
        The name of the column to check for near-duplicates.
    - similarity_threshold: float, default=0.8
        The similarity threshold (0 to 1) to determine if two rows are near-duplicates.

    Returns:
    - Bamboo: The Bamboo instance with near-duplicates handled.
    """

    unique_values = self.data[column].unique()
    merged_values = {}

    for val in unique_values:
        matches = process.extract(val, unique_values, limit=None)
        for match, score in matches:
            if score >= similarity_threshold * 100:
                merged_values[match] = val

    self.data[column] = self.data[column].replace(merged_values)
    self.log_changes(f"Handled near-duplicates in column '{column}' with similarity threshold={similarity_threshold}.")
    return self


Bamboo.identify_duplicates = identify_duplicates
Bamboo.drop_duplicates = drop_duplicates
Bamboo.mark_duplicates = mark_duplicates
Bamboo.subset_duplicates = subset_duplicates
Bamboo.merge_duplicates = merge_duplicates
Bamboo.handle_near_duplicates = handle_near_duplicates