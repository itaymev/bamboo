# bamboo/categorical.py
import pandas as pd
from bamboo.utils import log
from bamboo.bamboo import Bamboo

@log
def convert_to_categorical(self, columns=None):
    """
    Convert specified columns to categorical data type.

    Parameters:
    - columns: list or None, default=None
        A list of columns to convert. If None, all object-type columns will be converted.

    Returns:
    - Bamboo: The Bamboo instance with converted categorical columns.
    """
    if columns is None:
        columns = self.data.select_dtypes(include=['object']).columns

    for col in columns:
        self.data[col] = self.data[col].astype('category')
        self.log_changes(f"Converted column '{col}' to categorical.")
    return self

@log
def get_unique_categories(self, column):
    """
    Get the unique categories in a categorical column.

    Parameters:
    - column: str
        The name of the column to retrieve unique categories from.

    Returns:
    - list: A list of unique categories.
    """
    if column not in self.data.select_dtypes(include=['category']).columns:
        raise ValueError(f"Column '{column}' is not a categorical column.")
    
    unique_categories = self.data[column].cat.categories.tolist()
    self.log_changes(f"Retrieved unique categories from column '{column}'.")
    return unique_categories

@log
def encode_categorical(self, columns=None, method='onehot'):
    """
    Encode categorical variables using one-hot encoding or label encoding.

    Parameters:
    - columns: list or None, default=None
        A list of columns to encode. If None, all categorical columns will be encoded.
    - method: str, default='onehot'
        The encoding method to use. Options are 'onehot' for one-hot encoding and 'label' for label encoding.

    Returns:
    - Bamboo: The Bamboo instance with encoded categorical columns.
    """
    if columns is None:
        columns = self.data.select_dtypes(include=['category']).columns

    if method == 'onehot':
        self.data = pd.get_dummies(self.data, columns=columns, drop_first=False)
        self.log_changes(f"Applied one-hot encoding to columns: {columns}")
    elif method == 'label':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in columns:
            self.data[col] = le.fit_transform(self.data[col])
        self.log_changes(f"Applied label encoding to columns: {columns}")
    else:
        raise ValueError("Unsupported encoding method. Use 'onehot' or 'label'.")
    
    return self

@log
def handle_missing_categories(self, column, fill_value=None):
    """
    Handle missing categories by filling with a default or specified value.

    Parameters:
    - column: str
        The categorical column to handle missing values in.
    - fill_value: str or None, default=None
        The value to fill missing categories with. If None, fills with the column's most frequent category.

    Returns:
    - Bamboo: The Bamboo instance with handled missing categories.
    """
    if fill_value is None:
        fill_value = self.data[column].mode()[0]

    self.data[column] = self.data[column].cat.add_categories([fill_value]).fillna(fill_value)
    self.log_changes(f"Filled missing categories in column '{column}' with '{fill_value}'.")
    return self

@log
def encode_frequency(self, columns=None):
    """
    Encode categorical variables based on their frequency.

    Parameters:
    - columns: list or None, default=None
        A list of columns to apply frequency encoding. If None, all categorical columns will be encoded.

    Returns:
    - Bamboo: The Bamboo instance with frequency-encoded categorical columns.
    """
    if columns is None:
        columns = self.data.select_dtypes(include=['category']).columns

    for col in columns:
        freq_map = self.data[col].value_counts(normalize=True)
        self.data[col] = self.data[col].map(freq_map)
        self.log_changes(f"Encoded column '{col}' based on frequency.")
    return self

@log
def map_categories(self, column, mapping_dict):
    """
    Map existing categories to new values based on a mapping dictionary.

    Parameters:
    - column: str
        The name of the categorical column to map.
    - mapping_dict: dict
        A dictionary defining the category mappings.

    Returns:
    - Bamboo: The Bamboo instance with mapped categories.
    """
    if column not in self.data.select_dtypes(include=['category']).columns:
        raise ValueError(f"Column '{column}' is not a categorical column.")
    
    self.data[column] = self.data[column].map(mapping_dict)
    self.log_changes(f"Mapped categories in column '{column}' with {mapping_dict}.")
    return self

@log
def detect_rare_categories(self, column, threshold=0.01):
    """
    Detect rare categories in a categorical column, i.e., those that appear with a frequency below the given threshold.

    Parameters:
    - column: str
        The name of the categorical column to check.
    - threshold: float, default=0.01
        The frequency threshold below which a category is considered rare.

    Returns:
    - list: A list of rare categories.
    """
    if column not in self.data.select_dtypes(include=['category']).columns:
        raise ValueError(f"Column '{column}' is not a categorical column.")

    value_counts = self.data[column].value_counts(normalize=True, dropna=True)
    rare_categories = value_counts[value_counts < threshold].index.tolist()
    self.log_changes(f"Detected rare categories in column '{column}': {rare_categories}")
    return rare_categories

@log
def replace_rare_categories(self, column, threshold=0.01, replacement='Other'):
    """
    Replace rare categories in a categorical column with a default or specified value.

    Parameters:
    - column: str
        The name of the categorical column to check.
    - threshold: float, default=0.01
        The frequency threshold below which a category is considered rare.
    - replacement: str, default='Other'
        The value to replace rare categories with.

    Returns:
    - Bamboo: The Bamboo instance with rare categories replaced.
    """
    rare_categories = self.detect_rare_categories(column, threshold)
    
    if replacement not in self.data[column].cat.categories:
        self.data[column] = self.data[column].cat.add_categories([replacement])
    print(rare_categories, replacement)
    self.data[column] = self.data[column].replace(rare_categories, replacement)
    self.data[column] = self.data[column].cat.remove_unused_categories()
    self.log_changes(f"Replaced rare categories in column '{column}' with '{replacement}'.")
    
    return self

Bamboo.convert_to_categorical = convert_to_categorical
Bamboo.get_unique_categories = get_unique_categories
Bamboo.encode_categorical = encode_categorical
Bamboo.handle_missing_categories = handle_missing_categories
Bamboo.encode_frequency = encode_frequency
Bamboo.map_categories = map_categories
Bamboo.detect_rare_categories = detect_rare_categories
Bamboo.replace_rare_categories = replace_rare_categories