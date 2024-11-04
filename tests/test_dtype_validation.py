# tests/test_dtype_validation.py

import pytest
import pandas as pd
import numpy as np
from bamboo.bamboo import Bamboo

@pytest.fixture
def sample_data():
    """Fixture to provide sample data with various types for testing."""
    return pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'Derek', 123],  # One invalid type (int)
        'age': [25, 'thirty', 35, 40, None],  # One invalid type (str)
        'salary': [50000.00, None, 60000.00, 'abcd', 70000.00],  # One invalid type (str)
        'join_date': ['2020-01-01', '2020-02-15', None, 'Invalid Date', '2020-03-10']
    })

def test_check_dtype_consistency(sample_data):
    """Test checking the consistency of data types in columns."""
    bamboo = Bamboo(sample_data)
    consistency = bamboo.check_dtype_consistency()
    
    # 'name' and 'age' should be inconsistent due to the presence of mixed types
    assert consistency['name'] is False
    assert consistency['age'] is False
    assert consistency['salary'] is False
    assert consistency['join_date'] is True  # All are strings, even if invalid
    
    print(consistency)

def test_convert_column_types(sample_data):
    """Test converting columns to specified data types."""
    bamboo = Bamboo(sample_data)
    bamboo.convert_column_types({'age': 'float64', 'salary': 'float64'})
    
    assert pd.api.types.is_float_dtype(bamboo.get_data()['age'])
    assert pd.api.types.is_float_dtype(bamboo.get_data()['salary'])
    
    print(bamboo.get_data())

def test_identify_invalid_types(sample_data):
    """Test identifying rows with invalid data types."""
    bamboo = Bamboo(sample_data)
    invalid_types = bamboo.identify_invalid_types(columns=['age'], expected_dtype='int')

    # Expect invalid entries for 'age' where 'thirty' is present
    assert invalid_types['age'][0] == False
    assert invalid_types['age'][1] == True

    invalid_types = bamboo.identify_invalid_types(columns=['salary'], expected_dtype='float64')

    assert invalid_types['salary'][0] == False
    assert invalid_types['salary'][3] == True  # '100000' is a string
    
    print(invalid_types)

def test_enforce_column_types(sample_data):
    """Test enforcing specific data types, invalid values should be coerced to NaN."""
    bamboo = Bamboo(sample_data)
    bamboo.enforce_column_types({'age': 'float64', 'salary': 'float64'})

    assert pd.isna(bamboo.get_data()['age'][1]) # 'age' should have NaN where the string 'thirty' was
    assert pd.isna(bamboo.get_data()['salary'][3]) # 'salary' should have NaN where 'abcd' (string) was
    
    print(bamboo.get_data())

def test_coerce_data_types(sample_data):
    """Test coercing data into target types with error handling."""
    bamboo = Bamboo(sample_data)
    bamboo.coerce_data_types({'age': 'float64', 'salary': 'float64'})
    
    # After coercion, non-numeric entries should become NaN
    assert pd.isna(bamboo.get_data()['age'][1])
    assert pd.isna(bamboo.get_data()['salary'][3])
    
    print(bamboo.get_data())

def test_detect_categorical_columns(sample_data):
    """Test detecting categorical columns based on data types or unique values."""
    bamboo = Bamboo(sample_data)
    categorical_columns = bamboo.detect_categorical_columns()
    
    assert 'name' in categorical_columns
    assert 'join_date' in categorical_columns
    
    print(categorical_columns)

def test_detect_numeric_columns(sample_data):
    """Test detecting numeric columns."""
    bamboo = Bamboo(sample_data)
    numeric_columns = bamboo.detect_numeric_columns()
    
    assert 'age' in numeric_columns
    assert 'salary' in numeric_columns
    
    print(numeric_columns)

if __name__ == '__main__':
    data = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'Derek', 123],  # One invalid type (int)
        'age': [25, 'thirty', 35, 40, None],  # One invalid type (str)
        'salary': [50000.00, None, 60000.00, '100000', 70000.00],  # One invalid type (str)
        'join_date': ['2020-01-01', '2020-02-15', None, 'Invalid Date', '2020-03-10']
    })
    test_enforce_column_types(data)