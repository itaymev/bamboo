# tests/test_validation.py
import pytest
from bamboo.bamboo import Bamboo
import pandas as pd

@pytest.fixture
def sample_data():
    """Fixture to provide sample DataFrame data for testing."""
    return pd.DataFrame({
        'age': [25, None, 35, 40],
        'salary': [50000.00, None, 60000.00, 100000.00],
        'name': ['Alice', ' Bob ', 'CHARLIE', 'derek']
    })

@pytest.fixture
def clean_data():
    """Fixture to provide a clean DataFrame data for testing."""
    return pd.DataFrame({
        'age': [25, 30, 35, 40],
        'salary': [50000.00, 60000.00, 60000.00, 100000.00],
        'name': ['Alice', 'Bob', 'Charlie', 'Derek'],
        'hire_date': ['2020-01-01', '2020-02-15', '2020-03-30', '2020-04-30']
    })

def test_validate_missing_data(sample_data):
    bamboo = Bamboo(sample_data)

    assert not bamboo.validate_missing_data(columns=['age', 'salary'])
    assert bamboo.validate_missing_data(columns=['name'])

def test_validate_data_types(clean_data):
    bamboo = Bamboo(clean_data)

    expected_types = {'age': 'int64', 'salary': 'float64', 'name': 'object'}
    assert bamboo.validate_data_types(expected_types)

    expected_types = {'age': 'float64', 'salary': 'int64', 'name': 'float64'}
    assert not bamboo.validate_data_types(expected_types)

def test_validate_value_ranges(clean_data):
    bamboo = Bamboo(clean_data)

    assert bamboo.validate_value_ranges('age', min_value=20, max_value=50)
    assert not bamboo.validate_value_ranges('age', min_value=30, max_value=50)

    assert bamboo.validate_value_ranges('salary', min_value=50000.00, max_value=100000.00)
    assert not bamboo.validate_value_ranges('salary', min_value=50000.00, max_value=80000.00)

def test_validate_unique_values(clean_data):
    bamboo = Bamboo(clean_data)

    assert bamboo.validate_unique_values('name')
    assert bamboo.validate_unique_values('name')
    assert not bamboo.validate_unique_values('salary')

def test_validate_categories(clean_data):
    bamboo = Bamboo(clean_data)

    assert bamboo.validate_categories('name', valid_categories=['Alice', 'Bob', 'Charlie', 'Derek'])
    assert not bamboo.validate_categories('name', valid_categories=['Alice', 'Bob', 'Charlie'])

def test_validate_date_range(clean_data):
    bamboo = Bamboo(clean_data)

    assert bamboo.validate_date_range('hire_date', start_date='2020-01-01', end_date='2021-01-01')
    assert not bamboo.validate_date_range('hire_date', start_date='2020-01-01', end_date='2020-03-15')

def test_custom_validation(clean_data):
    bamboo = Bamboo(clean_data)

    def custom_validation(data):
        return data.mean() > 30

    assert bamboo.custom_validation('age', custom_validation)