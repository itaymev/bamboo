# tests/test_formatting.py

import pandas as pd
import numpy as np
import pytest
from bamboo.bamboo import Bamboo

@pytest.fixture
def sample_data():
    """Fixture to provide sample data with various types for testing."""
    return pd.DataFrame({
        'name': ['Alice     ', '    Bob', 'charlie', '  Derek  ', 'CAT'],
        'age': [25, 'thirty', 35, 40, None],
        'salary': [50000.00, None, 60000.00, 'abcd', 70000.00],
        'join_date': ['2020-1-12', '2024-11-2', None, 'Invalid Date', '2020-3-10']
    })

@pytest.fixture
def special_char_data():
    """Fixture to provide sample data with various types for testing."""
    return pd.DataFrame({
        'name': ['Alice&     ', '    %Bob', '#charlie', '  $Derek  ', 'CAT@'],
        'age': [25, 'thirty', 35, 40, None],
        'salary': [50000.00, None, 60000.00, 100000, 70000.00],
        'join_date': ['2020-1-12', '2024-11-2', None, 'Invalid Date', '2020-03-10']
    })

def test_trim_whitespace(sample_data):
    """Test trimming whitespace from string columns."""
    bamboo = Bamboo(sample_data)
    bamboo.trim_whitespace()

    # Check if whitespace is trimmed
    assert bamboo.get_data()['name'][0] == 'Alice'
    assert bamboo.get_data()['name'][1] == 'Bob'
    assert bamboo.get_data()['name'][2] == 'charlie'
    assert bamboo.get_data()['name'][3] == 'Derek'
    assert bamboo.get_data()['name'][4] == 'CAT'

    print(bamboo.get_data())

def test_standardize_case(sample_data):
    """Test standardizing the case of text in string columns."""
    bamboo = Bamboo(sample_data)
    bamboo.trim_whitespace()

    bamboo.standardize_case(case='title') # Check if case is standardized to title
    assert bamboo.get_data()['name'][0] == 'Alice'
    assert bamboo.get_data()['name'][1] == 'Bob'
    assert bamboo.get_data()['name'][2] == 'Charlie'
    assert bamboo.get_data()['name'][3] == 'Derek'
    assert bamboo.get_data()['name'][4] == 'Cat'

    bamboo.standardize_case(case='upper') # Check if case is standardized to upper
    assert bamboo.get_data()['name'][0] == 'ALICE'
    assert bamboo.get_data()['name'][1] == 'BOB'
    assert bamboo.get_data()['name'][2] == 'CHARLIE'
    assert bamboo.get_data()['name'][3] == 'DEREK'
    assert bamboo.get_data()['name'][4] == 'CAT'

    bamboo.standardize_case(case='lower') # Check if case is standardized to lower
    assert bamboo.get_data()['name'][0] == 'alice'
    assert bamboo.get_data()['name'][1] == 'bob'
    assert bamboo.get_data()['name'][2] == 'charlie'
    assert bamboo.get_data()['name'][3] == 'derek'
    assert bamboo.get_data()['name'][4] == 'cat'

    print(bamboo.get_data())

def test_format_dates(sample_data):
    """Test standardizing the format of date columns."""
    bamboo = Bamboo(sample_data)
    bamboo.format_dates(columns = 'join_date')

    # Check if date format is standardized
    assert bamboo.get_data()['join_date'][0] == '2020-01-12'
    assert bamboo.get_data()['join_date'][1] == '2024-11-02'
    assert pd.isnull(bamboo.get_data()['join_date'][2])
    assert pd.isnull(bamboo.get_data()['join_date'][3])
    assert bamboo.get_data()['join_date'][4] == '2020-03-10'

    print(bamboo.get_data())    

def test_remove_special_characters(special_char_data):
    """Test removing special characters from string columns."""
    bamboo = Bamboo(special_char_data)
    bamboo.trim_whitespace()
    bamboo.remove_special_characters()
    bamboo.standardize_case(case='title')

    # Check if special characters are removed
    assert bamboo.get_data()['name'][0] == 'Alice'
    assert bamboo.get_data()['name'][1] == 'Bob'
    assert bamboo.get_data()['name'][2] == 'Charlie'
    assert bamboo.get_data()['name'][3] == 'Derek'
    assert bamboo.get_data()['name'][4] == 'Cat'

    print(bamboo.get_data())

def test_standardize_currency_format(special_char_data):
    """Test standardizing the format of currency columns."""
    bamboo = Bamboo(special_char_data)
    bamboo.standardize_currency_format(columns = 'salary')

    # Check if currency format is standardized
    assert bamboo.get_data()['salary'][0] == '$50,000.00'
    assert bamboo.get_data()['salary'][1] == '$nan'
    assert bamboo.get_data()['salary'][2] == '$60,000.00'
    assert bamboo.get_data()['salary'][3] == '$100,000.00'
    assert bamboo.get_data()['salary'][4] == '$70,000.00'

    print(bamboo.get_data())