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
        'join_date': ['January 1st, 2020', 'February 15th, 2020', None, 'Invalid Date', '2020-03-10']
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
    assert bamboo.get_data()['join_date'][0] == '2020-01-01'
    assert bamboo.get_data()['join_date'][1] == '2020-02-15'
    assert pd.isnull(bamboo.get_data()['join_date'][2])
    assert pd.isnull(bamboo.get_data()['join_date'][3])
    assert bamboo.get_data()['join_date'][4] == '2020-03-10'

    print(bamboo.get_data())    