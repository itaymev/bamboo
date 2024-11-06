# tests/test_imputation.py

import pandas as pd
import numpy as np
import pytest
from bamboo.bamboo import Bamboo

@pytest.fixture
def sample_data():
    """Fixture to provide sample data with various types for testing."""
    return pd.DataFrame({
        'name': ['Alice', 'Bob', 'charlie', 'Derek', 'CAT'],
        'age': [25, 'thirty', 35, 40, 40],
        'salary': [50000.00, None, 60000.00, 'abcd', 70000.00],
        'join_date': ['2020-1-12', '2024-11-2', None, 'Invalid Date', '2020-3-10']
    })

def test_impute_missing(sample_data):
    """Test imputing missing values in the dataset."""
    bamboo = Bamboo(sample_data)

    bamboo.impute_missing(strategy='mean', columns=['age', 'salary', 'join_date'])
    assert bamboo.get_data()['age'].isnull().sum() == 0
    assert bamboo.get_data()['salary'].isnull().sum() == 0
    assert bamboo.get_data()['join_date'].isnull().sum() == 0

    print(bamboo.get_data())

def test_drop_missing(sample_data):
    """Test dropping rows with missing values from the dataset."""
    bamboo = Bamboo(sample_data)

    bamboo.drop_missing(axis=0, how='any')
    assert len(bamboo.get_data()) == 3

    bamboo = Bamboo(sample_data)

    # threshold based dropping
    bamboo.drop_missing(axis=0, how=None, thresh=2)
    assert len(bamboo.get_data()) == 3

    print(bamboo.get_data())

