# tests/test_duplicates.py

import pytest
import pandas as pd
import numpy as np
from bamboo.bamboo import Bamboo

@pytest.fixture
def sample_data():
    """Fixture to provide sample data with various types for testing."""
    return pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'Derek', 'Cat'],
        'age': [25, 'thirty', 35, 40, None],
        'salary': [50000.00, None, 60000.00, 'abcd', 70000.00],
        'join_date': ['2020-01-01', '2020-02-15', None, 'Invalid Date', '2020-03-10']
    })

def test_identify_duplicates(sample_data):
    """Test identifying duplicate rows in the dataset."""
    bamboo = Bamboo(sample_data)

    duplicates = bamboo.identify_duplicates()
    assert len(duplicates) == 0

    # Add a duplicate row
    sample_data.loc[5] = ['Alice', 25, 50000.00, '2020-01-01']
    duplicates = bamboo.identify_duplicates()
    assert len(duplicates) == 2

    print(duplicates)

def test_drop_duplicates(sample_data):
    """Test dropping duplicate rows from the dataset."""
    bamboo = Bamboo(sample_data)

    bamboo.drop_duplicates()
    assert len(bamboo.get_data()) == 5 # No duplicates to drop

    sample_data.loc[5] = ['Alice', 25, 50000.00, '2020-01-01']
    assert len(bamboo.get_data()) == 6 # Confirm duplicate

    bamboo.drop_duplicates()
    assert len(bamboo.get_data()) == 5 # Confirm duplicate was dropped

    print(bamboo.get_data())

def test_mark_duplicates(sample_data):
    """Test marking duplicate rows in a dataset with no duplicates."""
    bamboo = Bamboo(sample_data)

    bamboo.mark_duplicates()
    assert 'is_duplicate' in bamboo.get_data().columns
    assert len(bamboo.get_data()['is_duplicate']) == 5

    # remove is_duplicate column
    bamboo.get_data().drop(columns=['is_duplicate'], inplace=True)

    sample_data.loc[5] = ['Alice', 25, 50000.00, '2020-01-01']
    bamboo.mark_duplicates()
    assert len(bamboo.get_data()['is_duplicate']) == 6
    assert bamboo.get_data().loc[5]['is_duplicate'] == True

    print(bamboo.get_data())

def test_merge_duplicates(sample_data):
    """Test merging duplicate rows in the dataset."""
    bamboo = Bamboo(sample_data)

    # Add a duplicate row
    sample_data.loc[5] = ['Alice', 30, 100000000000.00, '2024-01-01']
    bamboo.merge_duplicates(subset=['name'])
    assert len(bamboo.get_data()) == 5

    print(bamboo.get_data())

def test_handle_near_duplicates(sample_data):
    """Test handling near-duplicate rows in the dataset."""
    bamboo = Bamboo(sample_data)

    assert bamboo.get_data().loc[0]['name'] == 'Alice'

    # Add a near-duplicate row
    sample_data.loc[5] = ['AlicE', 25, 50000.00, '2020-01-01']
    bamboo.handle_near_duplicates(column=['name'], threshold=0.9)
    assert len(bamboo.get_data()) == 6
    assert bamboo.get_data().loc[0]['name'] == 'AlicE'

    print(bamboo.get_data())