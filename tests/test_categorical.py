# tests/test_categorical.py

import pytest
import pandas as pd
from bamboo.bamboo import Bamboo

@pytest.fixture
def sample_data():
    """Fixture to provide sample categorical data."""
    return pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob', None],
        'city': ['New York', 'Los Angeles', 'New York', 'Chicago', None, 'Chicago'],
        'gender': ['F', 'M', None, 'F', 'M', 'F']
    })

def test_convert_to_categorical(sample_data):
    """Test converting columns to categorical data type."""
    bamboo = Bamboo(sample_data)
    bamboo.convert_to_categorical(columns=['name', 'city'])
    
    assert bamboo.get_data()['name'].dtype.name == 'category'
    assert bamboo.get_data()['city'].dtype.name == 'category'
    print(bamboo.get_data())

def test_get_unique_categories(sample_data):
    """Test getting unique categories from a categorical column."""
    bamboo = Bamboo(sample_data)
    bamboo.convert_to_categorical(columns=['name'])
    unique_categories = bamboo.get_unique_categories('name')
    
    assert set(unique_categories) == {'Alice', 'Bob', 'Charlie'}
    print(unique_categories)

def test_encode_categorical_onehot(sample_data):
    """Test one-hot encoding of categorical columns."""
    bamboo = Bamboo(sample_data)
    bamboo.convert_to_categorical(columns=['name'])
    bamboo.encode_categorical(columns=['name'], method='onehot')
    
    assert 'name_Alice' in bamboo.get_data().columns
    assert 'name_Bob' in bamboo.get_data().columns
    assert 'name_Charlie' in bamboo.get_data().columns
    print(bamboo.get_data())

def test_handle_missing_categories(sample_data):
    """Test handling missing categories by filling with a default value."""
    bamboo = Bamboo(sample_data)
    bamboo.convert_to_categorical(columns=['gender'])
    bamboo.handle_missing_categories('gender', fill_value='Unknown')
    
    assert bamboo.get_data()['gender'].isnull().sum() == 0
    assert 'Unknown' in bamboo.get_data()['gender'].cat.categories
    print(bamboo.get_data())

def test_detect_rare_categories(sample_data):
    """Test detecting rare categories in a categorical column."""
    bamboo = Bamboo(sample_data)
    bamboo.convert_to_categorical(columns=['name'])
    rare_categories = bamboo.detect_rare_categories('name', threshold=0.3)

    assert rare_categories == ['Charlie']
    print(bamboo.get_data())

def test_replace_rare_categories(sample_data):
    """Test replacing rare categories with a default value."""
    bamboo = Bamboo(sample_data)
    bamboo.convert_to_categorical(columns=['name'])
    bamboo.replace_rare_categories('name', threshold=0.3, replacement='Other')
    
    assert 'Other' in bamboo.get_data()['name'].cat.categories
    assert 'Charlie' not in bamboo.get_data()['name'].cat.categories
    print(bamboo.get_data())