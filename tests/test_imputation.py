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

def test_fill_with_custom(sample_data):
    """Test filling missing values with a custom function."""
    bamboo = Bamboo(sample_data)

    # random custom function to fill missing values
    custom_function = lambda x: 9999 if pd.isna(x) else x

    bamboo.fill_with_custom(custom_function, columns=['age', 'salary'])
    
    assert bamboo.get_data()['age'].isnull().sum() == 0
    assert bamboo.get_data()['salary'].isnull().sum() == 0
    assert bamboo.get_data()['salary'].iloc[1] == 9999  # Verify custom fill value
    
    print(bamboo.get_data())

def test_impute_knn(sample_data):
    """Test KNN imputation for missing values."""
    bamboo = Bamboo(sample_data)
    
    bamboo.data['age'] = pd.to_numeric(bamboo.data['age'], errors='coerce')
    bamboo.data['salary'] = pd.to_numeric(bamboo.data['salary'], errors='coerce')
    
    bamboo.impute_knn(n_neighbors=2, columns=['age', 'salary'])
    
    assert bamboo.get_data()['age'].isnull().sum() == 0
    assert bamboo.get_data()['salary'].isnull().sum() == 0
    assert bamboo.get_data()['age'].iloc[1] == 35.0 # verify knn value
    
    print(bamboo.get_data())

def test_interpolate_missing(sample_data):
    """Test interpolating missing values in the dataset."""
    bamboo = Bamboo(sample_data)

    bamboo.data['salary'] = pd.to_numeric(bamboo.data['age'], errors='coerce')

    bamboo.interpolate_missing(method='linear', axis=0)
    
    assert bamboo.get_data()['salary'].isnull().sum() == 0
    assert bamboo.get_data()['salary'].iloc[1] == 30.0 # verify interpolated value

    print(bamboo.get_data())

def test_impute_regression(sample_data):
    """Test regression-based imputation of missing values."""
    bamboo = Bamboo(sample_data)

    bamboo.data['age'] = pd.to_numeric(bamboo.data['age'], errors='coerce')
    bamboo.data['salary'] = pd.to_numeric(bamboo.data['salary'], errors='coerce')
    bamboo.data['age'].iloc[1] = 30.0

    bamboo.impute_regression(target_column='salary', predictor_columns=['age'])
    
    assert bamboo.get_data()['salary'].isnull().sum() == 0
    assert np.isclose(bamboo.get_data()['salary'].iloc[1], 55714.28571428571) # verify regression value
    assert np.isclose(bamboo.get_data()['salary'].iloc[3], 68571.42857142858) 

    print(bamboo.get_data())

def test_impute_mice(sample_data):
    """Test MICE imputation for missing values."""
    bamboo = Bamboo(sample_data)

    bamboo.data['age'] = pd.to_numeric(bamboo.data['age'], errors='coerce')
    bamboo.data['salary'] = pd.to_numeric(bamboo.data['salary'], errors='coerce')

    bamboo.impute_mice(columns=['age', 'salary'], max_iter=5)
    
    assert bamboo.get_data()['age'].isnull().sum() == 0
    assert bamboo.get_data()['salary'].isnull().sum() == 0
    print(bamboo.get_data())

def test_impute_em(sample_data):
    """Test EM (Expectation-Maximization) imputation for missing values."""
    bamboo = Bamboo(sample_data)

    bamboo.data['age'] = pd.to_numeric(bamboo.data['age'], errors='coerce')
    bamboo.data['salary'] = pd.to_numeric(bamboo.data['salary'], errors='coerce')

    bamboo.impute_em(columns=['age', 'salary'], rank=1, max_iter=10)
    
    assert bamboo.get_data()['age'].isnull().sum() == 0
    assert bamboo.get_data()['salary'].isnull().sum() == 0
    print(bamboo.get_data())