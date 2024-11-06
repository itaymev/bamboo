# tests/test_outliers.py

import pandas as pd
import numpy as np
import pytest
from bamboo.bamboo import Bamboo

@pytest.fixture
def sample_data():
    """Fixture to provide sample data with numeric outliers for testing."""
    return pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'Derek', 'Eve'],
        'age': [25, 100, 35, 40, 25],  # Age 100 is an outlier
        'salary': [50000, 60000, 70000, 80000, 3000000],  # Salary 3000000 is an outlier
        'join_date': ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05']
    })

def test_detect_outliers_zscore(sample_data):
    """Test Z-Score outlier detection."""
    bamboo = Bamboo(sample_data)
    outliers = bamboo.detect_outliers_zscore(columns=['age', 'salary'], threshold=1.5)
    
    # Expect 'age' and 'salary' to have at least one outlier detected
    assert bamboo.get_data()['outliers'].sum() > 0
    print(outliers)

def test_detect_outliers_iqr(sample_data):
    """Test IQR outlier detection."""
    bamboo = Bamboo(sample_data)
    outliers = bamboo.detect_outliers_iqr(columns=['age', 'salary'], multiplier=1.5)
    
    # Expect 'salary' to have an outlier detected (3000000)
    assert outliers['salary'].sum() > 0
    print(outliers)

def test_detect_outliers_isolation_forest(sample_data):
    """Test Isolation Forest outlier detection."""
    bamboo = Bamboo(sample_data)
    outliers = bamboo.detect_outliers_isolation_forest(contamination=0.2, columns=['age', 'salary'])
    
    # Expect outliers in the isolation forest output
    assert bamboo.get_data()['outliers'].sum() > 0
    print(outliers)

def test_detect_outliers_dbscan(sample_data):
    """Test DBSCAN outlier detection."""
    bamboo = Bamboo(sample_data)
    outliers = bamboo.detect_outliers_dbscan(eps=0.5, min_samples=2, columns=['age', 'salary'])
    
    # Expect some outliers with DBSCAN method
    assert bamboo.get_data()['outliers'].sum() > 0
    print(outliers)

def test_detect_outliers_modified_zscore(sample_data):
    """Test Modified Z-Score outlier detection."""
    bamboo = Bamboo(sample_data)
    outliers = bamboo.detect_outliers_modified_zscore(columns=['age', 'salary'], threshold=3.5)
    
    # Expect outliers to be detected based on modified Z-score
    assert bamboo.get_data()['outliers'].sum() > 0
    print(outliers)

def test_detect_outliers_robust_covariance(sample_data):
    """Test Robust Covariance outlier detection."""
    bamboo = Bamboo(sample_data)
    outliers = bamboo.detect_outliers_robust_covariance(contamination=0.2, columns=['age', 'salary'])
    
    # Expect robust covariance to detect some outliers
    assert bamboo.get_data()['outliers'].sum() > 0
    print(outliers)

def test_detect_outliers_lof(sample_data):
    """Test Local Outlier Factor (LOF) outlier detection."""
    bamboo = Bamboo(sample_data)
    outliers = bamboo.detect_outliers_lof(n_neighbors=2, contamination=0.2, columns=['age', 'salary'])
    
    # Expect LOF to identify some outliers
    assert bamboo.get_data()['outliers'].sum() > 0
    print(outliers)

def test_remove_outliers_zscore(sample_data):
    """Test removing outliers using Z-Score method."""
    bamboo = Bamboo(sample_data)
    bamboo.remove_outliers(method='zscore', threshold=2.5, columns=['age', 'salary'])
    
    # Expect remaining rows to have no outliers
    assert bamboo.get_data()['outliers'].sum() == 0
    print(bamboo.get_data())

def test_remove_outliers_isolation_forest(sample_data):
    """Test removing outliers using Isolation Forest."""
    bamboo = Bamboo(sample_data)
    bamboo.remove_outliers_isolation_forest(contamination=0.2, columns=['age', 'salary'])
    
    # Expect that outliers have been removed
    assert 'outliers' not in bamboo.get_data() or bamboo.get_data()['outliers'].sum() == 0
    print(bamboo.get_data())

def test_remove_outliers_dbscan(sample_data):
    """Test removing outliers using DBSCAN."""
    bamboo = Bamboo(sample_data)
    bamboo.remove_outliers_dbscan(eps=0.5, min_samples=2, columns=['age', 'salary'])
    
    # Expect that outliers have been removed
    assert 'outliers' not in bamboo.get_data() or bamboo.get_data()['outliers'].sum() == 0
    print(bamboo.get_data())

def test_cap_outliers(sample_data):
    """Test capping outliers with specified bounds."""
    bamboo = Bamboo(sample_data)
    bamboo.cap_outliers(method='zscore', lower_cap=30, upper_cap=200000, threshold=1.5, columns=['age', 'salary'])
    
    # Check that outliers are capped within specified limits
    assert bamboo.get_data()['age'].max() <= 200000
    assert bamboo.get_data()['salary'].max() <= 200000
    print(bamboo.get_data())
