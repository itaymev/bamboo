# tests/test_profiling.py
import pytest
from unittest.mock import patch
from bamboo.bamboo import Bamboo
import pandas as pd
import numpy as np

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
        'salary': [50000.00, 60000.00, 60000.00, 10000000000.00],
        'name': ['Alice', 'Bob', 'Charlie', 'Derek'],
        'hire_date': ['2020-01-01', '2020-02-15', '2020-03-30', '2020-04-30']
    })

@pytest.fixture
def dup_data():
    """Fixture to provide a DataFrame with duplicate rows for testing."""
    return pd.DataFrame({
        'age': [25, 30, 35, 40, 25, 30],
        'salary': [50000.00, 60000.00, 60000.00, 10000000000.00, 50000.00, 60000.00],
        'name': ['Alice', 'Bob', 'Charlie', 'Derek', 'Alice', 'Bob'],
        'hire_date': ['2020-01-01', '2020-02-15', '2020-03-30', '2020-04-30', '2020-01-01', '2020-02-15']
    })

def test_basic_summary(clean_data):
    bamboo = Bamboo(clean_data)

    summary = bamboo.basic_summary()
    assert summary.shape[0] == 11
    assert summary.shape[1] == 4
    assert summary.loc['count', 'age'] == 4
    assert summary.loc['count', 'salary'] == 4
    assert summary.loc['mean', 'age'] == 32.5
    assert summary.loc['mean', 'salary'] == 2500042500.0

def test_missing_data_report(sample_data):
    bamboo = Bamboo(sample_data)

    missing_report = bamboo.missing_data_report()
    assert missing_report.shape[0] == 3
    assert missing_report.shape[1] == 2
    assert missing_report.loc['salary', 'missing_count'] == 1
    assert missing_report.loc['salary', 'missing_percentage'] == 25.0

def test_outliers_report(clean_data):
    bamboo = Bamboo(clean_data)

    outliers_report_zscore = bamboo.outliers_report(columns=['salary'], method='zscore', threshold=1.5)
    assert isinstance(outliers_report_zscore, pd.Series)
    assert outliers_report_zscore.sum() > 0

    outliers_report_zscore_high_threshold = bamboo.outliers_report(columns=['age', 'salary'], method='zscore', threshold=3)
    assert outliers_report_zscore_high_threshold.sum() == 0

    outliers_report_iqr = bamboo.outliers_report(columns=['age', 'salary'], method='iqr')
    assert isinstance(outliers_report_iqr, pd.Series)
    assert outliers_report_iqr.sum() > 0

    with pytest.raises(ValueError, match="Unsupported method for outliers report"):
        bamboo.outliers_report(columns=['age', 'salary'], method='unsupported_method')

def test_distribution_report(clean_data):
    bamboo = Bamboo(clean_data)

    with patch("matplotlib.pyplot.show") as mock_show:
        with patch("seaborn.countplot") as mock_countplot, \
             patch("seaborn.histplot") as mock_histplot, \
             patch.object(bamboo, 'log_changes') as mock_log:

            bamboo.distribution_report(columns=['age', 'salary', 'name'])

            assert mock_show.call_count == 3, "Expected show() to be called once per column"
            mock_countplot.assert_called_once()
            assert mock_histplot.call_count == 2
            mock_log.assert_called_once_with("Generated distribution report for columns: ['age', 'salary', 'name'].")

def test_correlation_report(clean_data):
    bamboo = Bamboo(clean_data)

    with patch("matplotlib.pyplot.show") as mock_show:
        with patch("seaborn.heatmap") as mock_heatmap, \
             patch.object(bamboo, 'log_changes') as mock_log:

            bamboo.correlation_report()

            mock_heatmap.assert_called_once()
            mock_log.assert_called_once_with("Generated correlation report.")

def test_data_types_overview(clean_data):
    """
    expected output:
             data_type_count  data_type_percentage
    object                 2                  50.0
    int64                  1                  25.0
    float64                1                  25.0
    """
    bamboo = Bamboo(clean_data)

    data_types = bamboo.data_types_overview()
    assert data_types.shape[0] == 3
    assert data_types.shape[1] == 2
    assert data_types.iloc[0, 0] == 2
    assert data_types.iloc[0, 1] == 50.0
    assert data_types.iloc[1, 0] == 1
    assert data_types.iloc[1, 1] == 25.0
    assert data_types.iloc[2, 0] == 1
    assert data_types.iloc[2, 1] == 25.0

def test_duplicate_report(clean_data, dup_data):
    bamboo = Bamboo(clean_data)

    duplicate_report = bamboo.duplicate_report()
    assert duplicate_report.shape[0] == 1
    assert duplicate_report.shape[1] == 3
    assert duplicate_report.loc[0, 'duplicate_count'] == 0

    bamboo = Bamboo(dup_data)

    duplicate_report = bamboo.duplicate_report()
    assert duplicate_report.shape[0] == 1
    assert duplicate_report.shape[1] == 3
    assert duplicate_report.loc[0, 'duplicate_count'] == 2