# tests/test_dates.py

import pytest
import pandas as pd
from bamboo.bamboo import Bamboo

@pytest.fixture
def sample_data():
    """Fixture to provide sample date-related data."""
    return pd.DataFrame({
        'event_date': ['2020-01-01', '2020-02-15', '2020-03-10', 'Invalid Date', None],
        'name': ['Alice', 'Bob', 'Charlie', 'Derek', 'Eva']
    })

def test_convert_to_datetime(sample_data):
    """Test converting columns to datetime format."""
    bamboo = Bamboo(sample_data)
    bamboo.convert_to_datetime(columns=['event_date'])
    
    assert pd.api.types.is_datetime64_any_dtype(bamboo.get_data()['event_date'])
    print(bamboo.get_data())

def test_extract_date_parts(sample_data):
    """Test extracting specific date parts from a datetime column."""
    bamboo = Bamboo(sample_data)
    bamboo.convert_to_datetime(columns=['event_date'])
    bamboo.extract_date_parts('event_date', parts=['year', 'month', 'day'])
    
    assert 'event_date_year' in bamboo.get_data().columns
    assert 'event_date_month' in bamboo.get_data().columns
    assert 'event_date_day' in bamboo.get_data().columns
    print(bamboo.get_data())

def test_create_date_range():
    """Test creating a new column with a date range."""
    bamboo = Bamboo(pd.DataFrame())
    bamboo.create_date_range(start='2020-01-01', end='2020-01-10', freq='D', column_name='date_range')
    
    assert 'date_range' in bamboo.get_data().columns
    assert len(bamboo.get_data()) == 10
    print(bamboo.get_data())

def test_handle_invalid_dates(sample_data):
    """Test handling invalid dates in a datetime column."""
    bamboo = Bamboo(sample_data)
    bamboo.convert_to_datetime(columns=['event_date'])
    bamboo.handle_invalid_dates(columns=['event_date'], fill_value='2020-01-01')
    
    assert not bamboo.get_data()['event_date'].isnull().any()
    print(bamboo.get_data())

def test_calculate_date_differences(sample_data):
    """Test calculating date differences between two columns."""
    sample_data['event_end_date'] = ['2020-01-05', '2020-02-18', '2020-03-15', '2020-04-01', None]
    bamboo = Bamboo(sample_data)
    bamboo.convert_to_datetime(columns=['event_date', 'event_end_date'])
    bamboo.calculate_date_differences('event_date', 'event_end_date', unit='days', new_column='date_diff')
    
    non_null_dates = bamboo.get_data().dropna(subset=['event_date', 'event_end_date'])
    
    assert 'date_diff' in bamboo.get_data().columns
    assert non_null_dates['date_diff'].notna().all()
    
    print(bamboo.get_data())

def test_shift_dates(sample_data):
    """Test shifting dates by a specified number of periods."""
    bamboo = Bamboo(sample_data)
    bamboo.convert_to_datetime(columns=['event_date'])
    bamboo.shift_dates(columns=['event_date'], periods=2, freq='D')
    
    shifted_dates = bamboo.get_data()['event_date'].dropna()
    assert (shifted_dates == pd.to_datetime(['2020-01-03', '2020-02-17', '2020-03-12'])).all()
    print(bamboo.get_data())

def test_round_dates(sample_data):
    """Test rounding dates with timestamps to the nearest day."""

    sample_data['event_date'] = ['2020-01-01 10:15:00', '2020-02-15 23:45:00', '2020-03-10 05:30:00', None, None]
    bamboo = Bamboo(sample_data)

    bamboo.convert_to_datetime(columns=['event_date'])
    bamboo.round_dates(columns=['event_date'], freq='D')
    rounded_dates = bamboo.get_data()['event_date'].dropna()
    
    expected_dates = pd.to_datetime(['2020-01-01', '2020-02-16', '2020-03-10'])
    assert (rounded_dates == expected_dates).all()
    
    print(bamboo.get_data())


def test_detect_time_gaps(sample_data):
    """Test detecting gaps in a sequence of dates."""
    bamboo = Bamboo(sample_data)
    bamboo.convert_to_datetime(columns=['event_date'])
    missing_dates_df = bamboo.detect_time_gaps('event_date', freq='D')
    
    assert 'missing_dates' in missing_dates_df.columns
    print(missing_dates_df)

def test_convert_bogus_date():
    """Test converting a column with a bogus date to datetime format."""
    # Sample data including a bogus date
    bogus_data = pd.DataFrame({
        'event_date': ['2020-01-01', '2020-02-15', 'Bogus Date', '2020-03-10', None],
        'name': ['Alice', 'Bob', 'Charlie', 'Derek', 'Eva']
    })
    
    bamboo = Bamboo(bogus_data)
    bamboo.convert_to_datetime(columns=['event_date'])
    
    # Check if the invalid "Bogus Date" and None values were converted to NaT
    assert pd.isna(bamboo.get_data()['event_date'][2]), "Bogus date should be converted to NaT"
    assert pd.isna(bamboo.get_data()['event_date'][4]), "None should remain NaT"
    assert pd.api.types.is_datetime64_any_dtype(bamboo.get_data()['event_date'])
    
    print(bamboo.get_data())
