# tests/test_pipelines.py
import pytest
from bamboo.pipelines import BambooPipeline
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
def complex_sample_data():
    """Fixture to provide a more complex DataFrame data for testing."""
    return pd.DataFrame({
        'age': [25, None, 35, 40, 22, 45, None, 60],
        'salary': [50000.00, None, 60000.00, 100000.00, 45000.00, None, 120000.00, 150000.00],
        'name': ['Alice', ' Bob ', 'CHARLIE', 'derek', 'Eva', 'Frank', ' George ', 'harry'],
        'gender': ['F', 'M', 'M', 'M', 'F', 'M', None, 'M'],
        'hire_date': ['2020-01-01', '2019-02-15', None, '2018-07-30', '2021-03-22', None, '2019-06-11', '2018-10-05']
    })

def test_basic_bamboo_pipeline(sample_data):
    pipeline = BambooPipeline()
    pipeline.add_step('impute_missing', strategy='mean') 
    pipeline.add_step('trim_whitespace')  
    pipeline.add_step('standardize_case', case='lower') 

    bamboo = pipeline.execute_pipeline(Bamboo(sample_data))

    # Assertions to verify the changes
    assert not bamboo.get_data()['age'].isnull().any() 
    assert bamboo.get_data()['name'][1] == 'bob' 
    assert bamboo.get_data()['name'][0] == 'alice'
    assert bamboo.get_data()['name'][2] == 'charlie'

    print(bamboo.get_data())

def test_complex_cleaning_pipeline(complex_sample_data):
    # Initialize the complex pipeline
    pipeline = BambooPipeline()

    # Add steps to the pipeline
    pipeline.add_step('impute_missing', strategy='mean')  # Impute missing values
    pipeline.add_step('trim_whitespace')  # Trim whitespaces
    pipeline.add_step('standardize_case', case='lower')  # Standardize text to lowercase
    pipeline.add_step('detect_outliers_zscore', threshold=3)  # Detect outliers using Z-Score
    pipeline.add_step('convert_to_datetime', columns=['hire_date'])  # Convert hire_date to datetime
    pipeline.add_step('encode_categorical', columns=['gender'], method='onehot')  # One-hot encode categorical 'gender'

    # Execute the pipeline
    bamboo = pipeline.execute_pipeline(Bamboo(complex_sample_data))

    # Assertions to verify that the pipeline steps are applied correctly
    # Imputation: no missing values in 'age', 'salary', 'gender'
    assert not bamboo.get_data()['age'].isnull().any()
    assert not bamboo.get_data()['salary'].isnull().any()
    assert not bamboo.get_data()['gender_f'].isnull().any()
    assert not bamboo.get_data()['gender_m'].isnull().any()

    # Standardization: names should be lowercase and trimmed
    assert bamboo.get_data()['name'][1] == 'bob'
    assert bamboo.get_data()['name'][6] == 'george'

    # Outlier detection: ensure outliers column is created
    assert 'outliers' in bamboo.get_data().columns

    # Datetime conversion: 'hire_date' should be in datetime format
    assert pd.api.types.is_datetime64_any_dtype(bamboo.get_data()['hire_date'])

    # One-hot encoding: 'gender' should be replaced with one-hot encoded columns
    # Ensure that both 'gender_f' and 'gender_m' are present after encoding
    assert 'gender_m' in bamboo.get_data().columns
    assert 'gender_f' in bamboo.get_data().columns