# BambooChute: Advanced Data Cleaning for Pandas

**BambooChute** is a comprehensive Python package built on top of Pandas, dedicated to simplifying and enhancing data cleaning workflows. With a wide range of functionality for managing missing data, outlier detection, data validation, and more, BambooChute provides an efficient and user-friendly API for complex data cleaning processes.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Functionality Overview](#functionality-overview)
  - [Loading Data](#loading-data)
  - [Handling Missing Data](#handling-missing-data)
  - [Outlier Detection and Removal](#outlier-detection-and-removal)
  - [Categorical Data Processing](#categorical-data-processing)
  - [Date Handling and Transformation](#date-handling-and-transformation)
  - [Data Type Validation](#data-type-validation)
  - [Duplicate Management](#duplicate-management)
  - [Data Formatting](#data-formatting)
  - [Data Profiling](#data-profiling)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Versatile Data Loading:** Load data from multiple formats (CSV, Excel, JSON, DataFrames).
- **Advanced Missing Data Handling:** Use a variety of imputation strategies (mean, median, KNN, custom) and drop functions.
- **Flexible Outlier Detection & Removal:** Detect outliers with Z-score, IQR, Isolation Forest, and DBSCAN, with customizable removal methods.
- **Categorical Data Encoding:** Efficiently convert and encode categorical data with options for one-hot encoding, frequency encoding, and more.
- **Robust Date Handling:** Convert dates, extract parts, manage date ranges, and handle invalid dates.
- **Data Validation Tools:** Validate data types, range of values, and consistency.
- **Duplicate Management:** Identify, mark, merge, and handle near-duplicates using fuzzy matching.
- **Consistent Data Formatting:** Clean string data by trimming whitespace, standardizing cases, and removing special characters.
- **In-depth Data Profiling:** Generate summary reports and detect trends within data.

## Installation

Install BambooChute via pip:

```bash
pip install BambooChute
```

Ensure dependencies from `requirements.txt` are installed:

```bash
pip install -r requirements.txt
```

## Getting Started

Here’s a quick start to load and clean data:

```python
import pandas as pd
from BambooChute import Bamboo

# Load data
data = pd.read_csv('data.csv')

# Initialize Bamboo
bamboo = Bamboo(data)

# Preview data
print(bamboo.preview_data())

# Handle missing data
bamboo.impute_missing(strategy='mean')

# Detect and remove outliers
bamboo.detect_outliers_zscore(threshold=3)

# Export cleaned data
bamboo.export_data('cleaned_data.csv')
```

## Functionality Overview

### Loading Data

Bamboo supports loading data from CSV, Excel, JSON, and directly from Pandas DataFrames.

```python
# Load data from various formats
bamboo = Bamboo('data.csv')  
bamboo = Bamboo(df) 
```

### Handling Missing Data

Various strategies allow for flexibility in handling missing values.

```python
# Basic imputation
bamboo.impute_missing(strategy='mean')

# Custom filling
bamboo.fill_with_custom(lambda x: 'default' if x is None else x)

# Drop missing data
bamboo.drop_missing(axis=0, how='any')
```

### Outlier Detection and Removal

Multiple methods for outlier detection, including Isolation Forest, IQR, and Z-score.

```python
# Detect outliers
outliers = bamboo.detect_outliers_zscore(threshold=3)

# Remove outliers using specific methods
bamboo.remove_outliers(method='iqr')
bamboo.remove_outliers_isolation_forest(contamination=0.1)
```

### Categorical Data Processing

Easily encode and manipulate categorical data.

```python
# Convert columns to categorical type
bamboo.convert_to_categorical(['column'])

# Encode categorical data with one-hot or label encoding
bamboo.encode_categorical(method='onehot')
bamboo.encode_frequency()
```

### Date Handling and Transformation

Comprehensive support for managing date data.

```python
# Convert columns to datetime
bamboo.convert_to_datetime(['date_column'])

# Extract specific date parts (e.g., year, month)
bamboo.extract_date_parts('date_column', parts=['year', 'month'])

# Detect missing intervals in time series
missing_dates = bamboo.detect_time_gaps('date_column')
```

### Data Type Validation

Ensure data types are consistent and enforce specific types across columns.

```python
# Check for data type consistency
consistency = bamboo.check_dtype_consistency()

# Convert columns to specific data types
bamboo.convert_column_types({'age': 'int64'})
```

### Duplicate Management

Handle duplicates efficiently, with support for merging and marking.

```python
# Identify duplicates
duplicates = bamboo.identify_duplicates(subset=['name'])

# Drop duplicates, keeping specific occurrences
bamboo.drop_duplicates(keep='first')

# Handle near-duplicates using fuzzy matching
bamboo.handle_near_duplicates(column='name', threshold=0.8)
```

### Data Formatting

Various utilities to clean and standardize data formatting.

```python
# Trim whitespace in string columns
bamboo.trim_whitespace()

# Standardize case in text
bamboo.standardize_case(case='title')

# Remove special characters
bamboo.remove_special_characters(columns=['column'])
```

### Data Profiling

Generate data profiling reports for better insight into dataset structure and quality.

```python
# Generate profiling summary
summary = bamboo.generate_summary_report()
```

## Testing

To run unit tests:

```bash
python -m unittest discover tests
```

## Contributing

To contribute:

1. Fork the repository.
2. Create a new branch.
3. Make changes and submit a pull request.

## License

Licensed under the MIT License.