# Bamboo: Advanced Data Cleaning for Pandas

**Bamboo** is a Python package built on top of Pandas to streamline and simplify advanced data cleaning processes. It offers a rich set of tools for handling missing data, outliers, categorical transformations, date manipulations, and more. With Bamboo, you can perform common and complex data cleaning tasks more efficiently, with an easy-to-use, extensible API.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Usage](#usage)
  - [Loading Data](#loading-data)
  - [Handling Missing Data](#handling-missing-data)
  - [Outlier Detection](#outlier-detection)
  - [Categorical Data](#categorical-data)
  - [Date Manipulations](#date-manipulations)
  - [Data Validation](#data-validation)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Missing Data Handling:** Imputation using strategies like mean, median, KNN, and more.
- **Outlier Detection & Removal:** Z-Score, IQR, Isolation Forest, and others.
- **Date Handling:** Conversion, extraction, range creation, and more.
- **Categorical Data Processing:** Encoding, conversion, and handling missing categories.
- **Data Validation:** Validate missing data, data types, value ranges, and custom validations.
- **Pipelines:** Save and load cleaning pipelines for reuse.
- **Profiling:** Generate summary reports on missing data, outliers, distribution, and correlations.

## Installation

Install Bamboo via pip:

```bash
pip install bamboo-cleaning
```

Make sure you have Python 3.6+ and the dependencies in requirements.txt are installed:

```bash
pip install -r requirements.txt
```

## Getting Started

Here’s a quick example to get you started:

```python
import pandas as pd
from bamboo import Bamboo

# Load data
data = pd.read_csv('example.csv')

# Initialize Bamboo
bamboo = Bamboo(data)

# Preview the first few rows
print(bamboo.preview_data())

# Handle missing data
bamboo.impute_missing(strategy='mean')

# Detect and remove outliers using Z-Score method
bamboo.detect_outliers_zscore(threshold=3)

# Export cleaned data
bamboo.export_data('cleaned_data.csv')
```

## Usage

### Loading Data

Bamboo supports loading data from various formats, including CSV, Excel, JSON, and Pandas DataFrames:

```python
bamboo = Bamboo('data.csv')  # Load from CSV
bamboo = Bamboo(df)  # Load directly from DataFrame
```

### Handling Missing Data

Impute missing values using different strategies:

```python
bamboo.impute_missing(strategy='mean')
bamboo.impute_knn(n_neighbors=5)
bamboo.drop_missing(axis=0, how='any')
```

### Outlier Detection

Detect outliers using various methods:

```python
# Detect outliers with Z-Score method
outliers = bamboo.detect_outliers_zscore(threshold=3)

# Remove outliers
bamboo.remove_outliers(method='zscore')
```

### Categorical Data

Handle categorical columns easily:

```python
# Convert to categorical
bamboo.convert_to_categorical()

# One-hot encode categorical columns
bamboo.encode_categorical(method='onehot')
```

### Date Manipulations

Perform complex date manipulations with ease:

```python
# Convert columns to datetime
bamboo.convert_to_datetime(['date_column'])

# Extract year, month, day from a date column
bamboo.extract_date_parts('date_column', parts=['year', 'month'])
```

### Data Validation

Validate your dataset before and after cleaning:

```python
# Validate that no missing data exists
is_valid = bamboo.validate_missing_data()

# Validate data types
expected_types = {'age': 'int64', 'name': 'object'}
bamboo.validate_data_types(expected_types)
```

## Testing

The package comes with a set of unit tests under the `tests` directory. You can run the tests using:

```bash
python -m unittest discover tests
```

## Contributing

Contributions are welcome. Please open an issue or submit a pull request if you have suggestions.

### Steps to Contribute:
1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Submit a pull request.

## License

Bamboo is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---