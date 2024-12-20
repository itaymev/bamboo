# BambooChute: Data Cleaning for Pandas

**BambooChute** is a comprehensive data cleaning toolkit built on top of Pandas, offering an array of functions to streamline your data preparation process.

## Table of Contents

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Key Features Overview](#key-features-overview)
  - [1. Data Loading](#1-data-loading)
  - [2. Imputation (Handling Missing Data)](#2-imputation-handling-missing-data)
  - [3. Outlier Detection & Removal](#3-outlier-detection--removal)
  - [4. Categorical Data Processing](#4-categorical-data-processing)
  - [5. Date Handling & Transformation](#5-date-handling--transformation)
  - [6. Data Type Validation & Conversion](#6-data-type-validation--conversion)
  - [7. Duplicates & Near-Duplicates](#7-duplicates--near-duplicates)
  - [8. Data Formatting & String Cleaning](#8-data-formatting--string-cleaning)
  - [9. Data Profiling](#9-data-profiling)
  - [10. Pipelines](#10-pipelines)
  - [11. Undo/Redo & Logging](#11-undoredo--logging)
  - [12. Data Validation Rules](#12-data-validation-rules)
- [Example Usage](#example-usage)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

Install **BambooChute** using **pip**:

```bash
pip install BambooChute
```

(If you’re installing from source or a local repo, ensure any dependencies in `requirements.txt` are satisfied.)

---

## Getting Started

**BambooChute** can load data from multiple file formats or directly from a Pandas DataFrame. Below is a minimal quickstart example demonstrating data loading, missing value imputation, outlier detection, and exporting.

```python
import pandas as pd
from bamboochute import Bamboo

# 1. Load data
data = pd.read_csv("data.csv")
bamboo = Bamboo(data)

# 2. Preview data
print(bamboo.preview_data())

# 3. Impute missing values (mean for numeric, mode for categorical)
bamboo.impute_missing(strategy='mean')

# 4. Detect outliers with Z-score (threshold=3) and remove them
bamboo.remove_outliers(method='zscore', threshold=3)

# 5. Export cleaned data
bamboo.export_data("cleaned_data.csv", format="csv")
```

---

## Key Features Overview

### 1. Data Loading

**BambooChute** supports:
- **CSV**: `Bamboo("path/to/data.csv")`
- **Excel**: `Bamboo("path/to/data.xlsx")`
- **JSON**: `Bamboo("path/to/data.json")`
- **Pandas DataFrame**: `Bamboo(df)`

It automatically infers the file type and loads it into a Pandas DataFrame, or uses the DataFrame you provide.

### 2. Imputation (Handling Missing Data)

**BambooChute** offers multiple imputation strategies out of the box:

- **Basic Strategies**  
  - **Mean, Median, Mode**  
  ```python
  bamboo.impute_missing(strategy='mean')
  ```
- **KNN Imputation**  
  ```python
  bamboo.impute_knn(n_neighbors=5)
  ```
- **Regression Imputation**  
  ```python
  bamboo.impute_regression(target_column='Y', predictor_columns=['X1','X2'])
  ```
- **MICE (Multiple Imputation by Chained Equations)**  
  ```python
  bamboo.impute_mice(max_iter=10, tol=1e-3)
  ```
- **EM (Expectation-Maximization)**  
  ```python
  bamboo.impute_em(max_iter=100, tol=1e-3)
  ```
- **Custom Function**  
  ```python
  bamboo.fill_with_custom(lambda x: 'Unknown' if pd.isna(x) else x)
  ```

Or simply drop missing data:

```python
bamboo.drop_missing(axis=0, how='any')
```

### 3. Outlier Detection & Removal

Various **outlier detection** methods and corresponding **removal** or **clipping**:

- **Z-score**, **IQR**, **Isolation Forest**, **DBSCAN**, **LOF**, **Robust Covariance**, **Modified Z-score**:

```python
# Detect outliers with Z-score
outliers = bamboo.detect_outliers_zscore(threshold=3)

# Remove outliers with Isolation Forest
bamboo.remove_outliers_isolation_forest(contamination=0.1)
```

You can also **clip** outliers to a specific value or range:
```python
bamboo.cap_outliers(method='iqr', lower_cap=0, upper_cap=100)
```

### 4. Categorical Data Processing

- **Convert to Categorical**  
  ```python
  bamboo.convert_to_categorical(columns=['CategoryColumn'])
  ```
- **Categorical Encoding** (One-Hot, Label, Frequency)  
  ```python
  bamboo.encode_categorical(method='onehot')
  bamboo.encode_frequency(['CategoryColumn'])
  ```
- **Rare Category Detection & Replacement**  
  ```python
  rare = bamboo.detect_rare_categories('CategoryColumn', threshold=0.01)
  bamboo.replace_rare_categories('CategoryColumn', replacement='Other')
  ```

### 5. Date Handling & Transformation

- **Convert to Datetime**  
  ```python
  bamboo.convert_to_datetime(['DateColumn'])
  ```
- **Extract Date Parts** (year, month, day, weekday…)  
  ```python
  bamboo.extract_date_parts('DateColumn', parts=['year','month','weekday'])
  ```
- **Create/Shift/Round Dates** & **Detect Time Gaps**  
  ```python
  bamboo.shift_dates(['DateColumn'], periods=7, freq='D')  # Shift a week forward
  missing_date_gaps = bamboo.detect_time_gaps('DateColumn', freq='D')
  ```

### 6. Data Type Validation & Conversion

- **Check Data Type Consistency**  
  ```python
  bamboo.check_dtype_consistency()
  ```
- **Convert & Enforce Column Types**  
  ```python
  bamboo.enforce_column_types({'Age': 'int64', 'Price': 'float64'})
  ```
- **Detect Numeric & Categorical Columns**  
  ```python
  numeric_cols = bamboo.detect_numeric_columns()
  cat_cols = bamboo.detect_categorical_columns()
  ```

### 7. Duplicates & Near-Duplicates

- **Identify, Drop, or Mark Duplicates**  
  ```python
  duplicates = bamboo.identify_duplicates(subset=['Name'])
  bamboo.drop_duplicates(keep='first')
  ```
- **Merge Duplicates** with different strategies (`most_frequent`, `most_recent`).
- **Near-Duplicate Detection** via **fuzzy matching**  
  ```python
  bamboo.handle_near_duplicates(column='Name', threshold=0.8)
  ```

### 8. Data Formatting & String Cleaning

- **Trim Whitespace & Standardize Case**  
  ```python
  bamboo.trim_whitespace().standardize_case(case='lower')
  ```
- **Remove Special Characters**  
  ```python
  bamboo.remove_special_characters(columns=['TextColumn'], chars_to_remove='@#$')
  ```
- **Format Dates**  
  ```python
  bamboo.format_dates(format='%Y-%m-%d', columns=['DateColumn'])
  ```
- **Currency Formatting**  
  ```python
  bamboo.standardize_currency_format(columns=['Price'])
  ```

### 9. Data Profiling

**BambooChute** provides an array of **profiling** methods:

- **Basic Summary**  
  ```python
  summary = bamboo.basic_summary()
  ```
- **Missing Data Report**  
  ```python
  missing_report = bamboo.missing_data_report()
  ```
- **Outliers Report**  
  ```python
  outliers_report = bamboo.outliers_report(method='zscore', threshold=3)
  ```
- **Distribution & Correlation Reports**  
  ```python
  bamboo.distribution_report(columns=['Price','Quantity'])
  corr_matrix = bamboo.correlation_report()
  ```
- **Duplicate Report**  
  ```python
  dup_report = bamboo.duplicate_report()
  ```

### 10. Pipelines

Create **reproducible data cleaning pipelines**:

```python
from bamboochute import Bamboo, BambooPipeline

pipeline = BambooPipeline()
pipeline.add_step('impute_missing', strategy='mean')
pipeline.add_step('drop_missing', axis=0, how='any')
pipeline.add_step('remove_outliers', method='zscore', threshold=3)

# Save to JSON file
pipeline.save_pipeline("my_pipeline.json")

# Load & execute pipeline
loaded_pipeline = BambooPipeline.load_pipeline("my_pipeline.json")
bamboo = Bamboo("data.csv")
cleaned_bamboo = loaded_pipeline.execute_pipeline(bamboo)
```

### 11. Undo/Redo & Logging

- **Undo/Redo**: BambooChute automatically **tracks** changes:
  ```python
  bamboo.save_state()  # Save a snapshot
  # ...some cleaning...
  bamboo.undo()        # Revert the last change
  bamboo.reset_data()  # Revert to original data
  ```
- **Logging**: By default, **logging is on**. You can enable/disable:
  ```python
  from bamboochute.settings.log import set_logging
  set_logging(False)   # Turn off logging globally
  set_logging(True)    # Re-enable logging
  ```

### 12. Data Validation Rules

Built-in **validation** methods to ensure data integrity:

- **Validate Missing Data**  
  ```python
  no_missing = bamboo.validate_missing_data(columns=['ColumnA','ColumnB'])
  ```
- **Validate Data Types**  
  ```python
  dtype_ok = bamboo.validate_data_types({'ColumnA': 'int64', 'ColumnB': 'object'})
  ```
- **Validate Value Ranges**  
  ```python
  in_range = bamboo.validate_value_ranges(column='Age', min_value=0, max_value=120)
  ```
- **Validate Unique Values**, **Valid Categories**, **Date Ranges**, or write a **Custom Validation** function.

---

## Example Usage

Below is a more extended snippet demonstrating how you might chain multiple cleaning operations:

```python
import pandas as pd
from bamboochute import Bamboo

df = pd.read_csv("raw_dataset.csv")
bamboo = Bamboo(df)

# 1. Convert certain columns to date & categorical
bamboo.convert_to_datetime(columns=['date'])
bamboo.convert_to_categorical(columns=['category_col'])

# 2. Impute missing data with advanced methods
bamboo.impute_knn(n_neighbors=5, columns=['numeric_col1','numeric_col2'])
bamboo.impute_mice(columns=['numeric_col3'], max_iter=5)

# 3. Handle outliers
bamboo.remove_outliers_isolation_forest(contamination=0.05)

# 4. Clean strings & format currency
bamboo.trim_whitespace().standardize_case()
bamboo.standardize_currency_format(columns=['price_col'])

# 5. Validate and produce a summary report
assert bamboo.validate_missing_data() is True, "Error: Missing data!"
summary = bamboo.basic_summary()
print(summary)

# 6. Export cleaned dataset
bamboo.export_data("final_dataset.csv", format='csv')
```

---

## Testing

**BambooChute** uses [pytest](https://docs.pytest.org) for testing.  
To run all tests in the `tests` folder:

```bash
pytest tests/
```

Optional flags:
- **`-v`** for verbose mode  
- **`--maxfail=3`** to stop after three test failures

---

## Contributing

1. **Fork** the repository.  
2. Create a **new branch**.  
3. Make changes and **submit a pull request**.  
   - You will receive an email asking about your changes—please reply.  
4. Thank you for helping improve **BambooChute**!

---

## License

**BambooChute** is released under the [MIT License](LICENSE). You’re free to use, modify, and distribute this library for personal or commercial projects, subject to the license terms.

---

**Happy Cleaning!** For additional examples or advanced usage, refer to the [documentation](https://pypi.org/project/BambooChute/) or explore the source code in this repo.