import pandas as pd
from sklearn.impute import SimpleImputer

def handle_missing_values(df: pd.DataFrame):
    """
    Fill missing values using advanced imputation methods.
    """
    # Impute numerical columns using the median
    imputer = SimpleImputer(strategy='median')
    for col in df.select_dtypes(include=['float64', 'int64']):
        df[col] = imputer.fit_transform(df[[col]])

    # For categorical columns, use the most frequent value
    cat_imputer = SimpleImputer(strategy='most_frequent')
    for col in df.select_dtypes(include=['object']):
        df[col] = cat_imputer.fit_transform(df[[col]])

    return df
