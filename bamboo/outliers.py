import pandas as pd
import numpy as np

def detect_outliers(df: pd.DataFrame, threshold=3.0):
    """
    Detect outliers using the Z-score method.
    """
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include=[np.number]):
        z_scores = (df_clean[col] - df_clean[col].mean()) / df_clean[col].std()
        df_clean[col] = df_clean[col].where(np.abs(z_scores) < threshold, np.nan)  # Replace outliers with NaN
    return df_clean
