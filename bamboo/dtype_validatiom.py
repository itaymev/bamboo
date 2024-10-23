import pandas as pd

def correct_data_types(df: pd.DataFrame):
    """
    Automatically detect and correct incorrect data types.
    """
    # Convert columns to datetime where possible
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col], errors='ignore')
            except (ValueError, TypeError):
                continue  # Keep original type if conversion fails
    return df
