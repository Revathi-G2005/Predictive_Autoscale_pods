import pandas as pd

def create_lags(df: pd.DataFrame, columns, lags=(1,2,3)):
    df = df.sort_values('timestamp').copy()
    for col in columns:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df.dropna().reset_index(drop=True)