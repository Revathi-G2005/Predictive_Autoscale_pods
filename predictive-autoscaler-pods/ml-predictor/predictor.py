# ml_predictor/predictor.py
import pandas as pd
import joblib
from feature_pipeline import create_lags

class MLPredictor:
    def __init__(self, model_path="model.pkl", scaler_path="scaler.pkl"):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
    
    def predict(self, data: pd.DataFrame):
        cols = ["cpu_usage", "memory_usage", "requests_per_second"]
        data = create_lags(data, cols, lags=(1,2,3))

        for col in cols:
            data[f"{col}_roll_mean3"] = data[col].rolling(3).mean()
            data[f"{col}_roll_std3"] = data[col].rolling(3).std()
            data[f"{col}_diff"] = data[col].diff()

        data = data.dropna()
        X = data.drop(["timestamp", "replicas"], axis=1, errors="ignore")

        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        return preds
