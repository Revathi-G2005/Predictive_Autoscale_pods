import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Try XGBoost first
try:
    from xgboost import XGBRegressor
    ModelClass = XGBRegressor
    model_kwargs = {"n_estimators": 100, "random_state": 0}
except Exception:
    from sklearn.ensemble import RandomForestRegressor
    ModelClass = RandomForestRegressor
    model_kwargs = {"n_estimators": 100, "random_state": 0, "n_jobs": -1}

from feature_pipeline import create_lags

# ------------------------------
# 1. Load & preprocess data
# ------------------------------
data_path = os.path.join("data", "metrics.csv")
data = pd.read_csv(data_path)

# Try to parse timestamp
for col in data.columns:
    if "time" in col.lower():
        data[col] = pd.to_datetime(data[col], errors="coerce")

# Map expected columns
if "CPU" in data.columns and "cpu_usage" not in data.columns:
    data["cpu_usage"] = pd.to_numeric(data["CPU"].astype(str).str.replace("%", ""), errors="coerce")
if "PackRecv" in data.columns and "requests_per_second" not in data.columns:
    data["requests_per_second"] = pd.to_numeric(data["PackRecv"], errors="coerce")
if "PodsNumber" in data.columns and "replicas" not in data.columns:
    data["replicas"] = pd.to_numeric(data["PodsNumber"], errors="coerce")

# Ensure basic numeric columns exist
for col in ["cpu_usage", "memory_usage", "requests_per_second", "replicas"]:
    if col not in data.columns:
        data[col] = 0.0 if col != "replicas" else 0

# Create lag/rolling features
cols = ["cpu_usage", "memory_usage", "requests_per_second"]
data = create_lags(data, cols, lags=(1, 2, 3))
for col in cols:
    data[f"{col}_roll_mean3"] = data[col].rolling(3).mean()
    data[f"{col}_roll_std3"] = data[col].rolling(3).std()
    data[f"{col}_diff"] = data[col].diff()

data = data.dropna().reset_index(drop=True)

# Drop timestamp-like columns
time_cols = [c for c in data.columns if "time" in c.lower()]
if time_cols:
    print(f"⚠️ Dropping timestamp columns: {time_cols}")
    data.drop(columns=time_cols, inplace=True)

# Drop any remaining non-numeric columns
non_numeric = data.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print(f"⚠️ Dropping non-numeric columns: {non_numeric}")
    data.drop(columns=non_numeric, inplace=True)

# ------------------------------
# 2. Prepare X and y
# ------------------------------
y = pd.to_numeric(data["replicas"], errors="coerce").fillna(0).astype(int)
X = data.drop(columns=["replicas"], errors="ignore")

# Keep only numeric features to avoid errors
X = X.select_dtypes(include=[np.number])
X = X.fillna(0)

print("✅ Final columns used for training:", X.columns.tolist())

# ------------------------------
# 3. Time-series CV + Scaling
# ------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

tscv = TimeSeriesSplit(n_splits=5)
best_model = None
best_score = -np.inf

# ------------------------------
# 4. Model training & tuning
# ------------------------------
for lr in [0.01, 0.05, 0.1]:
    for depth in [3, 5, 7]:
        if ModelClass.__name__ == "XGBRegressor":
            model = ModelClass(n_estimators=100, random_state=0,
                               learning_rate=lr, max_depth=depth)
        else:
            model = ModelClass(n_estimators=100, random_state=0,
                               max_depth=depth, n_jobs=-1)

        scores = []
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            scores.append(r2_score(y_test, y_pred))

        avg_score = np.mean(scores)
        print(f"lr={lr}, depth={depth}, r2={avg_score:.3f}")
        if avg_score > best_score:
            best_score = avg_score
            best_model = model

# ------------------------------
# 5. Final evaluation
# ------------------------------
y_pred = best_model.predict(X_scaled)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"\n✅ Final R²: {r2:.3f}, MSE: {mse:.4f}")

# ------------------------------
# 6. Save model and scaler
# ------------------------------
joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Model & scaler saved successfully.")

