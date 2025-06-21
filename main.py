import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor  # type: ignore
from sklearn.multioutput import MultiOutputRegressor  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore
from xgboost import XGBRegressor  # type: ignore
print("Loading data")
train = pd.read_parquet("train_data.parquet")
test = pd.read_parquet("test_data.parquet")
print("Identifying features and targets...")
target_cols = [col for col in train.columns if col.startswith("call_iv_")]
all_features = [col for col in train.columns if col not in target_cols]
print(" Aligning test set columns with training")
for col in all_features:
    if col not in test.columns:
        print(f"Adding missing column to test:{col}")
        test[col] = 0
print("Processing date column into numeric features...")
for df in [train, test]:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["date_timestamp"] = df["date"].astype(np.int64) // 10**9
        df["date_dayofweek"] = df["date"].dt.dayofweek
        df["date_dayofmonth"] = df["date"].dt.day
        df["date_month"] = df["date"].dt.month
        df["date_year"] = df["date"].dt.year
        df["date_dayofyear"] = df["date"].dt.dayofyear
feature_cols = [col for col in train.columns if col not in target_cols and col != "date"]
X_train = train[feature_cols].select_dtypes(include=[np.number])
y_train = train[target_cols]
X_test = test[feature_cols].select_dtypes(include=[np.number])

print(f"Train shape: {X_train.shape}, Targets: {len(target_cols)}")
print("Setting up stacked model")
base_models = [
    ("rf", RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)),
    ("gbr", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)),
]
meta_model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=8, random_state=42, n_jobs=-1)

stack = StackingRegressor(estimators=base_models, final_estimator=meta_model, n_jobs=-1)
model = MultiOutputRegressor(stack)
print("Splitting training data for validation...")
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
print(" Training model")
model.fit(X_tr, y_tr)

print("Evaluating model...")
val_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"\n Validation RMSE: {rmse:.10f}")

print("Retraining on full dataset...")
model.fit(X_train, y_train)

print(" Predicting on test set...")
test_predictions = model.predict(X_test)

print(" Saving predictions to submission.csv...")
submission = pd.DataFrame(test_predictions, columns=target_cols)
submission.to_csv("submission.csv", index=False)
print(" Submission file created.")
