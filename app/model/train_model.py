# train_model.py

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
import joblib
import os
import mlflow
import mlflow.sklearn

from preprocessing import build_pipeline, NUMERIC_FEATURES

# Load training data
df = pd.read_csv("data/hr_payroll_data.csv")
X = df[NUMERIC_FEATURES]

# Build preprocessing pipeline
pipeline = build_pipeline()
X_processed = pipeline.fit_transform(X)

# Train the model
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X_processed)

# ─── Save model & pipeline locally ──────────────────────────────────────────────
os.makedirs("app/model", exist_ok=True)
model_path = "app/model/isolation_forest_model.pkl"
pipeline_path = "app/model/payroll_scaler.pkl"
joblib.dump(model, model_path)
joblib.dump(pipeline, pipeline_path)

print("✅ Model and preprocessing pipeline saved.")

# ─── MLflow Logging ─────────────────────────────────────────────────────────────
mlflow.set_experiment("Payroll-Anomaly-Detection")

with mlflow.start_run():
    mlflow.log_param("model_type", "IsolationForest")
    mlflow.log_param("contamination", 0.1)

    # Log artifacts
    mlflow.log_artifact(model_path, artifact_path="model")
    mlflow.log_artifact(pipeline_path, artifact_path="pipeline")

    # Optional: Add placeholder metrics (since IsolationForest is unsupervised)
    mlflow.log_metric("num_training_rows", X.shape[0])
    mlflow.log_metric("num_features", X.shape[1])

    print("✅ Logged to MLflow.")
