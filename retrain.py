# retrain.py

import pandas as pd
import joblib
import os
import mlflow
from sklearn.ensemble import IsolationForest

from app.model.preprocessing import build_pipeline, NUMERIC_FEATURES

MODEL_DIR = "app/model"
DATA_PATH = "data/hr_payroll_data.csv"

def retrain_model():
    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC_FEATURES]

    # Build preprocessing pipeline
    pipeline = build_pipeline()
    X_processed = pipeline.fit_transform(X)

    # Train model
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_processed)

    # Save artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "isolation_forest_model.pkl")
    pipeline_path = os.path.join(MODEL_DIR, "payroll_scaler.pkl")
    joblib.dump(model, model_path)
    joblib.dump(pipeline, pipeline_path)

    # Log to MLflow
    mlflow.set_experiment("Payroll-Anomaly-Detection")
    with mlflow.start_run():
        mlflow.log_param("model_type", "IsolationForest")
        mlflow.log_param("contamination", 0.1)
        mlflow.log_metric("num_training_rows", X.shape[0])
        mlflow.log_metric("num_features", X.shape[1])
        mlflow.log_artifact(model_path, artifact_path="model")
        mlflow.log_artifact(pipeline_path, artifact_path="pipeline")

    return {
        "status": "success",
        "message": "Model retrained and saved.",
        "rows": X.shape[0],
        "features": X.shape[1]
    }
