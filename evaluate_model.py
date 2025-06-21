# evaluate_model.py

import pandas as pd
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os
from app.model.preprocessing import NUMERIC_FEATURES

# Load labeled data
df = pd.read_csv("data/hr_payroll_data.csv")

# Drop rows without a Label
df = df.dropna(subset=['Label'])

X = df[NUMERIC_FEATURES]
y_true = df['Label']

# Load model and preprocessing pipeline
model = joblib.load("app/model/isolation_forest_model.pkl")
pipeline = joblib.load("app/model/payroll_scaler.pkl")

# Preprocess features
X_scaled = pipeline.transform(X)

# Predict using IsolationForest
y_pred = model.predict(X_scaled)

# Convert prediction output: -1 = anomaly → 1, 1 = normal → 0
y_pred = [1 if x == -1 else 0 for x in y_pred]

# Evaluate metrics
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

# Print results
print("\n📊 Model Evaluation Metrics:")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1 Score:  {f1:.2f}")
print(f"Accuracy:  {accuracy:.2f}")

# Log to file
os.makedirs("logs", exist_ok=True)
with open("logs/evaluation.log", "w") as f:
    f.write(f"Precision: {precision:.2f}\n")
    f.write(f"Recall:    {recall:.2f}\n")
    f.write(f"F1 Score:  {f1:.2f}\n")
    f.write(f"Accuracy:  {accuracy:.2f}\n")

print("✅ Results saved to logs/evaluation.log")
