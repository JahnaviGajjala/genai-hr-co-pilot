import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load data
df = pd.read_csv("data/hr_payroll_data.csv")

# Select features
features = ["BaseSalary", "Bonus", "Deductions", "NetPay"]
X = df[features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X_scaled)

# Create output dir if it doesn't exist
os.makedirs("app/model", exist_ok=True)

# Save model and scaler
joblib.dump(model, "app/model/isolation_forest_model.pkl")
joblib.dump(scaler, "app/model/payroll_scaler.pkl")

print("✅ Model and scaler saved successfully in app/model/")
