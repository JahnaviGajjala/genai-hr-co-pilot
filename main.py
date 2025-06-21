from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from app.qa_co_pilot import answer_question  # <-- LLM Integration

# ─── Correct Base Path to Find Model Files ──────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "app", "model")

model_path = os.path.join(MODEL_DIR, "isolation_forest_model.pkl")  # ✅ corrected typo
scaler_path = os.path.join(MODEL_DIR, "payroll_scaler.pkl")

# ─── Load Model and Scaler ──────────────────────────────────────────────────────
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# ─── Create FastAPI App ─────────────────────────────────────────────────────────
app = FastAPI(title="Payroll Anomaly Detector")

# ─── Define Input Schema ────────────────────────────────────────────────────────
class PayrollInput(BaseModel):
    BaseSalary: float
    Bonus: float
    Deductions: float

# ─── Predict Endpoint ───────────────────────────────────────────────────────────
@app.post("/predict")
def predict(input_data: PayrollInput):
    input_dict = input_data.dict()
    input_df = pd.DataFrame([input_dict])
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)
    return {"anomaly": bool(prediction[0])}

# ─── GenAI HR Co-Pilot Endpoint ─────────────────────────────────────────────────
class Question(BaseModel):
    query: str

@app.post("/ask")
def ask_question(q: Question):
    try:
        response = answer_question(q.query)
        return {"answer": response}
    except Exception as e:
        return {"error": str(e)}
