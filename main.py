from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from app.qa_co_pilot import answer_question
from retrain import retrain_model  # 🔹 import retraining function

# ─── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "app", "model")
model_path = os.path.join(MODEL_DIR, "isolation_forest_model.pkl")
scaler_path = os.path.join(MODEL_DIR, "payroll_scaler.pkl")

# ─── Load Model and Scaler ──────────────────────────────────────────────────────
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# ─── Create FastAPI App ─────────────────────────────────────────────────────────
app = FastAPI(title="Payroll Anomaly Detector")

# ─── Input Schema ───────────────────────────────────────────────────────────────
class PayrollInput(BaseModel):
    BaseSalary: float
    Bonus: float
    Deductions: float

class Question(BaseModel):
    query: str

# ─── Predict Endpoint ───────────────────────────────────────────────────────────
@app.post("/predict")
def predict(input_data: PayrollInput):
    input_dict = input_data.dict()
    input_df = pd.DataFrame([input_dict])
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)
    return {"anomaly": bool(prediction[0])}

# ─── LLM HR Co-Pilot ─────────────────────────────────────────────────────────────
@app.post("/ask")
def ask_question(q: Question):
    try:
        response = answer_question(q.query)
        return {"answer": response}
    except Exception as e:
        return {"error": str(e)}

# ─── Retrain Endpoint ───────────────────────────────────────────────────────────
@app.post("/retrain")
def retrain():
    try:
        result = retrain_model()
        # 🔄 Reload updated model and scaler into memory
        global model, scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}
