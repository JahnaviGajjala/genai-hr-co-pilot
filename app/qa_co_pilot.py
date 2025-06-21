# app/qa_co_pilot.py

from transformers import pipeline
import pandas as pd
import os

# Load the payroll CSV
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'hr_payroll_data.csv')
df = pd.read_csv(DATA_PATH)

# Convert the data into a plain-text context paragraph
def generate_context(df):
    context = ""
    for _, row in df.iterrows():
        context += f"Employee has BaseSalary {row['BaseSalary']}, Bonus {row['Bonus']}, and Deductions {row['Deductions']}. "
    return context[:2000]  # Keep it within LLM limits

# Load the QA pipeline using PyTorch explicitly
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", framework="pt")

# Generate the answer
def answer_question(question: str) -> str:
    context = generate_context(df)
    result = qa_pipeline(question=question, context=context)
    return result['answer']
