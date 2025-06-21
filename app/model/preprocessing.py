# app/model/preprocessing.py

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pandas as pd

# Columns expected for processing
NUMERIC_FEATURES = ['BaseSalary', 'Bonus', 'Deductions']

def build_pipeline():
    # Numeric preprocessing pipeline
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Full column transformer (currently numeric only)
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, NUMERIC_FEATURES)
    ])

    return preprocessor
