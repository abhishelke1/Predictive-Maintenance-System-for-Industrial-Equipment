from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

app = FastAPI(title="Predictive Maintenance API")

# Load Model
with open('rul_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define Input Schema
class EngineData(BaseModel):
    s7: float
    s12: float
    s21: float
    s7_mean: float
    s7_std: float
    s12_mean: float
    s12_std: float
    s21_mean: float
    s21_std: float

# --- 1. THE MISSING PIECE (Home Route) ---
@app.get("/")
def home():
    return {"message": "Predictive Maintenance API is Online!"}

# --- 2. The Prediction Route (With the Fix) ---
@app.post("/predict")
def predict_rul(data: EngineData):
    # Convert JSON to DataFrame
    input_data = pd.DataFrame([data.dict()])
    
    # Enforce Column Order (Must match training exactly)
    expected_order = [
        's7', 's12', 's21', 
        's7_mean', 's12_mean', 's21_mean', 
        's7_std', 's12_std', 's21_std'
    ]
    
    # Re-arrange columns
    input_data = input_data[expected_order]
    
    # Make Prediction
    prediction = model.predict(input_data)
    
    return {
        "predicted_RUL": float(prediction[0]),
        "status": "Critical" if prediction[0] < 30 else "Normal"
    }

# To run this server, you will use the terminal command below.
# uvicorn 3_api:app --reload
#if not then use this according ur system 
#& C:/Users/User/AppData/Local/Microsoft/WindowsApps/python3.13.exe -m uvicorn 3_api:app --reload