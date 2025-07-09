from fastapi import FastAPI
import pickle
import numpy as np
import pandas as pd
from pydantic import BaseModel

# Definisi API
app = FastAPI()

# Load model dan scaler
with open("models/kmeans_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

with open("models/kmeans_scaler.pkl", "rb") as file:
    loaded_scaler = pickle.load(file)

# Definisi input data
class InputData(BaseModel):
    annual_spending: float
    purchase_frequency: float

@app.post("/predict")
def predict(data: InputData):
    try:
        # Konversi input ke DataFrame agar cocok dengan scaler yang sudah dilatih
        x_input = pd.DataFrame(
            [[data.annual_spending, data.purchase_frequency]],
            columns=["Annual Spending (USD)", "Purchase Frequency"]
        )

        # Normalisasi input
        x_scaled = loaded_scaler.transform(x_input)

        # Prediksi cluster
        cluster = loaded_model.predict(x_scaled)
        return {"cluster": int(cluster[0])}

    except Exception as e:
        return {"error": str(e)}