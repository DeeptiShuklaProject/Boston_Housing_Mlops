from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the trained model
model = joblib.load("ml/model.pkl")

# Input schema
class PredictionInput(BaseModel):
    features: list[float]

@app.post("/predict")
async def predict(input: PredictionInput):
    features = np.array(input.features).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": prediction.tolist()}
