from fastapi import FastAPI
from pydantic import BaseModel
from ml.predict import predict

app = FastAPI()


class InputData(BaseModel):
    features: list


@app.post("/predict")
def predict_price(input_data: InputData):
    prediction = predict(input_data.features)
    return {"predicted_price": prediction}
