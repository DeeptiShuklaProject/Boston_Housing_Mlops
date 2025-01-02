from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)


def test_predict():
    response = client.post("/predict", json={"features": [6.575, 65.2, 4.09, 1]})
    assert response.status_code == 200
    assert "predicted_price" in response.json()
