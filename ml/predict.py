import joblib
import numpy as np


def load_model():
    return joblib.load("ml/model.pkl")


def predict(features):
    model = load_model()
    features = np.array(features).reshape(1, -1)
    return model.predict(features)[0]
