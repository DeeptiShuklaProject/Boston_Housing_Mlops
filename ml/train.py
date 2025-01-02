import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib


def train_model():
    # Load data
    df = pd.read_csv("data/dataset.csv")

    # Prepare features and target
    X = df.drop("price", axis=1)
    y = df["price"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model Mean Squared Error: {mse}")

    # Save the model
    joblib.dump(model, "ml/model.pkl")


if __name__ == "__main__":
    train_model()
