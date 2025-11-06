import numpy as np
import pandas as pd
import joblib
from keras.models import load_model
from src import config_loader as config


def predict_next_day():
    """
    Predict the next day's gold price based on the last 60 days.
    """
    print("Loading model, scaler, and final data...")
    try:
        model = load_model(config.MODEL_PATH)
        scaler = joblib.load(config.SCALER_PATH)
        df = pd.read_csv(config.PROCESSED_DATA_PATH)
    except IOError as e:
        print(f"Error: File not found. Have you run 'python src/data_processing.py' and 'python src/train.py'?")
        print(e)
        return

    # 1. Get last 60 days data
    last_60_days = df['Price'].values[-config.WINDOW_SIZE:]

    # 2. Scale the data
    last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))

    # 3. Reshape for model input
    X_pred = np.reshape(last_60_days_scaled, (1, config.WINDOW_SIZE, 1))

    # 4. Predict
    predicted_price_scaled = model.predict(X_pred)

    # 5. Inverse scale to get actual value
    predicted_price = scaler.inverse_transform(predicted_price_scaled)

    print("\n--- Gold Price Prediction ---")
    print(f"Predicted price for the next day is: {predicted_price[0][0]:.2f}")


if __name__ == "__main__":
    predict_next_day()