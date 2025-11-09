import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import load_model
import pandas as pd
from src import config_loader as config


def evaluate_model():

    print("Loading model and test data...")
    model = load_model(config.MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    X_test = np.load(os.path.join(config.FINAL_DATA_DIR, 'X_test.npy'))
    y_test_scaled = np.load(os.path.join(config.FINAL_DATA_DIR, 'y_test.npy'))

    # 2. Predict
    y_pred_scaled = model.predict(X_test)

    # 3. Evaluate (scaled)
    mape = mean_absolute_percentage_error(y_test_scaled, y_pred_scaled)
    accuracy = 1 - mape
    print(f"Test Accuracy (scaled): {accuracy * 100:.2f}%")

    # 4. Inverse scale
    y_test_true = scaler.inverse_transform(y_test_scaled)
    y_test_pred = scaler.inverse_transform(y_pred_scaled)

    accuracy_actual = 1 - mean_absolute_percentage_error(y_test_true, y_test_pred)
    print(f"Test Accuracy (Actual): {accuracy_actual * 100:.2f}%")

    # 5. Plot graph and save file
    print("Plotting results graph...")
    df = pd.read_csv(config.RAW_DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    test_size = df[df.Date.dt.year == config.TEST_YEAR].shape[0]
    test_dates = df['Date'].iloc[-test_size:]

    os.makedirs(config.FIGURE_DIR, exist_ok=True)
    plt.figure(figsize=(15, 6), dpi=150)

    plt.plot(test_dates, y_test_true, color='blue', lw=2, label='Actual Test Data')
    plt.plot(test_dates, y_test_pred, color='red', lw=2, label='Predicted Test Data')

    plt.title(f'Model Performance (Accuracy: {accuracy_actual * 100:.2f}%)', fontsize=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(loc='upper left', prop={'size': 12})
    plt.grid(color='black')

    plt.savefig(config.PLOT_PATH)
    print(f"Saved results plot at: {config.PLOT_PATH}")


if __name__ == "__main__":
    evaluate_model()