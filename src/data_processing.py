import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from src import config_loader as config


def create_dataset(data, window_size=config.WINDOW_SIZE):
    """Create sliding window dataset."""
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def process_data():
    """
    Load raw data, clean, scale, and create sliding windows.
    Save result files to data/final/ and models/
    """
    print("Starting data processing...")

    os.makedirs(os.path.dirname(config.PROCESSED_DATA_PATH), exist_ok=True)
    os.makedirs(config.FINAL_DATA_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    # 1. Load and basic cleaning
    try:
        df = pd.read_csv(config.RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at: {config.RAW_DATA_PATH}")
        print("Please ensure 'Gold Price (2013-2023).csv' is in 'data/raw/'")
        return

    df.drop(['Vol.', 'Change %'], axis=1, inplace=True)

    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    NumCols = df.columns.drop(['Date'])
    df[NumCols] = df[NumCols].replace({',': ''}, regex=True)
    df[NumCols] = df[NumCols].astype('float64')

    df.to_csv(config.PROCESSED_DATA_PATH, index=False)
    print(f"Saved cleaned data to: {config.PROCESSED_DATA_PATH}")

    # 2. Split Train/Test
    test_size = df[df.Date.dt.year == config.TEST_YEAR].shape[0]
    train_data_series = df.Price[:-test_size]

    # 3. Scaling
    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data_series.values.reshape(-1, 1))

    joblib.dump(scaler, config.SCALER_PATH)
    print(f"Saved scaler to: {config.SCALER_PATH}")

    # 4. Create sliding window for Train
    X_train, y_train = create_dataset(train_data_scaled)

    # 5. Create sliding window for Test
    test_data_series = df.Price[-test_size - config.WINDOW_SIZE:]  # <-- Use config
    test_data_scaled = scaler.transform(test_data_series.values.reshape(-1, 1))

    X_test, y_test = create_dataset(test_data_scaled)

    # 6. Reshape
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_train = np.reshape(y_train, (-1, 1))
    y_test = np.reshape(y_test, (-1, 1))

    print(f"Shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Shapes: X_test: {X_test.shape}, y_test: {y_test.shape}")

    # 7. Save numpy files
    np.save(os.path.join(config.FINAL_DATA_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(config.FINAL_DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(config.FINAL_DATA_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(config.FINAL_DATA_DIR, 'y_test.npy'), y_test)
    print(f"Saved .npy files to directory: {config.FINAL_DATA_DIR}")


if __name__ == "__main__":
    # Ensure this script is run from the root directory (e.g., python src/data_processing.py)
    process_data()