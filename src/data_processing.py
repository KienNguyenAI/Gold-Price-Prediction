import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


RAW_DATA_PATH = 'data/raw/gold_2013_2023.csv'
PROCESSED_DATA_PATH = 'data/processed/gold_clean_2013_2023.csv'
FINAL_DATA_DIR = 'data/final'
SCALER_PATH = 'models/price_scaler.pkl'

TEST_YEAR = 2022
WINDOW_SIZE = 60

def create_dataset(data, window_size=60):
    """Tạo bộ dữ liệu cửa sổ trượt."""
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def process_data():
    print("Bắt đầu xử lý dữ liệu...")
    os.makedirs(PROCESSED_DATA_PATH.rpartition('/')[0], exist_ok=True)
    os.makedirs(FINAL_DATA_DIR, exist_ok=True)
    os.makedirs(SCALER_PATH.rpartition('/')[0], exist_ok=True)

    # 1. Tải và làm sạch (từ cells 7, 12, 14, 16)
    df = pd.read_csv(RAW_DATA_PATH)
    df.drop(['Vol.', 'Change %'], axis=1, inplace=True)

    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    NumCols = df.columns.drop(['Date'])
    df[NumCols] = df[NumCols].replace({',': ''}, regex=True)
    df[NumCols] = df[NumCols].astype('float64')

    # Lưu file đã làm sạch
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Đã lưu dữ liệu đã làm sạch vào: {PROCESSED_DATA_PATH}")

    # 2. Tách Train/Test
    test_size = df[df.Date.dt.year == TEST_YEAR].shape[0]
    train_data_series = df.Price[:-test_size]

    # 3. Chuẩn hóa (Scaling) - Sửa lỗi rò rỉ dữ liệu từ notebook
    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data_series.values.reshape(-1, 1))

    # Lưu scaler lại để dùng cho predict.py
    joblib.dump(scaler, SCALER_PATH)
    print(f"Đã lưu scaler vào: {SCALER_PATH}")

    # 4. Tạo cửa sổ trượt cho Train
    X_train, y_train = create_dataset(train_data_scaled, WINDOW_SIZE)

    # 5. Tạo cửa sổ trượt  cho Test
    test_data_series = df.Price[-test_size - WINDOW_SIZE:]
    test_data_scaled = scaler.transform(test_data_series.values.reshape(-1, 1))

    X_test, y_test = create_dataset(test_data_scaled, WINDOW_SIZE)

    # 6. Reshape (từ cell 39)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_train = np.reshape(y_train, (-1, 1))
    y_test = np.reshape(y_test, (-1, 1))

    print(f"Shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Shapes: X_test: {X_test.shape}, y_test: {y_test.shape}")

    # 7. Lưu các file numpy
    np.save(os.path.join(FINAL_DATA_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(FINAL_DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(FINAL_DATA_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(FINAL_DATA_DIR, 'y_test.npy'), y_test)
    print(f"Đã lưu các file .npy vào thư mục: {FINAL_DATA_DIR}")


if __name__ == "__main__":
    process_data()