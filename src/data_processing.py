import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from src import config_loader as config  # <-- THAY ĐỔI Ở ĐÂY


# --- Hàm trợ giúp (không đổi) ---
def create_dataset(data, window_size=config.WINDOW_SIZE):  # <-- Dùng config
    """Tạo bộ dữ liệu cửa sổ trượt."""
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


# --- Hàm chính ---
def process_data():
    """
    Tải dữ liệu thô, làm sạch, chuẩn hóa và tạo cửa sổ trượt.
    Lưu các file kết quả vào data/final/ và models/
    """
    print("Bắt đầu xử lý dữ liệu...")

    # Đảm bảo các thư mục tồn tại (dùng biến từ config)
    os.makedirs(os.path.dirname(config.PROCESSED_DATA_PATH), exist_ok=True)
    os.makedirs(config.FINAL_DATA_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    # 1. Tải và làm sạch cơ bản
    try:
        df = pd.read_csv(config.RAW_DATA_PATH)  # <-- Dùng config
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file dữ liệu thô tại: {config.RAW_DATA_PATH}")
        print("Vui lòng đảm bảo file 'Gold Price (2013-2023).csv' nằm trong 'data/raw/'")
        return

    df.drop(['Vol.', 'Change %'], axis=1, inplace=True)

    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    NumCols = df.columns.drop(['Date'])
    df[NumCols] = df[NumCols].replace({',': ''}, regex=True)
    df[NumCols] = df[NumCols].astype('float64')

    df.to_csv(config.PROCESSED_DATA_PATH, index=False)  # <-- Dùng config
    print(f"Đã lưu dữ liệu đã làm sạch vào: {config.PROCESSED_DATA_PATH}")

    # 2. Tách Train/Test
    test_size = df[df.Date.dt.year == config.TEST_YEAR].shape[0]  # <-- Dùng config
    train_data_series = df.Price[:-test_size]

    # 3. Chuẩn hóa (Scaling)
    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data_series.values.reshape(-1, 1))

    joblib.dump(scaler, config.SCALER_PATH)  # <-- Dùng config
    print(f"Đã lưu scaler vào: {config.SCALER_PATH}")

    # 4. Tạo cửa sổ trượt (windowing) cho Train
    X_train, y_train = create_dataset(train_data_scaled)  # window_size đã có trong hàm

    # 5. Tạo cửa sổ trượt (windowing) cho Test
    test_data_series = df.Price[-test_size - config.WINDOW_SIZE:]  # <-- Dùng config
    test_data_scaled = scaler.transform(test_data_series.values.reshape(-1, 1))

    X_test, y_test = create_dataset(test_data_scaled)

    # 6. Reshape
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_train = np.reshape(y_train, (-1, 1))
    y_test = np.reshape(y_test, (-1, 1))

    print(f"Shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Shapes: X_test: {X_test.shape}, y_test: {y_test.shape}")

    # 7. Lưu các file numpy
    np.save(os.path.join(config.FINAL_DATA_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(config.FINAL_DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(config.FINAL_DATA_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(config.FINAL_DATA_DIR, 'y_test.npy'), y_test)
    print(f"Đã lưu các file .npy vào thư mục: {config.FINAL_DATA_DIR}")


if __name__ == "__main__":
    # Đảm bảo chạy script này từ thư mục gốc (vd: python src/data_processing.py)
    process_data()