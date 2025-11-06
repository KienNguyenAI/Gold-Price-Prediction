import numpy as np
import pandas as pd
import joblib
import os
from keras.models import load_model

# --- Thiết lập Cấu hình ---
PROCESSED_DATA_PATH = 'data/processed/gold_clean_2013_2023.csv'
MODEL_PATH = 'models/lstm_model.h5'
SCALER_PATH = 'models/price_scaler.pkl'
WINDOW_SIZE = 60


def predict_next_day():
    print("Đang tải mô hình, scaler và dữ liệu cuối cùng...")
    try:
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        df = pd.read_csv(PROCESSED_DATA_PATH)
    except IOError as e:
        print(f"Lỗi: Không tìm thấy file. Bạn đã chạy 'python src/data_processing.py' và 'python src/train.py' chưa?")
        print(e)
        return

    # 1. Lấy 60 ngày dữ liệu cuối cùng
    last_60_days = df['Price'].values[-WINDOW_SIZE:]

    # 2. Chuẩn hóa (Scale) dữ liệu
    last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))

    # 3. Reshape để đưa vào mô hình
    # Mô hình cần input shape: (1, 60, 1)
    X_pred = np.reshape(last_60_days_scaled, (1, WINDOW_SIZE, 1))

    # 4. Dự đoán
    predicted_price_scaled = model.predict(X_pred)

    # 5. Đảo ngược scale để có giá trị thực
    predicted_price = scaler.inverse_transform(predicted_price_scaled)

    print("\n--- Dự đoán giá vàng ---")
    print(f"Giá dự đoán cho ngày tiếp theo là: {predicted_price[0][0]:.2f}")


if __name__ == "__main__":
    predict_next_day()