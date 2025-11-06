import numpy as np
import pandas as pd
import joblib
import os
from keras.models import load_model
from src import config_loader as config  # <-- THAY ĐỔI Ở ĐÂY


def predict_next_day():
    """
    Dự đoán giá vàng của ngày tiếp theo dựa trên 60 ngày cuối cùng.
    """
    print("Đang tải mô hình, scaler và dữ liệu cuối cùng...")
    try:
        model = load_model(config.MODEL_PATH)
        scaler = joblib.load(config.SCALER_PATH)
        df = pd.read_csv(config.PROCESSED_DATA_PATH)
    except IOError as e:
        print(f"Lỗi: Không tìm thấy file. Bạn đã chạy 'python src/data_processing.py' và 'python src/train.py' chưa?")
        print(e)
        return

    # 1. Lấy 60 ngày dữ liệu cuối cùng
    last_60_days = df['Price'].values[-config.WINDOW_SIZE:]

    # 2. Chuẩn hóa (Scale) dữ liệu
    last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))

    # 3. Reshape để đưa vào mô hình
    X_pred = np.reshape(last_60_days_scaled, (1, config.WINDOW_SIZE, 1))

    # 4. Dự đoán
    predicted_price_scaled = model.predict(X_pred)

    # 5. Đảo ngược scale để có giá trị thực
    predicted_price = scaler.inverse_transform(predicted_price_scaled)

    print("\n--- Dự đoán giá vàng ---")
    print(f"Giá dự đoán cho ngày tiếp theo là: {predicted_price[0][0]:.2f}")


if __name__ == "__main__":
    predict_next_day()