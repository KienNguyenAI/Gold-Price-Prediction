import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import load_model
import pandas as pd

# --- Thiết lập Cấu hình ---
FINAL_DATA_DIR = 'data/final'
MODEL_PATH = 'models/lstm_model.h5'
SCALER_PATH = 'models/price_scaler.pkl'
RAW_DATA_PATH = 'data/raw/gold_2013_2023.csv'
PLOT_PATH = 'reports/figures/actual_vs_predicted.png'
TEST_YEAR = 2022


def evaluate_model():
    """
    Tải mô hình đã huấn luyện, dự đoán trên tập test và báo cáo kết quả.
    """
    print("Đang tải mô hình và dữ liệu test...")
    # 1. Tải mô hình, scaler và dữ liệu test
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    X_test = np.load(os.path.join(FINAL_DATA_DIR, 'X_test.npy'))
    y_test_scaled = np.load(os.path.join(FINAL_DATA_DIR, 'y_test.npy'))

    # 2. Dự đoán (từ cell 45)
    y_pred_scaled = model.predict(X_test)

    # 3. Đánh giá (từ cell 46)
    mape = mean_absolute_percentage_error(y_test_scaled, y_pred_scaled)
    accuracy = 1 - mape

    print(f"--- Kết quả Đánh giá (trên dữ liệu scaled) ---")
    print(f"Test MAPE: {mape:.4f}")
    print(f"Test Accuracy: {accuracy:.4f} (hoặc {accuracy * 100:.2f}%)")

    # 4. Đảo ngược scale để xem kết quả thực tế (từ cell 49)
    y_test_true = scaler.inverse_transform(y_test_scaled)
    y_test_pred = scaler.inverse_transform(y_pred_scaled)

    mape_actual = mean_absolute_percentage_error(y_test_true, y_test_pred)
    accuracy_actual = 1 - mape_actual

    print(f"\n--- Kết quả Đánh giá (trên dữ liệu thực tế) ---")
    print(f"Test MAPE (Actual): {mape_actual:.4f}")
    print(
        f"Test Accuracy (Actual): {accuracy_actual:.4f} (hoặc {accuracy_actual * 100:.2f}%)")  # Kết quả 96% của bạn ở đây

    # 5. Vẽ biểu đồ và lưu file (từ cell 51)
    print("Đang vẽ biểu đồ kết quả...")

    # Lấy lại ngày tháng của tập test để vẽ
    df = pd.read_csv(RAW_DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    test_size = df[df.Date.dt.year == TEST_YEAR].shape[0]
    test_dates = df['Date'].iloc[-test_size:]

    os.makedirs(PLOT_PATH.rpartition('/')[0], exist_ok=True)
    plt.figure(figsize=(15, 6), dpi=150)
    plt.rcParams['axes.facecolor'] = 'yellow'
    plt.rc('axes', edgecolor='white')

    plt.plot(test_dates, y_test_true, color='blue', lw=2, label='Actual Test Data')
    plt.plot(test_dates, y_test_pred, color='red', lw=2, label='Predicted Test Data')

    plt.title('Model Performance on Gold Price Prediction (Accuracy: {:.2f}%)'.format(accuracy_actual * 100),
              fontsize=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(loc='upper left', prop={'size': 12})
    plt.grid(color='white')

    plt.savefig(PLOT_PATH)
    print(f"Đã lưu biểu đồ kết quả tại: {PLOT_PATH}")


if __name__ == "__main__":
    evaluate_model()