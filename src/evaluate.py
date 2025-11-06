import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import load_model
import pandas as pd
from src import config_loader as config  # <-- THAY ĐỔI Ở ĐÂY


def evaluate_model():
    """
    Tải mô hình đã huấn luyện, dự đoán trên tập test và báo cáo kết quả.
    """
    print("Đang tải mô hình và dữ liệu test...")
    model = load_model(config.MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    X_test = np.load(os.path.join(config.FINAL_DATA_DIR, 'X_test.npy'))
    y_test_scaled = np.load(os.path.join(config.FINAL_DATA_DIR, 'y_test.npy'))

    # 2. Dự đoán
    y_pred_scaled = model.predict(X_test)

    # 3. Đánh giá (scaled)
    mape = mean_absolute_percentage_error(y_test_scaled, y_pred_scaled)
    accuracy = 1 - mape
    print(f"Test Accuracy (scaled): {accuracy * 100:.2f}%")

    # 4. Đảo ngược scale
    y_test_true = scaler.inverse_transform(y_test_scaled)
    y_test_pred = scaler.inverse_transform(y_pred_scaled)

    accuracy_actual = 1 - mean_absolute_percentage_error(y_test_true, y_test_pred)
    print(f"Test Accuracy (Actual): {accuracy_actual * 100:.2f}%")

    # 5. Vẽ biểu đồ và lưu file
    print("Đang vẽ biểu đồ kết quả...")
    df = pd.read_csv(config.RAW_DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    test_size = df[df.Date.dt.year == config.TEST_YEAR].shape[0]
    test_dates = df['Date'].iloc[-test_size:]

    os.makedirs(config.FIGURE_DIR, exist_ok=True)
    plt.figure(figsize=(15, 6), dpi=150)
    plt.rcParams['axes.facecolor'] = 'yellow'
    plt.rc('axes', edgecolor='white')

    plt.plot(test_dates, y_test_true, color='blue', lw=2, label='Actual Test Data')
    plt.plot(test_dates, y_test_pred, color='red', lw=2, label='Predicted Test Data')

    plt.title(f'Model Performance (Accuracy: {accuracy_actual * 100:.2f}%)', fontsize=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(loc='upper left', prop={'size': 12})
    plt.grid(color='white')

    plt.savefig(config.PLOT_PATH)
    print(f"Đã lưu biểu đồ kết quả tại: {config.PLOT_PATH}")


if __name__ == "__main__":
    evaluate_model()