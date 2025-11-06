import numpy as np
import os
from model import define_model  # Import mô hình từ file model.py

# --- Thiết lập Cấu hình ---
FINAL_DATA_DIR = 'data/final'
MODEL_PATH = 'models/lstm_model.h5'
WINDOW_SIZE = 60
EPOCHS = 150  # Bạn đã chạy 150 epochs trong notebook
BATCH_SIZE = 32


def train_model():
    """
    Tải dữ liệu đã xử lý và huấn luyện mô hình.
    """
    print("Đang tải dữ liệu huấn luyện...")
    # Tải dữ liệu đã được xử lý bởi data_processing.py
    X_train = np.load(os.path.join(FINAL_DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(FINAL_DATA_DIR, 'y_train.npy'))
    X_test = np.load(os.path.join(FINAL_DATA_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(FINAL_DATA_DIR, 'y_test.npy'))

    print(f"Đã tải X_train shape: {X_train.shape}")

    # 1. Định nghĩa mô hình (từ cell 43)
    model = define_model(window_size=WINDOW_SIZE)
    model.summary()

    print("\nBắt đầu quá trình huấn luyện...")
    # 2. Huấn luyện mô hình (từ cell 43)
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,  # Tách 10% tập train làm validation
        verbose=1
    )

    # 3. Lưu mô hình
    os.makedirs(MODEL_PATH.rpartition('/')[0], exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nĐã huấn luyện và lưu mô hình tại: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()