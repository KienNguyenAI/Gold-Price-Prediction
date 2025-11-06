import numpy as np
import os
from src.model import define_model
from src import config_loader as config  # <-- THAY ĐỔI Ở ĐÂY


def train_model():
    """
    Tải dữ liệu đã xử lý và huấn luyện mô hình.
    """
    print("Đang tải dữ liệu huấn luyện...")
    X_train = np.load(os.path.join(config.FINAL_DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(config.FINAL_DATA_DIR, 'y_train.npy'))

    print(f"Đã tải X_train shape: {X_train.shape}")

    # 1. Định nghĩa mô hình
    model = define_model()  # window_size đã được đặt mặc định từ config
    model.summary()

    print("\nBắt đầu quá trình huấn luyện...")
    # 2. Huấn luyện mô hình
    history = model.fit(
        X_train,
        y_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )

    # 3. Lưu mô hình
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    model.save(config.MODEL_PATH)
    print(f"\nĐã huấn luyện và lưu mô hình tại: {config.MODEL_PATH}")


if __name__ == "__main__":
    train_model()