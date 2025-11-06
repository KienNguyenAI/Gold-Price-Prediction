import numpy as np
import os
from src.model import define_model
from src import config_loader as config


def train_model():
    """
    Load processed data and train the model.
    """
    print("Loading training data...")
    X_train = np.load(os.path.join(config.FINAL_DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(config.FINAL_DATA_DIR, 'y_train.npy'))

    print(f"Loaded X_train shape: {X_train.shape}")

    # 1. Define model
    model = define_model()
    model.summary()

    print("\nStarting training process...")

    # 2. Train model
    history = model.fit(
        X_train,
        y_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )

    # 3. Save model
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    model.save(config.MODEL_PATH)
    print(f"\nModel trained and saved at: {config.MODEL_PATH}")


if __name__ == "__main__":
    train_model()