import tensorflow as tf
from keras import Model
from keras.layers import Input, Dense, Dropout, LSTM
from src import config_loader as config  # <-- THAY ĐỔI Ở ĐÂY


def define_model(window_size=config.WINDOW_SIZE):  # <-- Dùng config
    """
    Định nghĩa kiến trúc mô hình LSTM.
    """
    input1 = Input(shape=(window_size, 1))
    x = LSTM(units=64, return_sequences=True)(input1)
    x = Dropout(0.2)(x)
    x = LSTM(units=64, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(units=64)(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='softmax')(x)
    dnn_output = Dense(1)(x)

    model = Model(inputs=input1, outputs=[dnn_output])
    model.compile(loss='mean_squared_error', optimizer='Nadam')

    return model


if __name__ == "__main__":
    model = define_model()
    model.summary()