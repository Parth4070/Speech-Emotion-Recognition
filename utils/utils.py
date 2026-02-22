import os
import numpy as np
import keras_tuner as kt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
import keras

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "train")

def load_data(file_name):
    path = os.path.join(DATA_DIR, f"{file_name}.npy")
    return np.load(path)


def build_model(hp, timesteps, features, num_classes):
    model = Sequential()

    model.add(Input(shape=(timesteps , features)))

    # tuning
    units = hp.Choice("lstm_units", [32, 64, 128])
    model.add(LSTM(units=units, return_sequences=True))

    dropout_rate = hp.Choice("dropout", [0.2, 0.3, 0.5])
    model.add(Dropout(dropout_rate))

    model.add(LSTM(units=units // 2))
    model.add(Dropout(dropout_rate))

    model.add(Dense(num_classes, activation="softmax"))

    lr = hp.Choice("learning_rate", [1e-3, 5e-4, 1e-4])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy']
    )

    return model

