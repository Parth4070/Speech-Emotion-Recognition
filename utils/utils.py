import os
import numpy as np
import keras_tuner as kt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from keras.regularizers import l2
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
    units = hp.Choice("lstm_units", [32])
    model.add(Bidirectional(LSTM(units=units, return_sequences=True, kernel_regularizer=l2(0.001))))

    dropout_rate = hp.Choice("dropout", [0.4])
    model.add(Dropout(dropout_rate))

    model.add(Bidirectional(LSTM(units=units // 2, kernel_regularizer=l2(0.001))))
    model.add(Dropout(dropout_rate))

    model.add(Dense(num_classes, activation="softmax"))     

    lr = hp.Choice("learning_rate", [ 5e-4 ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy']
    )

    return model

