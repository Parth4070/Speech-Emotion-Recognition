from keras.layers import LSTM, Dense, Dropout
from keras.src import Input
from keras.models import Sequential
import numpy as np


X_train = np.load("../data/train/X_train.npy")
X_test = np.load("../data/train/X_test.npy")
y_train = np.load("../data/train/y_train.npy")
y_test = np.load("../data/train/y_test.npy")

timesteps = X_train.shape[1]
features = X_train.shape[2]
num_classes = len(np.unique(y_train))


# Defining Model
model = Sequential([
    Input(shape=(timesteps, features)),

    LSTM(64, return_sequences=True),
    Dropout(0.2),

    LSTM(32),
    Dropout(0.2),

    Dense(32, activation="relu"),
    Dense(num_classes, activation="softmax")
])


model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), verbose=1, batch_size=32)

model.save("../models/model1.keras")
