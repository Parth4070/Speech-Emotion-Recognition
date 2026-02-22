from keras.layers import LSTM, Dense, Dropout
from keras.src import Input
from keras.models import Sequential
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils.utils import load_data, build_model
import keras_tuner as kt
from keras.callbacks import EarlyStopping

X_train = load_data("X_train")
X_test = load_data("X_test")
y_train = load_data("y_train")
y_test = load_data("y_test")

timesteps = X_train.shape[1]
features = X_train.shape[2]

num_classes = len(np.unique(y_test))

earlystop = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

tuner = kt.RandomSearch(
    lambda hp: build_model(hp, timesteps, features, num_classes),
    objective="val_accuracy",
    max_trials=10,
    directory = "tuning",
    project_name = "ser_lstm"
)

tuner.search(
    X_train,
    y_train,
    validation_data = (X_test, y_test),
    epochs = 20,
    batch_size = 32,
    callbacks = [earlystop]
)

best_model = tuner.get_best_models(1)[0]

history = best_model.fit(
    X_train,
    y_train,
    validation_data = (X_test, y_test),
    epochs = 50,
    batch_size = 32,
    callbacks = [earlystop]
)

best_model.save(os.path.join(BASE_DIR, "models", "model2.keras"))
