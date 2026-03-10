from keras.layers import LSTM, Dense, Dropout
from keras.src import Input
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
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

mean = np.mean(X_train, axis=(0,1), keepdims=True)
std = np.std(X_train, axis=(0,1), keepdims=True)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

print(X_train.shape, X_test.shape)

timesteps = X_train.shape[1]
features = X_train.shape[2]

num_classes = len(np.unique(y_test))

earlystop = EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True
)

lr_schedular = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    verbose=1
)

callbacks = [earlystop, lr_schedular]

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
    callbacks = callbacks
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)

history = best_model.fit(
    X_train,
    y_train,
    validation_data = (X_test, y_test),
    epochs = 50,
    batch_size = 32,
    callbacks = callbacks
)

best_model.save(os.path.join(BASE_DIR, "models", "best_model.keras"))
