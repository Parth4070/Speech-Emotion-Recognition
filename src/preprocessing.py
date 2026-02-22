import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

encoder = LabelEncoder()

X = np.load(os.path.join(BASE_DIR, "data", "processed", "X.npy"))
y = np.load(os.path.join(BASE_DIR, "data", "processed", "y.npy"))
actors = np.load(os.path.join(BASE_DIR, "data", "processed", "actors.npy"))

unique_actors = np.unique(actors)
np.random.seed(42)
np.random.shuffle(unique_actors)

split_idx = int(len(unique_actors) * 0.8)

train_actors = unique_actors[:split_idx]
test_actors = unique_actors[split_idx:]

train_mask = np.isin(actors, train_actors)
test_mask = np.isin(actors, test_actors)

X_train = X[train_mask]
y_train = y[train_mask]

X_test = X[test_mask]
y_test = y[test_mask]

y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# print(X_train.shape, y_train.shape)
train_dir = os.path.join(BASE_DIR, "data", "train")
os.makedirs(train_dir, exist_ok=True)
np.save(os.path.join(train_dir, "X_train.npy"), X_train)
np.save(os.path.join(train_dir, "y_train.npy"), y_train)
np.save(os.path.join(train_dir, "X_test.npy"), X_test)
np.save(os.path.join(train_dir, "y_test.npy"), y_test)
