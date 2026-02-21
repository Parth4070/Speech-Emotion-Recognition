import numpy as np
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

X = np.load("../data/processed/X.npy")
y = np.load("../data/processed/y.npy")
actors = np.load("../data/processed/actors.npy")

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
np.save("../data/train/X_train.npy", X_train)
np.save("../data/train/y_train.npy", y_train)
np.save("../data/train/X_test.npy", X_test)
np.save("../data/train/y_test.npy", y_test)
