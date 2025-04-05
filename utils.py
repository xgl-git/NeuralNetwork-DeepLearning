import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder


def load_batch(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    X = data[b'data'].astype(np.float32) / 255.0
    Y = np.array(data[b'labels']).reshape(-1, 1)
    return X, Y


def load_data(path='./cifar-10-batches-py'):
    X_train, Y_train = [], []
    for i in range(1, 6):
        X, Y = load_batch(f"{path}/data_batch_{i}")
        X_train.append(X)
        Y_train.append(Y)
    X_train = np.vstack(X_train)
    Y_train = np.vstack(Y_train)

    X_val, Y_val = X_train[:5000], Y_train[:5000]
    X_train, Y_train = X_train[5000:], Y_train[5000:]

    X_test, Y_test = load_batch(f"{path}/test_batch")

    enc = OneHotEncoder(sparse_output=False)
    Y_train = enc.fit_transform(Y_train)
    Y_val = enc.transform(Y_val)
    Y_test = enc.transform(Y_test)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def cross_entropy_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))


def compute_accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))