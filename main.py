from model import initialize_parameters
from train import train_model
from test import test_model
from utils import load_data
from visualize import plot_metrics, visualize_first_layer, visualize_second_layer, visualize_output_distribution
import numpy as np
import pickle
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
def forward_propagation(X, params):
    Z1 = np.dot(X, params['W1']) + params['b1']
    A1 = relu(Z1)
    Z2 = np.dot(A1, params['W2']) + params['b2']
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2
if __name__ == "__main__":
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data()

    params = initialize_parameters(3072, 128, 10)
    config = {
        'params': params,
        'lr': 0.01,
        'reg': 0.001,
        'epochs': 150,
        'batch_size': 64
    }

    train_loss, val_loss, val_acc = train_model(X_train, Y_train, X_val, Y_val, config)
    test_model(X_test, Y_test)
    plot_metrics(train_loss, val_loss, val_acc)
    with open("best_model.pkl", "rb") as f:
        best_params = pickle.load(f)
    visualize_first_layer(best_params['W1'])
    visualize_second_layer(best_params['W2'])
    visualize_output_distribution(X_val[:100], best_params, forward_propagation)
