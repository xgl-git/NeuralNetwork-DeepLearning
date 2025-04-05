from model import initialize_parameters
from train import train_model
from utils import load_data
import numpy as np

def hyperparameter_search():
    X_train, Y_train, X_val, Y_val, _, _ = load_data()

    for hidden in [64, 128]:
        for lr in [0.01, 0.005]:
            for reg in [0.001, 0.0001]:
                print(f"\nTesting: hidden={hidden}, lr={lr}, reg={reg}")
                params = initialize_parameters(3072, hidden, 10)
                config = {
                    'params': params,
                    'lr': lr,
                    'reg': reg,
                    'epochs': 20,
                    'batch_size': 64
                }
                train_model(X_train, Y_train, X_val, Y_val, config)
