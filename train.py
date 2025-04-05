import numpy as np
from model import forward, backward, update
from utils import cross_entropy_loss, compute_accuracy
import pickle
import os

def train_model(X_train, Y_train, X_val, Y_val, config):
    params = config['params']
    best_params, best_acc = None, 0
    train_loss, val_loss, val_acc = [], [], []

    for epoch in range(config['epochs']):
        indices = np.random.permutation(X_train.shape[0])
        X_train, Y_train = X_train[indices], Y_train[indices]

        for i in range(0, X_train.shape[0], config['batch_size']):
            X_batch = X_train[i:i + config['batch_size']]
            Y_batch = Y_train[i:i + config['batch_size']]
            Z1, A1, Z2, A2 = forward(X_batch, params)
            grads = backward(X_batch, Y_batch, params, Z1, A1, A2, config['reg'])
            params = update(params, grads, config['lr'])

        # Evaluate
        _, _, _, A_train = forward(X_train, params)
        _, _, _, A_val = forward(X_val, params)
        loss_train = cross_entropy_loss(Y_train, A_train)
        loss_val = cross_entropy_loss(Y_val, A_val)
        acc_val = compute_accuracy(Y_val, A_val)

        train_loss.append(loss_train)
        val_loss.append(loss_val)
        val_acc.append(acc_val)

        if acc_val > best_acc:
            best_acc = acc_val
            best_params = {k: v.copy() for k, v in params.items()}
            with open("best_model.pkl", "wb") as f:
                pickle.dump(best_params, f)

        print(f"Epoch {epoch+1}: Train Loss {loss_train:.4f}, Val Acc {acc_val:.4f}")

    return train_loss, val_loss, val_acc