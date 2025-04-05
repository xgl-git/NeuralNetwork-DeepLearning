import numpy as np

def relu(x): return np.maximum(0, x)
def relu_derivative(x): return (x > 0).astype(float)

def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def initialize_parameters(input_dim, hidden_dim, output_dim):
    np.random.seed(42)
    return {
        'W1': np.random.randn(input_dim, hidden_dim) * 0.01,
        'b1': np.zeros((1, hidden_dim)),
        'W2': np.random.randn(hidden_dim, output_dim) * 0.01,
        'b2': np.zeros((1, output_dim)),
    }

def forward(X, params):
    Z1 = X @ params['W1'] + params['b1']
    A1 = relu(Z1)
    Z2 = A1 @ params['W2'] + params['b2']
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def backward(X, Y, params, Z1, A1, A2, reg_lambda):
    m = X.shape[0]
    dZ2 = A2 - Y
    dW2 = A1.T @ dZ2 / m + reg_lambda * params['W2']
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dZ1 = (dZ2 @ params['W2'].T) * relu_derivative(Z1)
    dW1 = X.T @ dZ1 / m + reg_lambda * params['W1']
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

def update(params, grads, lr):
    for k in params:
        params[k] -= lr * grads['d' + k]
    return params
