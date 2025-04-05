import pickle
from model import forward
from utils import compute_accuracy

def test_model(X_test, Y_test):
    with open("best_model.pkl", "rb") as f:
        params = pickle.load(f)
    _, _, _, A_test = forward(X_test, params)
    acc = compute_accuracy(Y_test, A_test)
    print(f"Test Accuracy: {acc:.4f}")
    return acc
