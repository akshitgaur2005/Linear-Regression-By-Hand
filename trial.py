from getdata import get_data
from processor import get_processed
import numpy as np

def main():
    train_data = get_data("train.csv")
    X, y = get_processed(train_data)
    print(f"X: {X.shape}\n y: {y.shape}")

    W = np.zeros((X.shape[1], 1))
    print(W.shape)
    b = 0

    y_pred = predict(X, W, b)
    print(y_pred)
    loss = loss_fn(y_pred, y)
    print(loss)
    print(dJ_dW(X, y_pred, y))
    
    loss_track = []

    for i in range(500):
        y_pred = predict(X, W, b)
        loss = loss_fn(y_pred, y)
        loss_track.append(loss)
        W, b = gradient_descent(X, W, b, 0.005, y_pred, y)
        if (i % 10 == 1):
            print(f"Epoch: {i}, Loss: {loss}")


def predict(X, W, b):
    return (np.dot(X, W) + b)

def loss_fn(y_pred, y):
    #J = np.sum(np.square(y_pred) + np.square(y) - 2 * np.dot(y_pred.T, y))
    J = np.sum(np.square(y_pred - y))
    return J / (2 * y_pred.shape[0])

def dJ_dW(X, y_pred, y):
    m = y_pred.shape[0]
    dJ = np.sum(np.dot((y_pred - y).T, X)) # 1x317 1460x317
    return dJ / m

def dJ_db(y_pred, y):
    return np.sum(y_pred - y) / y_pred.shape[0]

def gradient_descent(X, W, b, lr, y_pred, y):
    W_copy = W
    b_copy = b

    W_copy = W - lr * dJ_dW(X, y_pred, y)
    b_copy = b - lr * dJ_db(y_pred, y)

    return W_copy, b_copy


main()
