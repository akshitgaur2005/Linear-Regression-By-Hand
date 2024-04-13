import numpy as np

class Model():

    def __init__(self, X, y, lr, epochs):
        self.X = X
        self.y = y
        self.m = self.X.shape[1]
        self.W = np.random.random((self.m, 1))
        self.b = np.random.random((1,))
        self.lr = lr
        self.epochs = epochs

    def predict(self, X):
        return np.dot(X, self.W) + self.b

    def loss_fn(self):
        return np.mean(np.square(self.predict(self.X) - self.y)) / 2

    def gradient(self):
        dW = np.mean(np.dot((self.predict(self.X) - self.y).T, self.X))
        db = np.mean(self.predict(self.X) - self.y)
        return dW, db

    def fit(self):
        for i in range(self.epochs):
            dW, db = self.gradient()
            self.W -= self.lr * dW
            self.b -= self.lr * db

            if (i % 100 == 0):
                print(f"Epoch: {i + 1}, Loss: {self.loss_fn()}")
