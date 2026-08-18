# Implement a Perceptron Network for binary classification using logic gates. 
import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        weighted_sum = np.dot(x, self.weights) + self.bias
        return self.activation(weighted_sum)

    def train(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                prediction = self.predict(xi)

                error = target - prediction

                self.weights += self.learning_rate * error * xi
                self.bias += self.learning_rate * error


# -----------------------------
# AND Gate
# -----------------------------

X_AND = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_AND = np.array([0, 0, 0, 1])

and_model = Perceptron()
and_model.train(X_AND, y_AND)

print("AND Gate")
for x in X_AND:
    print(x, "->", and_model.predict(x))


# -----------------------------
# OR Gate
# -----------------------------

y_OR = np.array([0, 1, 1, 1])

or_model = Perceptron()
or_model.train(X_AND, y_OR)

print("\nOR Gate")
for x in X_AND:
    print(x, "->", or_model.predict(x))


# -----------------------------
# NOT Gate
# -----------------------------

X_NOT = np.array([
    [0],
    [1]
])

y_NOT = np.array([1, 0])

not_model = Perceptron()
not_model.train(X_NOT, y_NOT)

print("\nNOT Gate")
for x in X_NOT:
    print(x, "->", not_model.predict(x))