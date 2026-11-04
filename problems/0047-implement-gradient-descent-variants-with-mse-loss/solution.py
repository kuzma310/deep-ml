import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method="batch"):

    m = len(y)

    for epoch in range(n_epochs):

        if method == "batch":

            predictions = X @ weights
            errors = predictions - y
            gradient = (2 / m) * (X.T @ errors)
            weights = weights - learning_rate * gradient

        elif method == "stochastic":

            for i in range(m):
                prediction = X[i] @ weights
                error = prediction - y[i]
                gradient = 2 * error * X[i]
                weights = weights - learning_rate * gradient

        elif method == "mini_batch":

            for i in range(0, m, batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                predictions = X_batch @ weights
                errors = predictions - y_batch
                gradient = (2 / len(y_batch)) * (X_batch.T @ errors)

                weights = weights - learning_rate * gradient

    return weights