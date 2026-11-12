import numpy as np

def lift(x):
    x_lifted = list(x)

    d = len(x)

    for i in range(d):
        for j in range(i + 1):
            x_lifted.append(x[i] * x[j])

    return np.array(x_lifted)

def liftDataset(X):
    X_lifted = []

    for row in X:
        X_lifted.append(lift(row))

    return np.array(X_lifted)