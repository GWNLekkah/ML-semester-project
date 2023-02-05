import numpy as np
from collections import Counter


def class_limit(X, Y):
    """
    Balances the amount of samples per class by limiting the majority classes.
    """
    # Shuffle the data
    order = np.random.permutation(len(X))
    X = X[order]
    Y = Y[order]

    # Determine minority class
    keys = Counter(Y).keys()
    limit = min(Counter(Y).values())
    X_limit = []
    Y_limit = []
    for key in keys:
        # Limit each class
        indices = [idx for idx, value in enumerate(Y) if value == key]
        X_limit.extend(X[indices][0:limit])
        Y_limit.extend(Y[indices][0:limit])
    X = np.array(X_limit)
    Y = np.array(Y_limit)

    # Reshuffle
    order = np.random.permutation(len(X))
    X = X[order]
    Y = Y[order]
    return X, Y
