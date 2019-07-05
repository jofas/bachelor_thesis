import numpy as np
import random

def random_int_matrix(elem, upper, lower = 0):
    X = np.array([random.randint(lower, upper)
        for _ in range(elem ** 2)])
    return np.reshape(X, (elem, elem))
