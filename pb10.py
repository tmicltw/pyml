import numpy as np


def relu(tensor):
    return np.maximum(tensor, 0)

print(relu([3, -2.5]))
