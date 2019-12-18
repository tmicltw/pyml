import numpy as np


def softmax(vector):
    exp_vector = np.exp(vector)
    return exp_vector / np.sum(exp_vector)


vector = np.array([1, 40, 30, 20, 10, 5])
result = softmax(vector)
print(result)
