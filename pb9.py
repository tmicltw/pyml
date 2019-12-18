import numpy as np


def softmax(vector):
    exp_vector = np.exp(vector)
    return exp_vector / np.sum(exp_vector)



x = np.array([3, 2])
W = np.array([[5,3], [2,-1], [4,2]])
activation = np.matmul(W, x)
output = softmax(activation)
print(np.sum(output))
