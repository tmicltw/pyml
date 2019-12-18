import numpy as np


def relu(tensor):
    return np.maximum(tensor, 0)


def softmax(vector):
    exp_vector = np.exp(vector)
    return exp_vector / np.sum(exp_vector)


def forward(x, W1, W2):
    a1 = np.matmul(x, W1)
    h1 = relu(a1)
    a2 = np.matmul(h1, W2)
    return softmax(a2)


# (2)
x = np.array([3, 2])
# (2 x 4)
W1 = np.array([[5, 3, 2, -1], [4, 2, 3, 2]])
# (4 x 3)
W2 = np.array([[5, 3, 1], [2, -1, 8], [4, 2, 1], [1, 3, 0]])
# given the parameters, the forward path will execute
# with input of 2 dimensions, 4 neurons in the hidden layer
# and output a result with 3 classes
forward(x, W1, W2)
