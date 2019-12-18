import csv

import numpy as np


def relu(tensor):
    return np.maximum(tensor, 0)


def softmax(vector):
    exp_vector = np.exp(vector)
    return exp_vector / np.sum(exp_vector, axis=1, keepdims=True)


def forward(X, W1, W2):
    a1 = np.matmul(X, W1)
    h1 = relu(a1)
    a2 = np.matmul(h1, W2)
    return softmax(a2)


def load_iris(filename):
    X = []
    y = []
    answers = {}
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            X.append([float(v) for v in row[:-1]])
            answer = row[-1]
            if answer not in answers:
                answers[answer] = len(answers)
            y.append(answers[answer])
    return np.array(X), np.array(y)


X, y = load_iris("Iris.csv")


# (5 x 4)
W1 = np.array([[5, 3, 2, 1], [2, 7, 2, 3], [2, 0, -1, -3], [2, -2, 3, 4], [3, 1, 1, -3]]) / 10
# (4 x 3)
W2 = np.array([[5, 3, 1], [2, -1, 8], [4, 2, 1], [1, 3, 0]]) / 10
# given the parameters, the forward path will execute
# with input of 2 dimensions, 4 neurons in the hidden layer
# and output a result with 3 classes
y_probs = forward(X, W1, W2)
y_preds = np.argmax(y_probs, axis=1)

correct_count = len(y[y == y_preds])
print("accuracy: {0:.3f}".format(correct_count / len(y)))
