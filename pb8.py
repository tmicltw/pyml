import numpy as np


W = np.array([[5, 3], [2, -1], [4, 2]])
x = np.array([3, 2])

# (5   3)             (5 x 3 +  3 x 2)    (21)
# (2  -1)  x  (3 2) = (2 x 3 + -1 x 2) =  ( 4)
# (4   2)             (4 x 3 +  2 x 2)    (16)

print(np.matmul(W, x))
