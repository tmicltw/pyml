from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

datalist = []
f = open('abalone.data', 'r')
for line in f:
    line = line.rstrip()
    elems = line.split(',')
    datalist.append(elems[1:])
f.close()
data = np.array(datalist)
x = np.c_[np.float_(data[:,0])]
y = np.c_[np.float_(data[:,4])]

lr = linear_model.LinearRegression()

X = np.hstack((x, np.power(x,2)))

lr.fit(X, y)

y_pred = lr.predict(X)
mse = ((y - y_pred) ** 2).mean()
print("MSE = {0}".format(mse))
