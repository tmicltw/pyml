import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

d = pd.read_csv('GlobalTemperatures.csv')
offset = 100
date = d['dt'][offset::12]
year = np.array([int(e.split('-')[0]) for e in date])
temperature = d['LandAverageTemperature'][offset::12]
lr = linear_model.LinearRegression()
lr.fit(year.reshape((-1,1)), temperature.values)
temperature_predict = lr.predict(year.reshape((-1,1)))

mse = ((temperature - temperature_predict) ** 2).mean()
print("MSE = {0}".format(mse))
