import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from math import pow as pw
from math import sqrt
N_ROWS = 100

x_data = [34, 108, 64, 88, 99, 51]
y_data = [5, 17, 11, 8, 14, 5]

class SimpleLinearRegression:
    def __init__(self, iv, dv):
        self.N_ROWS = len(dv)
        self.iv = iv
        self.dv = dv
        self.iv_mean = sum(i for i in iv) // self.N_ROWS
        self.dv_mean = sum(i for i in dv) // self.N_ROWS
        self.coef = 0
        self.intercept = 0

    def calculate_slope(self):
        p1 = 0
        p2 = 0
        for i in range(len(self.iv)):
            xi = self.iv[i]
            yi = self.dv[i]
            p1 += (xi - self.iv_mean) * (yi - self.dv_mean)
            p2 += (xi - self.iv_mean) ** 2
        self.coef = p1 / p2
        return self.coef

    def calculate_intercept(self):
        self.intercept = self.dv_mean - self.calculate_slope() * self.iv_mean
        return self.intercept


def get_sse(d1, d2):
    sse = 0
    for i in range(len(d1)):
        sse += pow(d1[i] - d2, 2)
    return sse


train_data = pd.read_csv("../data/kc_house_data.csv", nrows=30)
ind_variable = train_data['sqft_living']
dep_variable = train_data['price']

linear_regression = SimpleLinearRegression(ind_variable, dep_variable)
linear_regression.calculate_intercept()
intercept = linear_regression.intercept
coef = linear_regression.coef



plt.scatter(ind_variable, dep_variable)
plt.plot(ind_variable, [i * coef + intercept for i in ind_variable])
plt.ticklabel_format(useOffset=False, style='plain')
print(coef, intercept)
plt.show()
