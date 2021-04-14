import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

train_data = pd.read_csv("data/kc_house_data.csv", nrows=1000)
X = train_data['sqft_living']
y = train_data['price']

plt.scatter(X, y)

linear_regression = linear_model.LinearRegression()
linear_regression.fit(X.values.reshape(-1, 1), y)

y_linear = [linear_regression.coef_ * i +
            linear_regression.intercept_ for i in X]
plt.plot(X, y_linear, c='red')


plt.show()
