import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/logistic_regression.csv") #lees de csv en maak een data frame
X = df[['x', 'y']] # maak een 2d array voor x en y
z = df['z']


fig = plt.figure()
ax = fig.add_subplot(projection='3d') # maak een 3d graph
ax.scatter(df['x'], df['y'], df['z'])
# zet labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# ================= train mijn dataset

# print(.shape)
model = LogisticRegression() # logistic regression object
model.fit(X, z) # train op data 

if input("wil je de grafiek zien? ") == "ja":
    plt.show()