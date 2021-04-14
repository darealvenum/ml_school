import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree

df = pd.read_csv("data/logistic_regression.csv") #lees de csv en maak een data frame
X = df[['x', 'y']] # maak een 2d array voor x en y
z = df['z']

model_tree = tree.DecisionTreeClassifier() # dt object
model_tree.fit(X, z)


print(model_tree.predict([[-23, -23]]))