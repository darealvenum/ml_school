import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import tree


# lees de csv en maak een data frame
data = pd.read_csv("data/logistic_regression.csv")
X = data[["x", "y"]]  # maak een 2d array voor x en y
y = data["z"]


fig = plt.figure()
ax = fig.add_subplot(projection="3d")  # maak een 3d graph
ax.scatter(data["x"], data["y"], data["z"])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# ================= train mijn dataset met logistic regression

model = LogisticRegression()  # logistic regression object
model.fit(X, y)  # train op data

index = 0
for i in range(len(X)):
    x = X["x"]
    y_ = X["y"]
    print("was: ", y[index], "voorspelling:", model.predict([[x[i], y_[i]]]))
    index += 1


# ================= train mijn dataset met decision trees

model_tree = tree.DecisionTreeClassifier()  # dt object
model_tree.fit(X, y)

# =============== vergelijk accuracies
test_data = pd.read_csv("data/test_data.csv")
X_test = test_data[["x", "y"]]
y_test = test_data["z"]

print("logistic regression accuracy: ", model.score(X_test, y_test))
print("decision tree accuracy: ", model_tree.score(X_test, y_test))

plt.show()
