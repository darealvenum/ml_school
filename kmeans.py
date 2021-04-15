import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("data/kmeans.csv")
# ik denk dat er 3 clusters komen
num_clusters = int(input("hoeveel clusters? "))

# ik gok 3 clusters
# plt.scatter(df["x"], df["y"])
# plt.show()

km = KMeans(n_clusters=num_clusters)
# zal een model trainen waarbij elk punt in df een centroid krijgt
y_predicted = km.fit_predict(df[["x", "y"]])
df["cluster"] = y_predicted


# krijgt de coordinaten van de centroids
centroids = km.cluster_centers_
print(centroids)


for i in range(num_clusters):
    df1 = df[df.cluster == i]
    plt.scatter(df1.x, df1.y)
    plt.scatter(centroids[i][0], centroids[i][1], s=353)

plt.show()
print(km.predict([[38, 91]]))
