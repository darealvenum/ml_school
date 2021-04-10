import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("data/kmeans.csv") 
num_clusters = 3 # ik denk dat er 3 clusters komen

# ik gok 3 clusters
# plt.scatter(df["x"], df["y"])
# plt.show()

km = KMeans(n_clusters=num_clusters)
y_predicted = km.fit_predict(df[["x", "y"]]) # zal een model trainen waarbij elk punt in df een centroid krijgt
df["cluster"] = y_predicted

# vormt een dataframe waar de cluster "index" gelijk is aan i
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

# krijgt de coordinaten van de centroids
centroids = km.cluster_centers_
print(centroids)

plt.scatter(df1.x, df1.y)
plt.scatter(df2.x, df2.y)
plt.scatter(df3.x, df3.y)

for i in range(num_clusters):
    plt.scatter(centroids[i][0], centroids[i][1], s=353)

plt.show()
print(km.predict([[38, 91]]))