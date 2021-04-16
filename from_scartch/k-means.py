import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import pandas as pd
import random

# data
df = pd.read_csv("../data/kmeans.csv")
x_train = df[["x", "y"]]
x_train = np.array(x_train)


class KMeans:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = data[i]  # eerste random punten

        for i in range(self.max_iter):
            self.classefications = {}  # alle datapunten worder hier gelinkt aan hun centroids

            for i in range(self.k):
                # maak een list voor elke centroid
                self.classefications[i] = []

            for featureset in data:
                # krijg elke afstand van punt naar centroid
                distances = [np.linalg.norm(
                    featureset - self.centroids[centroid]) for centroid in self.centroids]
                classefication = distances.index(
                    min(distances))  # index van centroid
                self.classefications[classefication].append(
                    featureset)  # voeg dataset toe aan centroid

            prev_centroids = dict(self.centroids)

            for classefication in self.classefications:
                # reset the positie van de centroid naar het gemiddelde van alle punten
                self.centroids[classefication] = np.average(
                    self.classefications[classefication], axis=0)

        return self.centroids

kmeans = KMeans(k=3)
centroids = kmeans.fit(x_train)
plt.scatter(df["x"], df["y"])
for key, value in centroids.items():
    plt.scatter(value[0], value[1], c="red", s=330)
plt.show()
