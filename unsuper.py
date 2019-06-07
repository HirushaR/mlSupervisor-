import numpy as np
import matplotlib.pyplot as plt

from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans


#convert data into numpy array
X = np.array([[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]])

# we initialize K-means algorithm with the required parameter and we user .fit() to get the data

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

#getting the values of centroids and lebale based on the fitment

centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print(centroids)
print(labels)

#plotting and visualizing out put
colors = ["r.", "g.", "c.", "y."]
for i in range(len(X)):
    print("coordinateL",X[i],"lable:",labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize= 10)

plt.scatter(centroids[:,0],centroids[:,1],marker="X", s=150, linewidths =5, zorder =10)
plt.show()
