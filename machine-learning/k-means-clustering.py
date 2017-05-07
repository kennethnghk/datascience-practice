from numpy import random, array, float
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

# Create cluster data with N points and K clusters
def createClusteredData(N, k):
    random.seed(10)
    pointsPerCluster = float(N)/k
    X = []
    for i in range (k):
        incomeCentroid = random.uniform(20000.0, 200000.0)
        ageCentroid = random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            X.append([random.normal(incomeCentroid, 10000.0), random.normal(ageCentroid, 2.0)])
    X = array(X)
    return X

nCluster = 5
nDataPoints = 1000

data = createClusteredData(nDataPoints, nCluster)

model = KMeans(n_clusters = nCluster)

# Scale : normalize data so that they are in same scale
model = model.fit(scale(data))

# Print cluster data
print model.labels_ 

# Visualize data
plt.figure(figsize=(8, 6))
plt.scatter(data[:,0], data[:,1], c=model.labels_.astype(float))
plt.show()