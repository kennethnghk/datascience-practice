## Support Vector Machines - SVC kernel

import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# Create fake income/age clusters for N people in k clusters
def createClusteredData(N, k):
    pointsPerCluster = float(N)/k
    X = []
    y = []
    for i in range (k):
        incomeCentroid = np.random.uniform(20000.0, 200000.0)
        ageCentroid = np.random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            X.append([np.random.normal(incomeCentroid, 10000.0), np.random.normal(ageCentroid, 2.0)])
            y.append(i)
    X = np.array(X)
    y = np.array(y)
    return X, y

nCluster = 5
(X, y) = createClusteredData(100, nCluster)

## Plot cluster graph
# plt.figure(figsize=(8, 6))
# plt.scatter(X[:,0], X[:,1], c=y.astype(np.float))
# plt.show()

## Use linear SVC to partition our graph into clusters:
C = 1.0
kernel = 'poly' ## linear, rbf, poly etc
svc = svm.SVC(kernel=kernel, C=C).fit(X, y)

## Try to prediction which partition the data is belongs to
print svc.predict([[200000, 40]])

## Plot cluster graph with SVC partition
def plotPredictions(clf):
    xx, yy = np.meshgrid(np.arange(0, 250000, 10),
                     np.arange(10, 70, 0.5))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    plt.figure(figsize=(8, 6))
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:,0], X[:,1], c=y.astype(np.float))
    plt.show()
    
plotPredictions(svc)