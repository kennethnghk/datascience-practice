from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

np.random.seed(2)

pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = np.random.normal(50.0, 30.0, 1000) / pageSpeeds

# K-fold, get 80% as train data set, 20% for test data set
trainX = pageSpeeds[:80]
testX = pageSpeeds[80:]

trainY = purchaseAmount[:80]
testY = purchaseAmount[80:]

x = np.array(testX)
y = np.array(testY)

pOrder = 4
p4 = np.poly1d(np.polyfit(x, y, pOrder))

# Compare r-square values to check if it is overfitting
testR2 = r2_score(testY, p4(testX))
print "R2 score for test data-set: ", testR2

trainR2 = r2_score(trainY, p4(trainX))
print "R2 score for train data-set: ", trainR2

# show overfitting polynomial regression in graph
xp = np.linspace(0, 7, 1000)
axes = plt.axes()
axes.set_xlim([0,7])
axes.set_ylim([0, 200])
plt.scatter(x, y)
plt.plot(xp, p4(xp), c='r')
plt.show()
