import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

np.random.seed(2)
pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = np.random.normal(50.0, 10.0, 1000) / pageSpeeds

x = np.array(pageSpeeds)
y = np.array(purchaseAmount)

# create polynomial model with pOrder
pOrder = 6
pModel = np.poly1d(np.polyfit(x, y, pOrder))

r2 = r2_score(y, pModel(x))
print 'r-square:', r2

xp = np.linspace(0, 7, 100)
plt.scatter(x, y)
plt.plot(xp, pModel(xp), c='r')
plt.show()