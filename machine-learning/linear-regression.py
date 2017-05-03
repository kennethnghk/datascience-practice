import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ages = np.random.normal(45, 2.0, 1000)
incomes = (ages + np.random.normal(0, 3.0, 1000)) * 1000

slope, intercept, r_value, p_value, std_err = stats.linregress(ages, incomes)

r_square = r_value ** 2
print 'r_square: ', r_square

def predict(x):
    return slope * x + intercept

fitLine = predict(ages)

plt.scatter(ages, incomes)
plt.plot(ages, fitLine, c='r')
plt.show()

