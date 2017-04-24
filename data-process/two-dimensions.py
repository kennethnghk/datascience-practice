from lib import random_normal
import matplotlib.pyplot as plt

xs = [random_normal() for _ in range(1000)]
ys1 = [ x + random_normal() / 2 for x in xs]
ys2 = [ -x + random_normal() / 2 for x in xs]

plt.scatter(xs, ys1, marker=".", color="black", label="ys1")
plt.scatter(xs, ys2, marker=".", color="gray", label="ys2")
plt.xlabel('xs')
plt.ylabel('ys')
plt.legend(loc=9)
plt.title("Very different Joint Distribution")
plt.show()