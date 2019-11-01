import numpy as np
from matplotlib import pyplot as plt

#x = np.arange(1, 11)
x = [i for i in range(1, 11)]
y = [2.5*i for i in x]
y[0]=10
y[5]=35
plt.title("Matplotlib demo")
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
plt.plot(x, y)
plt.show()