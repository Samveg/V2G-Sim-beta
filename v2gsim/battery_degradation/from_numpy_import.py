import math
import matplotlib.pyplot as plt

t = range(200)
a = [2*t[i] for i in range(len(t))]
b = [3*t[i] for i in range(len(t))]

plt.plot(t,a,'r')
plt.plot(t,b,'b')

plt.show()