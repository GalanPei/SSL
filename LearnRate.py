import PlotStyle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

n = 10


def gd(x0, y0, alpha=0.1):
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0] = x0
    y[0] = y0
    for i in range(n):
        x[i + 1] = x[i] - alpha * 2 * x[i]
        y[i + 1] = y[i] - alpha * 4 * y[i]
    return x, y


x = np.linspace(-2, 2)
y = np.linspace(-2, 2)
X, Y = np.meshgrid(x, y)
Z = X ** 2 + 2 * Y ** 2

plt.contourf(X, Y, Z, 20, cmap=cm.coolwarm)
plt.contour(X, Y, Z, 20, cmap=cm.coolwarm)
x, y = gd(-2, -2, 0.48)
u = np.array([x[i + 1] - x[i] for i in range(len(x) - 1)])
v = np.array([y[i + 1] - y[i] for i in range(len(x) - 1)])
x = x[:len(u)]  # 使得维数和u,v一致
y = y[:len(v)]
# plt.quiver(x, y, u, v, angles='xy')
plt.plot(x, y, '*-', color='k', linewidth='1')
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$y$', fontsize=14)
plt.show()
