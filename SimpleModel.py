import numpy as np
import math
import matplotlib.pyplot as plt

'''
@:param
n - number of samples
d - dimension
N - number of epsilon
'''

n = 1000
d = 1
N = 20


def fun_eta(t, eps):
    if t < eps:
        return (1 / eps) ** d
    else:
        return 0


def fun_eta_gauss(t, eps):
    # return math.exp(-math.fabs(t) / 2 / (eps ** 2))
    return 1 / math.sqrt(2 * math.pi) / (eps ** d) * math.exp(-math.fabs(t) / 2 / (eps ** 2))


unlabeledData = np.random.rand(n, d)
x = np.concatenate((np.array([[0], [1]]), unlabeledData), axis=0)
fl = np.array([[0], [1]])
fu1 = np.zeros((n, 1))
fu2 = np.zeros((n, 1))
D1 = np.zeros((n + 2, n + 2))
D2 = np.zeros((n + 2, n + 2))
W1 = np.zeros((n + 2, n + 2))
W2 = np.zeros((n + 2, n + 2))
Duu1 = np.zeros((n, n))
Duu2 = np.zeros((n, n))
Wuu1 = np.zeros((n, n))
Wuu2 = np.zeros((n, n))
Wul1 = np.zeros((n, 2))
Wul2 = np.zeros((n, 2))
error1 = np.zeros((N, 1))
error2 = np.zeros((N, 1))
count = 0
eps_data = np.linspace(1e-3, 0.5, num=N)
eps_data = eps_data.reshape((N, 1))

for Eps in eps_data:
    for i in range(n + 2):
        for j in range(n + 2):
            W1[i, j] = fun_eta(math.fabs(x[i] - x[j]), Eps)
            W2[i, j] = fun_eta_gauss(math.fabs(x[i] - x[j]), Eps)
        D1[i, i] = np.sum(W1[i, :])
        D2[i, i] = np.sum(W2[i, :])
    Duu1 = D1[2:n + 2, 2:n + 2]
    Duu2 = D2[2:n + 2, 2:n + 2]
    Wul1 = W1[2:n + 2, 0:2]
    Wul2 = W2[2:n + 2, 0:2]
    fu1 = np.dot(np.linalg.inv(Duu1 - Wuu1), Wul1)
    fu1 = np.dot(fu1, fl)
    f1 = np.concatenate((fl, fu1))
    fu2 = np.dot(np.linalg.inv(Duu2 - Wuu2), Wul2)
    fu2 = np.dot(fu2, fl)
    f2 = np.concatenate((fl, fu2))
    error1[count, 0] = np.linalg.norm(f1 - x)
    error2[count, 0] = np.linalg.norm(f2 - x)
    count += 1

plt.figure(1)
plt.plot(eps_data, error1)

plt.figure(2)
plt.plot(eps_data, error2)
plt.show()
