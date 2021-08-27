import GenerateGraph
import numpy as np
import LPA
import matplotlib.pyplot as plt
import PlotStyle

N = 500  # number of each class
epsilon = np.hstack((np.linspace(0.1, 1, 15), np.linspace(2, 4, 5)))  # epsilon ranges from 10^-4 to 1
num_MC = 50
num_l = 20
acc_LPA = np.zeros((epsilon.shape[0]))

for i in range(num_MC):
    labeled_binary, unlabeled_binary, true_binary = GenerateGraph.SyntheticGraph(N, num_l, GraphType='Binary')
    acc_temp = np.zeros((epsilon.shape[0]))
    for j in range(epsilon.shape[0]):
        model_2 = LPA.LPA(labeled_binary, unlabeled_binary, type='Binary')
        binary_label, full_binary = model_2.labelPropImp(epsilon[j], weight_fun='Epsilon', alpha=0.5, iterMax=1e4, tol=1e-5)
        acc_temp[j] = model_2.accuracy(binary_label, true_binary)
    acc_LPA += acc_temp
acc_LPA /= num_MC

plt.figure(1)
plt.axes(xscale="log")
plt.plot(epsilon, acc_LPA, linewidth='1', marker='o', markersize=2)
plt.xlabel(r'$\epsilon$', fontsize=14)
plt.ylabel(r'Accuracy of LPA')
plt.show()
