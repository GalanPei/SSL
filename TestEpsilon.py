import GenerateGraph
import numpy as np
import LPA
import matplotlib.pyplot as plt
import PlotStyle

N = 500  # number of each class
epsilon = np.hstack((np.logspace(-4, 0, 10), np.linspace(2, 4, 5)))  # epsilon ranges from 10^-4 to 5
# epsilon = np.linspace(1e-3, 1, 4)
num_MC = 50
num_l = 20
acc_LPA = np.zeros((epsilon.shape[0]))
con_LPA = np.zeros((epsilon.shape[0]))

labeled_binary, unlabeled_binary, true_binary = GenerateGraph.SyntheticGraph(N, num_l, GraphType='Binary')
for i in range(num_MC):
    acc_temp = np.zeros((epsilon.shape[0]))
    con_temp = np.zeros((epsilon.shape[0]))
    for j in range(epsilon.shape[0]):
        model_2 = LPA.LPA(labeled_binary, unlabeled_binary, type='Binary')
        binary_label, full_binary = model_2.labelPropImp(epsilon[j], weight_fun='Epsilon', alpha=0.5)
        acc_temp[j] = model_2.accuracy(binary_label, true_binary)
        con_temp[j] = model_2.Connectivity(epsilon[j], weight_fun='Epsilon')
    acc_LPA += acc_temp
    con_LPA += con_temp
acc_LPA /= num_MC
con_LPA /= num_MC

fig = plt.figure(1)
ax1 = fig.add_subplot(111)
ax1.set_xscale("log")
plt.xlabel(r'$\epsilon$')
# plt.plot(epsilon, acc_LPA, linewidth='1', marker='o', markersize=2)
l1 = ax1.plot(epsilon, acc_LPA, linewidth='1', marker='o', markersize=2, color='teal', label='accuracy')
ax1.set_xlabel(r'$\varepsilon$')
ax1.set_ylabel(r'Accuracy of LPA')
ax1.spines['top'].set_visible(False)
# ax1.legend()

ax2 = ax1.twinx()
ax2.set_xscale("log")
l2 = ax2.plot(epsilon, con_LPA, linewidth='1', marker='o', markersize=2, color='sienna', label='connectivity')
ax2.set_ylabel(r'Graph Connectivity')
ax2.spines['top'].set_visible(False)
# ax2.legend()
l3 = l1 + l2
labs = [l.get_label() for l in l3]
ax1.legend(l3, labs)
plt.show()
