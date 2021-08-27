import GenerateGraph
import LPA
import matplotlib.pyplot as plt
import PlotStyle
import TSVM
import time
import numpy as np

N = 500  # number of each sample
epsilon = 0.8
num_MC = 50

# Relationship between ratio of labeled data and learning accuracy
num_l = np.arange(3, 50, 5)  # List to store the number of labeled data
num_test = num_l.shape[0]
num_l = num_l.reshape((num_test, 1))
acc_LPA = np.zeros((num_test, 1))  # List to store the accuracy of the algorithm
temp_acc = np.zeros((num_test, 1))
for i in range(num_MC):
    for j in range(num_test):
        labeled_multi, unlabeled_multi, true_multi = GenerateGraph.SyntheticGraph(N, num_l[j, 0])
        model_1 = LPA.LPA(labeled_multi, unlabeled_multi)
        multi_label, full_multi = model_1.labelPropImp(epsilon, weight_fun='Epsilon', alpha=0.5, iterMax=1e4, tol=1e-5)
        temp_acc[j, 0] = model_1.accuracy(multi_label, true_multi)
    acc_LPA += temp_acc
acc_LPA /= num_MC
plt.figure(1)
plt.plot(1/1500*num_l, acc_LPA)
plt.xlabel(r'Ratio of labeled data', fontsize=14)
plt.ylabel(r'Accuracy of LPA', fontsize=14)
plt.show()

# plt.figure(2)
# labeled_binary, unlabeled_binary, true_binary = GenerateGraph.SyntheticGraph(N, n, GraphType='Binary')
# model_2 = LPA.LPA(labeled_binary, unlabeled_binary)
# binary_label = model_2.labelPropImp(epsilon, weight_fun='Epsilon', alpha=0.5, iterMax=1e4, tol=1e-5)
# model_2.showResult(binary_label)
# plt.show()
#
# plt.figure(3)
# model = TSVM.TSVM()
# model.initial()
# model.train(labeled_binary[:, 0:-1], 2 * labeled_binary[:, -1] - 1, unlabeled_binary)
# Y_hat = model.predict(unlabeled_binary)
# Y_hat = Y_hat.reshape((Y_hat.shape[0], 1))
# model.showResult(labeled_binary[:, 0:-1], unlabeled_binary, labeled_binary[:, -1], Y_hat)
# plt.show()
