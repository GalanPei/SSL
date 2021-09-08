import GenerateGraph
import LPA
import matplotlib.pyplot as plt
import PlotStyle
import TSVM
import numpy as np

N = 500
n = 3
epsilon = 0.4

labeled_multi, unlabeled_multi, true_multi = GenerateGraph.SyntheticGraph(N, n)
model_1 = LPA.LPA(labeled_multi, unlabeled_multi)
iter_num = np.array([1, 5, 10, 20])
for i in range(iter_num.shape[0]):
    multi_label, full_multi = model_1.labelPropImp(epsilon, weight_fun='Epsilon', alpha=0.5, iterMax=iter_num[i], tol=0)
    plt.figure()
    model_1.showResult(multi_label)
plt.show()
# plt.figure(2)
# labeled_binary, unlabeled_binary, true_binary = GenerateGraph.SyntheticGraph(N, n, GraphType='Binary')
# model_2 = LPA.LPA(labeled_binary, unlabeled_binary)
# binary_label, full_binary = model_2.labelPropImp(epsilon, weight_fun='Epsilon', alpha=0.5, iterMax=1e4, tol=1e-5)
# model_2.showResult(binary_label)
#
# plt.figure(3)
# model = TSVM.TSVM()
# model.initial()
# model.train(labeled_binary[:, 0:-1], 2 * labeled_binary[:, -1] - 1, unlabeled_binary)
# Y_hat = model.predict(unlabeled_binary)
# Y_hat = Y_hat.reshape((Y_hat.shape[0], 1))
# model.showResult(labeled_binary[:, 0:-1], unlabeled_binary, labeled_binary[:, -1], Y_hat)
