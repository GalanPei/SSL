import GenerateGraph
import LPA
import matplotlib.pyplot as plt
import TSVM
import numpy as np

N = 500
n = 20
epsilon = 0.8

plt.figure(1)
labeled_multi, unlabeled_multi, true_multi = GenerateGraph.SyntheticGraph(N, n)
model_1 = LPA.LPA(labeled_multi, unlabeled_multi)
multi_label = model_1.labelPropImp(epsilon, weight_fun='Epsilon', alpha=0.5, iterMax=1e4, tol=1e-5)
model_1.showResult(multi_label)
plt.show()

plt.figure(2)
labeled_binary, unlabeled_binary, true_binary = GenerateGraph.SyntheticGraph(N, n, GraphType='Binary')
model_2 = LPA.LPA(labeled_binary, unlabeled_binary)
binary_label = model_2.labelPropImp(epsilon, weight_fun='Epsilon', alpha=0.5, iterMax=1e4, tol=1e-5)
model_2.showResult(binary_label)
plt.show()

plt.figure(3)
model = TSVM.TSVM()
model.initial()
model.train(labeled_binary[:, 0:-1], 2 * labeled_binary[:, -1] - 1, unlabeled_binary)
Y_hat = model.predict(unlabeled_binary)
Y_hat = Y_hat.reshape((Y_hat.shape[0], 1))
data = np.vstack((labeled_binary[:, 0:-1], unlabeled_binary))
L = 2 * labeled_binary[:, -1] - 1
L = L.reshape((L.shape[0], 1))
label = np.vstack((L, Y_hat))
model.showResult(data, label)
plt.show()
