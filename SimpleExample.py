import PlotStyle
import GenerateGraph
import LPA
import matplotlib.pyplot as plt

N = 1000
n = 200
epsilon = 0.8
labeled_binary, unlabeled_binary, true_binary = GenerateGraph.SimpleGraph(N, n)
model = LPA.LPA(labeled_binary, unlabeled_binary, type='Binary')
vec_binary, full_binary = model.labelPropOri(epsilon, weight_fun='Epsilon')
model.showResult(vec_binary)
plt.show()
model.PlotWeight(full_binary)
plt.show()