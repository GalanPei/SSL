import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mpl_toolkits.axisartist as AA
import PlotStyle

database = 'cora'
model = ['gcn', 'gcn_test1', 'gcn_test2', 'gcn_test3']
colors = list(mcolors.TABLEAU_COLORS.keys())
max_epoch = 100

fig = plt.figure()
ax = AA.Subplot(fig,111)
fig.add_axes(ax)
for i in range(len(model)):
    epoch_num, accuracy = np.loadtxt('data/' + database + '_' + model[i] + '_testAdj.csv', dtype=np.str,
                                     delimiter=",", usecols=(0, 2), unpack=True)
    epoch_num = epoch_num[2:max_epoch].astype(np.float32)
    accuracy = accuracy[2:max_epoch].astype(np.float32)
    plt.plot(epoch_num, accuracy, linewidth=0.9, color=mcolors.TABLEAU_COLORS[colors[int(i)]])
plt.legend((r'$A$', r'$A_1$', r'$A_2$', r'$A_3$'))
# plt.xticks((0, 5, 10, 15), (r'$0$', r'$5$', r'$10$', r'$15$'))
plt.yticks((0, 0.3, 0.6, 0.9), (r'$0$', r'$30\%$', r'$60\%$', r'$90\%$'))
plt.xlabel(r'Epoch', fontsize=16)
plt.ylabel(r'Accuracy', fontsize=16)
plt.show()
