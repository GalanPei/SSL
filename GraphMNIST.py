import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mpl_toolkits.axisartist as AA
import PlotStyle

label_ratio, accu_rate = np.loadtxt('data/TestMNIST.csv', dtype=np.float32, delimiter=",", usecols=(0, 1), unpack=True)
plt.plot(label_ratio, accu_rate, linewidth='0.9', color='k')
plt.xticks((0, 0.01, 0.02,0.03, 0.04, 0.05), (r'0', r'$1$\%', r'$2\%$', r'$3\%$', r'$4\%$', r'$5\%$'))
plt.yticks((0.45, 0.55, 0.65, 0.75), (r'$45$\%', r'$55\%$', r'$65\%$', r'$75\%$'))
plt.tick_params(labelsize=12)
plt.xlabel(r'Labeled ratio', fontsize=14)
plt.ylabel(r'Accuracy', fontsize=14)
plt.show()