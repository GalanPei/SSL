import LPA
import numpy as np
import math
import matplotlib.pyplot as plt


def labelProp(labeled_data, unlabeled_data, epsilon, weight_fun='Gaussian', iterMax=1e4, tol=1e-5):
    l = labeled_data.shape[0]
    u = unlabeled_data.shape[0]
    labels = labeled_data[:, -1]
    label_list = np.unique(labels)
    y = label_list.shape[0]
    learning_data = np.vstack((labeled_data[:, 0:-1], unlabeled_data))
    mat_W = np.zeros((l + u, l + u))
    inv_D = np.zeros((l + u, l + u))
    f = np.zeros((l + u, y))
    for i in range(l):
        f[i, labels[i]] = 1
    for i in range(l + u):
        for j in range(l + u):
            if weight_fun == 'Gaussian':
                mat_W[i, j] = LPA.gaussianWeight(np.linalg.norm(learning_data[i, :] - learning_data[j, :]), epsilon)
            elif weight_fun == 'Epsilon':
                mat_W[i, j] = LPA.epsilonWeight(np.linalg.norm(learning_data[i, :] - learning_data[j, :]), epsilon)
        inv_D[i, i] = 1 / np.sum(mat_W[i, :])
    mat_Prop = np.dot(inv_D, mat_W)
    for i in range(int(iterMax)):
        f_old = f
        f = np.dot(mat_Prop, f)
        # Recurrence expression of the iteration
        if np.linalg.norm(f - f_old) < tol:
            break
    vec_label = np.argmax(f, axis=1)
    return vec_label
