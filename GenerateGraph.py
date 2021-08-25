import numpy as np
import random
import math
import matplotlib.pyplot as plt


def SyntheticGraph(N, n=10, GraphType='Multi'):
    theta_1 = np.random.uniform(np.pi, 7 / 4 * np.pi, size=(N, 1))
    theta_2 = np.random.uniform(-0.6 * np.pi, 1 / 9 * np.pi, size=(N, 1))
    theta_3 = np.random.uniform(0, np.pi, size=(N, 1))
    r_1 = np.random.uniform(2.5, 3.5, size=(N, 1))
    r_2 = np.random.uniform(2.5, 3.5, size=(N, 1))
    r_3 = np.random.uniform(2.5, 3.5, size=(N, 1))
    x_1 = 4 + r_1 * np.cos(theta_1)
    y_1 = 4 + r_1 * np.sin(theta_1)
    x_2 = 3 + r_2 * np.cos(theta_2)
    y_2 = 6 + r_2 * np.sin(theta_2)
    x_3 = 6 + r_3 * np.cos(theta_3)
    y_3 = 5 + r_3 * np.sin(theta_3)
    labeledData = np.zeros((n, 3))
    sample = random.sample(list(range(N)), n)
    if GraphType == 'Multi':
        xData = np.vstack((x_1, x_2, x_3))
        yData = np.vstack((y_1, y_2, y_3))
        for i in range(n):
            if i % 3 == 1:
                theta = theta_1[sample[i], 0]
                labeledData[i, :] = np.array([4 + 3 * np.cos(theta), 4 + 3 * np.sin(theta), 0])
            if i % 3 == 2:
                theta = theta_2[sample[i], 0]
                labeledData[i, :] = np.array([3 + 3 * np.cos(theta), 6 + 3 * np.sin(theta), 1])
            if i % 3 == 0:
                theta = theta_3[sample[i], 0]
                labeledData[i, :] = np.array([6 + 3 * np.cos(theta), 5 + 3 * np.sin(theta), 2])
        unlabeledData = np.hstack((xData, yData))
        true_label = np.vstack((np.zeros((N, 1)), np.ones((N, 1)), 2 * np.ones((N, 1))))
        return labeledData, unlabeledData, true_label
    if GraphType == 'Binary':
        xData = np.vstack((x_2, x_3))
        yData = np.vstack((y_2, y_3))
        for i in range(n):
            if i % 2 == 1:
                theta = theta_2[sample[i], 0]
                labeledData[i, :] = np.array([3 + 3 * np.cos(theta), 6 + 3 * np.sin(theta), 0])
            if i % 2 == 0:
                theta = theta_3[sample[i], 0]
                labeledData[i, :] = np.array([6 + 3 * np.cos(theta), 5 + 3 * np.sin(theta), 1])
        unlabeledData = np.hstack((xData, yData))
        true_label = np.vstack((-1 * np.ones((N, 1)), np.ones((N, 1))))
        return labeledData, unlabeledData, true_label
