import numpy as np
import math
import matplotlib.pyplot as plt


def SyntheticGraph(N, GraphType='Multi'):
    theta_1 = np.random.uniform(np.pi, 7 / 4 * np.pi, size=(N, 1))
    theta_2 = np.random.uniform(3 / 2 * np.pi, 2 * np.pi, size=(N, 1))
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
    if GraphType == 'Multi':
        xData = np.vstack((x_1, x_2, x_3))
        yData = np.vstack((y_1, y_2, y_3))
        labeledData = np.array([[4 - 3 / math.sqrt(2), 4 - 3 / math.sqrt(2), 0],
                                [3 + 3 * np.cos(7 / 4 * np.pi), 6 + 3 * np.sin(7 / 4 * np.pi), 1],
                                [6, 8, 2]])
        unlabeledData = np.hstack((xData, yData))
        true_label = np.vstack((np.zeros((N, 1)), np.ones((N, 1)), 2 * np.ones((N, 1))))
        return labeledData, unlabeledData, true_label
    if GraphType == 'Binary':
        xData = np.vstack((x_2, x_3))
        yData = np.vstack((y_2, y_3))
        labeledData = np.array([[3 + 3 * np.cos(7 / 4 * np.pi), 6 + 3 * np.sin(7 / 4 * np.pi), 0],
                                [6, 8, 1]])
        unlabeledData = np.hstack((xData, yData))
        true_label = np.vstack((-1 * np.ones((N, 1)), np.ones((N, 1))))
        return labeledData, unlabeledData, true_label
