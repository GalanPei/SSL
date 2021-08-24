import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


class LPA(object):
    def __init__(self, labeled_data, unlabeled_data):
        self.labeled_data = labeled_data
        self.unlabeled_data = unlabeled_data

    def epsilonWeight(self, t, epsilon):
        if t < epsilon:
            # return 1
            return 1 / (epsilon ** 2)
        else:
            return 0

    def gaussianWeight(self, t, epsilon):
        return 1 / math.sqrt(2 * math.pi) / (epsilon ** 2) * math.exp(-t ** 2 / 2 / epsilon ** 2)
        # return 1 / math.sqrt(2 * math.pi) * math.exp(-t ** 2 / 2 / epsilon ** 2)

    def propagationMatrix(self, epsilon, weight_fun):
        l = self.labeled_data.shape[0]  # l - number of labeled data
        u = self.unlabeled_data.shape[0]  # u - number of unlabeled data
        # unlabeled_data = unlabeled_data.reshape((u, 2))
        learning_data = np.vstack((self.labeled_data[:, 0:-1], self.unlabeled_data))
        mat_W = np.zeros((l + u, l + u), np.float32)
        inv_sqrt_D = np.zeros((l + u, l + u), np.float32)
        for i in range(l + u):
            for j in range(l + u):
                if weight_fun == 'Gaussian':
                    mat_W[i, j] = self.gaussianWeight(np.linalg.norm(learning_data[i, :] - learning_data[j, :]),
                                                      epsilon)
                elif weight_fun == 'Epsilon':
                    mat_W[i, j] = self.epsilonWeight(np.linalg.norm(learning_data[i, :] - learning_data[j, :]),
                                                     epsilon)
            inv_sqrt_D[i, i] = 1 / math.sqrt(np.sum(mat_W[i, :]))
        matPropagation = np.linalg.multi_dot([inv_sqrt_D, mat_W, inv_sqrt_D])  # normalized Laplacian
        return matPropagation

    def labelPropImp(self, epsilon, weight_fun='Gaussian', alpha=0.5, iterMax=1e4, tol=1e-5):
        labels = self.labeled_data[:, -1]
        label_list = np.unique(labels)
        l = self.labeled_data.shape[0]
        u = self.unlabeled_data.shape[0]
        y = label_list.shape[0]
        F0 = np.zeros((l + u, y), np.float32)
        for i in range(l):
            F0[i, int(labels[i])] = 1
        full_label = F0
        S = self.propagationMatrix(epsilon, weight_fun)
        for iter in range(int(iterMax)):
            F_old = full_label
            full_label = alpha * np.dot(S, full_label) + (1 - alpha) * F0
            if np.linalg.norm(full_label - F_old) < tol:
                break
        vec_label = np.argmax(full_label, axis=1)
        return vec_label

    def labelPropOri(self, epsilon, weight_fun='Gaussian', iterMax=1e4, tol=1e-5):
        l = self.labeled_data.shape[0]
        u = self.unlabeled_data.shape[0]
        f_l = labeled_data[:, -1]
        # label_list = np.unique(f_l)
        for i in range(l):
            if f_l[i, 0] < 1e-5:
                f_l[i, 0] = -1
            else:
                f_l[i, 0] = 1
        learning_data = np.vstack((labeled_data[:, 0:-1], unlabeled_data))
        mat_W = np.zeros((l + u, l + u))
        inv_D = np.zeros((l + u, l + u))
        f_u = np.zeros((u, 1))
        for i in range(l + u):
            for j in range(l + u):
                if weight_fun == 'Gaussian':
                    mat_W[i, j] = self.gaussianWeight(np.linalg.norm(learning_data[i, :] - learning_data[j, :]),
                                                      epsilon)
                elif weight_fun == 'Epsilon':
                    mat_W[i, j] = self.epsilonWeight(np.linalg.norm(learning_data[i, :] - learning_data[j, :]),
                                                     epsilon)
            inv_D[i, i] = 1 / np.sum(mat_W[i, :])
        mat_Puu = np.dot(inv_D[l:l + u, l:l + u], mat_W[l:l + u, l:l + u])
        mat_Pul = np.dot(inv_D[l:l + u, l:l + u], mat_W[l:l + u, 0:l])
        for i in range(int(iterMax)):
            f_old = f_u
            f_u = np.dot(mat_Puu, f_u) + np.dot(mat_Pul, f_l)
            # Recurrence expression of the iteration
            if np.linalg.norm(f_u - f_old) < tol:
                break
        vec_label = np.sign(np.vstack((f_l, f_u)), axis=1)
        return vec_label + 1

    def showResult(self, vec_label):
        labels = self.labeled_data[:, -1]
        label_list = np.unique(labels)
        learning_data = np.vstack((self.labeled_data[:, 0:-1], self.unlabeled_data))
        colors = list(mcolors.TABLEAU_COLORS.keys())
        for i in range(vec_label.shape[0]):
            plt.plot(learning_data[i, 0], learning_data[i, 1], 'o', markersize=2,
                     color=mcolors.TABLEAU_COLORS[colors[int(vec_label[i])]])
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')




# # Label Propagation Algorithm
# def epsilonWeight(t, epsilon):
#     if t < epsilon:
#         # return 1
#         return 1 / (epsilon ** 2)
#     else:
#         return 0
#
#
# def gaussianWeight(t, epsilon):
#     return 1 / math.sqrt(2 * math.pi) / (epsilon ** 2) * math.exp(-t ** 2 / 2 / epsilon ** 2)
#     # return 1 / math.sqrt(2 * math.pi) * math.exp(-t ** 2 / 2 / epsilon ** 2)


# def kNN(dataSet, x, y, k, epsilon, weight_fun='Gaussian'):
#     dist = np.linalg.norm(x - y)
#     num_1 = 0
#     num_2 = 0
#     num_sample = dataSet.shape[0]
#     for i in range(num_sample):
#         if 0 < np.linalg.norm(dataSet[i, :] - x) < dist:
#             num_1 += 1
#         if num_1 == k:
#             break
#     for i in range(num_sample):
#         if 0 < np.linalg.norm(dataSet[i, :] - y) < dist:
#             num_2 += 1
#         if num_2 == k:
#             break
#     if num_1 < k or num_2 < k:
#         if weight_fun == 'Gaussian':
#             return gaussianWeight(x - y, epsilon)
#         if weight_fun == 'Epsilon':
#             return epsilonWeight(x - y, epsilon)
#     else:
#         return 0


# def propagationMatrix(labeled_data, unlabeled_data, epsilon, weight_fun):
#     l = labeled_data.shape[0]  # l - number of labeled data
#     u = unlabeled_data.shape[0]  # u - number of unlabeled data
#     # unlabeled_data = unlabeled_data.reshape((u, 2))
#     learning_data = np.vstack((labeled_data[:, 0:-1], unlabeled_data))
#     mat_W = np.zeros((l + u, l + u), np.float32)
#     inv_sqrt_D = np.zeros((l + u, l + u), np.float32)
#     for i in range(l + u):
#         for j in range(l + u):
#             if weight_fun == 'Gaussian':
#                 mat_W[i, j] = gaussianWeight(np.linalg.norm(learning_data[i, :] - learning_data[j, :]), epsilon)
#             elif weight_fun == 'Epsilon':
#                 mat_W[i, j] = epsilonWeight(np.linalg.norm(learning_data[i, :] - learning_data[j, :]), epsilon)
#         inv_sqrt_D[i, i] = 1 / math.sqrt(np.sum(mat_W[i, :]))
#     matPropagation = np.linalg.multi_dot([inv_sqrt_D, mat_W, inv_sqrt_D])  # normalized Laplacian
#     return matPropagation


# def labelPropImp(labeled_data, unlabeled_data, epsilon, weight_fun='Gaussian', alpha=0.5, iterMax=1e4, tol=1e-5):
#     labels = labeled_data[:, -1]
#     label_list = np.unique(labels)
#     l = labeled_data.shape[0]
#     u = unlabeled_data.shape[0]
#     y = label_list.shape[0]
#     F0 = np.zeros((l + u, y), np.float32)
#     for i in range(l):
#         F0[i, int(labels[i])] = 1
#     full_label = F0
#     S = propagationMatrix(labeled_data, unlabeled_data, epsilon, weight_fun)
#     for iter in range(int(iterMax)):
#         F_old = full_label
#         full_label = alpha * np.dot(S, full_label) + (1 - alpha) * F0
#         if np.linalg.norm(full_label - F_old) < tol:
#             break
#     vec_label = np.argmax(full_label, axis=1)
#     return vec_label


# def showResult(labeled_data, unlabeled_data, vec_label):
#     labels = labeled_data[:, -1]
#     label_list = np.unique(labels)
#     learning_data = np.vstack((labeled_data[:, 0:-1], unlabeled_data))
#     colors = list(mcolors.TABLEAU_COLORS.keys())
#     for i in range(vec_label.shape[0]):
#         plt.plot(learning_data[i, 0], learning_data[i, 1], 'o', markersize=2,
#                  color=mcolors.TABLEAU_COLORS[colors[int(vec_label[i])]])
#     plt.xlabel(r'$x_1$')
#     plt.ylabel(r'$x_2$')


# def labelPropOri(labeled_data, unlabeled_data, epsilon, weight_fun='Gaussian', iterMax=1e4, tol=1e-5):
#     l = labeled_data.shape[0]
#     u = unlabeled_data.shape[0]
#     f_l = labeled_data[:, -1]
#     label_list = np.unique(f_l)
#     for i in range(l):
#         if f_l[i, 0] == label_list[0]:
#             f_l[i, 0] = -1
#         else:
#             f_l[i, 0] = label_list[1]
#     learning_data = np.vstack((labeled_data[:, 0:-1], unlabeled_data))
#     mat_W = np.zeros((l + u, l + u))
#     inv_D = np.zeros((l + u, l + u))
#     f_u = np.zeros((u, 1))
#     for i in range(l + u):
#         for j in range(l + u):
#             if weight_fun == 'Gaussian':
#                 mat_W[i, j] = gaussianWeight(np.linalg.norm(learning_data[i, :] - learning_data[j, :]), epsilon)
#             elif weight_fun == 'Epsilon':
#                 mat_W[i, j] = epsilonWeight(np.linalg.norm(learning_data[i, :] - learning_data[j, :]), epsilon)
#         inv_D[i, i] = 1 / np.sum(mat_W[i, :])
#     mat_Puu = np.dot(inv_D[l:l + u, l:l + u], mat_W[l:l + u, l:l + u])
#     mat_Pul = np.dot(inv_D[l:l + u, l:l + u], mat_W[l:l + u, 0:l])
#     for i in range(int(iterMax)):
#         f_old = f_u
#         f_u = np.dot(mat_Puu, f_u) + np.dot(mat_Pul, f_l)
#         # Recurrence expression of the iteration
#         if np.linalg.norm(f_u - f_old) < tol:
#             break
#     vec_label = np.sign(np.vstack((f_l, f_u)), axis=1)
#     return vec_label + 1


# main function
if __name__ == "__main__":
    n = 200  # sample number
    d = 2  # dimension
    num_MC = 100  # number of Monte Carlo samples
    labeled_data = np.array([[0, 0, 0], [1, 1, 1]])
    l = labeled_data.shape[0]
    u = n
    num_sample = 20
    eps_data = np.linspace(0.01, 1, num=num_sample)
    mat_error = np.zeros((num_sample, 1))

    # Apply Monte Carlo method to estimate the accuracy of algorithm
    for i in range(num_MC):
        unlabeled_data = np.random.rand(n, d)
        true_data = np.vstack((labeled_data[:, 0:-1], unlabeled_data))
        true_data = np.hstack((true_data, np.zeros((l + u, 1))))
        for j in range(l + u):
            if np.sum(true_data[j, :]) > 1:
                true_data[j, -1] = 1
        for k in range(num_sample):
            vec_label = labelPropImp(labeled_data, unlabeled_data, eps_data[k], weight_fun='Epsilon', alpha=0.5,
                                     iterMax=1e4, tol=1e-5)
            mat_error[k, 0] += np.linalg.norm(vec_label - true_data[:, -1])
    mat_error *= 1 / num_MC

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(1)
    plt.plot(eps_data, mat_error)
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'$\|f(x)-y\|_2$')
    plt.show()

    # plt.figure(2)
    # plt.plot(eps_data, mat_error_2)

    # plt.figure(2)
    # showResult(labeled_data, unlabeled_data, vec_label)
    # plt.show()
