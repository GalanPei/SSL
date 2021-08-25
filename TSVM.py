# coding:utf-8
# This code is the original work of https://github.com/horcham/TSVM
import numpy as np
import sklearn.svm as svm
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import matplotlib.colors as mcolors
import pickle
from sklearn.model_selection import train_test_split, cross_val_score

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


class TSVM(object):
    def __init__(self):
        pass

    def initial(self, kernel='linear'):
        '''
        Initial TSVM
        Parameters
        ----------
        kernel: kernel of svm
        '''
        self.Cl, self.Cu = 1.5, 0.001
        self.kernel = kernel
        self.clf = svm.SVC(C=1.5, kernel=self.kernel)

    def load(self, model_path='./TSVM.model'):
        '''
        Load TSVM from model_path
        Parameters
        ----------
        model_path: model path of TSVM
                        model should be svm in sklearn and saved by sklearn.externals.joblib
        '''
        self.clf = joblib.load(model_path)

    def train(self, X1, Y1, X2):
        '''
        Train TSVM by X1, Y1, X2
        Parameters
        ----------
        X1: Input data with labels
                np.array, shape:[n1, m], n1: numbers of samples with labels, m: numbers of features
        Y1: labels of X1
                np.array, shape:[n1, ], n1: numbers of samples with labels
        X2: Input data without labels
                np.array, shape:[n2, m], n2: numbers of samples without labels, m: numbers of features
        '''
        N = len(X1) + len(X2)
        sample_weight = np.ones(N)
        sample_weight[len(X1):] = self.Cu

        self.clf.fit(X1, Y1)
        Y2 = self.clf.predict(X2)
        Y2 = np.expand_dims(Y2, 1)
        X2_id = np.arange(len(X2))
        X3 = np.vstack([X1, X2])
        Y1 = Y1.reshape((Y1.shape[0], 1))
        Y3 = np.vstack([Y1, Y2])

        while self.Cu < self.Cl:
            self.clf.fit(X3, Y3, sample_weight=sample_weight)
            while True:
                Y2_d = self.clf.decision_function(X2)    # linear: w^Tx + b
                Y2 = Y2.reshape(-1)
                epsilon = 1 - Y2 * Y2_d   # calculate function margin
                positive_set, positive_id = epsilon[Y2 > 0], X2_id[Y2 > 0]
                negative_set, negative_id = epsilon[Y2 < 0], X2_id[Y2 < 0]
                positive_max_id = positive_id[np.argmax(positive_set)]
                negative_max_id = negative_id[np.argmax(negative_set)]
                a, b = epsilon[positive_max_id], epsilon[negative_max_id]
                if a > 0 and b > 0 and a + b > 2.0:
                    Y2[positive_max_id] = Y2[positive_max_id] * -1
                    Y2[negative_max_id] = Y2[negative_max_id] * -1
                    Y2 = np.expand_dims(Y2, 1)
                    Y3 = np.vstack([Y1, Y2])
                    self.clf.fit(X3, Y3, sample_weight=sample_weight)
                else:
                    break
            self.Cu = min(2*self.Cu, self.Cl)
            sample_weight[len(X1):] = self.Cu

    def score(self, X, Y):
        """
        Calculate accuracy of TSVM by X, Y
        Parameters
        ----------
        X: Input data
                np.array, shape:[n, m], n: numbers of samples, m: numbers of features
        Y: labels of X
                np.array, shape:[n, ], n: numbers of samples
        Returns
        -------
        Accuracy of TSVM
                float
        """
        return self.clf.score(X, Y)

    def predict(self, X):
        """
        Feed X and predict Y by TSVM
        Parameters
        ----------
        X: Input data
                np.array, shape:[n, m], n: numbers of samples, m: numbers of features
        Returns
        -------
        labels of X
                np.array, shape:[n, ], n: numbers of samples
        """
        return self.clf.predict(X)

    def save(self, path='./TSVM.model'):
        """
        Save TSVM to model_path
        Parameters
        ----------
        model_path: model path of TSVM
                        model should be svm in sklearn
        """
        joblib.dump(self.clf, path)

    def showResult(self, X1, X2, Y1, Y2):
        """
        :param X1: labeled data
                   np.array, shape[l, m], n: number of labeled data, m: number of features
        :param X2: unlabeled data
                   np.array, shape[n, m], n: number of unlabeled data, m: number of features
        :param Y1: labels of labeled data
                   np.array, shape[n, ], n: number of labeled data
        :param Y2: labels of unlabeled data
                   np.array, shape[n, ], n: number of unlabeled data
        :return: the plot of the train result
        """
        Y_u = 1/2*Y2 + 1
        colors = list(mcolors.TABLEAU_COLORS.keys())
        # Plot the unlabeled nodes as hollow circles
        for i in range(X2.shape[0]):
            plt.plot(X2[i, 0], X2[i, 1], 'o', markersize=2,
                     color=mcolors.TABLEAU_COLORS[colors[int(Y_u[i])]], markerfacecolor='white')
        # Plot the labeled nodes as filled triangles
        for i in range(X1.shape[0]):
            plt.plot(X1[i, 0], X1[i, 1], '^', markersize=4,
                     color='black', markerfacecolor=mcolors.TABLEAU_COLORS[colors[int(Y1[i])]])
        plt.xlabel(r'$x_1$', fontsize=14)
        plt.ylabel(r'$x_2$', fontsize=14)