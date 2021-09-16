from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import PlotStyle
from sklearn.manifold import TSNE
import LPA
import pandas as pd


def plot_embedding(data, label):
    # Normalize the data
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    ax = plt.subplot(111)
    colors = list(mcolors.TABLEAU_COLORS.keys())
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]), color=mcolors.TABLEAU_COLORS[colors[int(label[i])]],
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xlim((-0.1, 1.1))
    plt.ylim((-0.1, 1.1))
    return fig


def main_1():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    unlabeled_num = 5000  # number of unlabeled nodes
    feature_dim = mnist.train.images.shape[1]  # dimension of the features
    train_images = mnist.train.images  # data of nodes
    train_labels = mnist.train.labels  # labels of nodes

    train_labels = np.argmax(train_labels, axis=1)
    true_labels = train_labels[5000: 5000 + unlabeled_num]
    train_labels = train_labels.reshape((train_labels.shape[0], 1))

    # Initialize the unlabeled nodes and the corresponding labels
    unlabeled_data = train_images[5000: 5000 + unlabeled_num, :]
    label_size_data = range(250, 270, 10)
    # ratio of labeled nodes in training
    len_label = len(label_size_data)
    label_ratio = np.zeros((len_label, 1))
    for i in range(len_label):
        label_ratio[i, 0] = label_size_data[i] / (unlabeled_num + label_size_data[i])
    accu_data = np.zeros((len_label, 1))  # Initialize the array to store the accuracy of the model

    true_labels = true_labels.reshape((unlabeled_num, 1))
    for i in range(len_label):
        labeled_num = int(label_size_data[i])
        labeled_data = np.zeros((labeled_num, feature_dim + 1))
        # Set the unlabeled nodes equally w.r.t their nodes
        for j in range(labeled_num):
            temp_label = j % 10
            temp_index = np.where(train_labels == temp_label)[0]
            labeled_data[j, :] = np.hstack((train_images[temp_index[j // 10], :], np.array([temp_label])))
        model_MNIST = LPA.LPA(labeled_data, unlabeled_data)
        vec_label, full_label = model_MNIST.labelPropImp(epsilon=1, weight_fun='Gaussian')
        accu_data[i, 0] = model_MNIST.accuracy(vec_label, true_labels)
        print(vec_label)
        print('Label size is', labeled_num, ', accuracy of the model is', accu_data[i, 0])

    plt.figure(1)
    plt.plot(label_size_data, accu_data)
    plt.xlabel('labeled nodes ratio', fontsize=15)
    plt.ylabel('accuracy')
    save_data = np.hstack((label_ratio, accu_data))
    save = pd.DataFrame(save_data)
    path = 'data/TestMNIST.csv'
    save.to_csv(path, mode='a', header=False, index=False)

    tsne = TSNE(n_components=2, init='pca', random_state=33, perplexity=50)
    result = tsne.fit_transform(unlabeled_data)
    fig_true = plot_embedding(result, true_labels)
    fig_train = plot_embedding(result, vec_label[labeled_num:])

    plt.show()


def main_2():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    labeled_num = 500
    unlabeled_num = range(500, 54500, 1000)  # number of unlabeled nodes
    feature_dim = mnist.train.images.shape[1]  # dimension of the features
    train_images = mnist.train.images  # data of nodes
    train_labels = mnist.train.labels  # labels of nodes


if __name__ == '__main__':
    main_1()
    main_2()
