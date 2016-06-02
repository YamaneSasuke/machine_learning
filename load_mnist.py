# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 19:35:52 2015

@author: ryuhei
"""

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


def load_mnist():
    '''
    Load the digits dataset
    fetch_mldata ... dataname is on mldata.org, data_home
    load 10 classes, from 0 to 9
    '''
    mnist = datasets.fetch_mldata('MNIST original')
    n_train = 60000  # The size of training set
    # Split dataset into training set (60000) and testing set (10000)
    mnist.data = mnist.data
    data_train = mnist.data[:n_train]
    target_train = mnist.target[:n_train]
    data_test = mnist.data[n_train:]
    target_test = mnist.target[n_train:]
    return (data_train.astype(np.float32), target_train.astype(np.float32),
            data_test.astype(np.float32), target_test.astype(np.float32))


if __name__ == '__main__':
    x_train, t_train, x_test, t_test = load_mnist()
    num_train, D = x_train.shape
    num_test = len(x_test)

    print "x_train.shape:", x_train.shape
    print "t_train.shape:", t_train.shape
    print "x_test.shape:", x_test.shape
    print "t_test.shape:", t_test.shape

    plt.matshow(x_train[0].reshape(28, 28), cmap=plt.cm.gray)
    plt.show()
