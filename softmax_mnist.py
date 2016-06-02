# -*- coding: utf-8 -*-
"""
Created on Wed Jun 01 12:40:19 2016

@author: yamane
"""

import load_mnist
import numpy as np
import scipy
import matplotlib.pyplot as plt


def softmax_logsumexp(z):
    return np.exp(z - scipy.misc.logsumexp(z, axis=1, keepdims=True))


def softmax(z):
    exp_z = np.exp(z)
    assert(not np.any(np.isinf(exp_z)))
    return exp_z / exp_z.sum(1, keepdims=True)


def check_softmax(z):
    assert(np.allclose(softmax(z), softmax_logsumexp(z)))


def test_softmax():
    check_softmax([[1, 2, 3]])
    check_softmax([[1, 2, 3],
                   [4, 5, 6]])
test_softmax()


def onehot(k, num_classes=10):
    t_onehot = np.zeros((len(k), num_classes))
    for i, k_i in enumerate(k):
        t_onehot[i][k_i] = 1
    return t_onehot


def score(X, t, w):
    # 正解クラスと予測クラスとの比較
    y = softmax_logsumexp(np.inner(X, w))
    predict_class = np.argmax(y, axis=1)
    return np.mean(t == predict_class)


if __name__ == '__main__':
    x_train, t_train, x_test, t_test = load_mnist.load_mnist()

    X_raw = x_train / 255.0
    X_test_raw = x_test / 255.0
    t = t_train
    num_examples = len(X_raw)
    num_examples_test = len(X_test_raw)
    X = np.hstack((X_raw, np.ones((num_examples, 1))))
    X_test = np.hstack((X_test_raw, np.ones((num_examples_test, 1))))
    num_classes = len(np.unique(t))
    num_features = X.shape[1]

    # 超パラメータ
    max_iteration = 100
    batch_size = 100
    rho = 0.5  # 学習率

    # num_features次元の重みをnum_classesクラス分用意する
    w = np.random.randn(num_classes, num_features)

    num_batches = num_examples / batch_size
    corrects = []  # グラフ描画用の配列
    for epoch in range(max_iteration):
        # 入力データXと正解ラベルを取り出す
        permu = np.random.permutation(num_examples)
        for indexes in np.array_split(permu, num_batches):
            x_batch = X[indexes]
            t_batch = t[indexes]
            # softmax関数を計算
            assert(not np.any(np.isnan(w)))
            y_batch = softmax_logsumexp(np.dot(x_batch, w.T))
            assert(not np.any(np.isnan(y_batch)))
            # one-hotを適用
            T = onehot(t_batch)
            # 勾配降下法
            w -= rho * np.dot((y_batch - T).T, x_batch)

        correct_rate = score(X_test, t_test, w)
        corrects.append(correct_rate)
        plt.plot(corrects)
        plt.show()
        print "epoch:", epoch, "correct:", correct_rate

        if correct_rate == 1.0:
            break
