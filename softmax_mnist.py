# -*- coding: utf-8 -*-
"""
Created on Wed Jun 01 12:40:19 2016

@author: yamane
"""

import load_mnist
import numpy as np
from sklearn.cross_validation import train_test_split
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
    # 訓練用データ
    X_raw = x_train / 255.0
    num_train = len(X_raw)
    x = np.hstack((X_raw, np.ones((num_train, 1))))
    X_train, X_valid, T_train, t_valid = train_test_split(x,
                                                          t_train,
                                                          test_size=0.1,
                                                          random_state=10)
    num_train = len(X_train)
    num_classes = len(np.unique(T_train))
    num_features = X_train.shape[1]
    # テスト用データ
    X_test_raw = x_test / 255.0
    num_test = len(X_test_raw)
    X_test = np.hstack((X_test_raw, np.ones((num_test, 1))))

    # 超パラメータ
    max_iteration = 500
    batch_size = 100
    rho = 0.0001  # 学習率
    w_scale = 0.01

    # num_features次元の重みをnum_classesクラス分用意する
    w = w_scale * np.random.randn(num_classes, num_features)
    w[:, -1] = 0  # バイアスパラメータの初期値

    num_batches = num_train / batch_size
    corrects = []  # グラフ描画用の配列
    correct_rate_best = 0
    for epoch in range(max_iteration):
        # 入力データXと正解ラベルを取り出す
        permu = np.random.permutation(num_train)
        for indexes in np.array_split(permu, num_batches):
            x_batch = X_train[indexes]
            t_batch = T_train[indexes]
            # softmax関数を計算
            assert(not np.any(np.isnan(w)))
            y_batch = softmax_logsumexp(np.dot(x_batch, w.T))
            assert(not np.any(np.isnan(y_batch)))
            # one-hotを適用
            T = onehot(t_batch)
            # 勾配降下法
            w -= rho * np.dot((y_batch - T).T, x_batch)

        print "epoch:", epoch
        correct_rate = score(X_valid, t_valid, w)
        if correct_rate > correct_rate_best:
            w_best = w
            correct_rate_best = correct_rate
            epoch_best = epoch
        corrects.append(correct_rate)
        print "correct:", correct_rate
        print "best_correct", correct_rate_best, "best_epoch", epoch_best
        print "best_w", w_best
        plt.plot(corrects)
        plt.show()

    test_correct_rate = score(X_test, t_test, w_best)
    print "test_correct_rate", test_correct_rate
