# -*- coding: utf-8 -*-
"""
Created on Fri May 27 13:15:20 2016

@author: yamane
"""

import numpy as np
from sklearn import datasets


def softmax(z):
    return np.exp(z) / np.exp(z).sum(1, keepdims=True)


def onehot(k, num_classes=10):
    t_onehot = np.zeros((len(k), num_classes))
    for i, k_i in enumerate(k):
        t_onehot[i][k_i] = 1
    return t_onehot


def score(X, t, w):
    # 正解クラスと予測クラスとの比較
    y = softmax(np.inner(X, w))
    predict_class = np.argmax(y, axis=1)
    return np.mean(t == predict_class)


if __name__ == '__main__':
    # データ・セットの読み込み
    digits = datasets.load_digits()
    X_raw = digits.data / 16.0
    t = digits.target
    num_examples = len(X_raw)
    X = np.hstack((X_raw, np.ones((num_examples, 1))))
    num_classes = len(np.unique(t))
    num_features = X.shape[1]

    # 超パラメータ
    max_iteration = 1000
    batch_size = 100
    rho = 0.5  # 学習率

    # num_features次元の重みをnum_classesクラス分用意する
    w = np.random.randn(num_classes, num_features)

    num_batches = num_examples / batch_size
    for epoch in range(max_iteration):
        # 入力データXと正解ラベルを取り出す
        permu = np.random.permutation(num_examples)
        for indexes in np.array_split(permu, num_batches):
            x_batch = X[indexes]
            t_batch = t[indexes]
            # softmax関数を計算
            assert(not np.any(np.isnan(w)))
            y_batch = softmax(np.dot(x_batch, w.T))
            assert(not np.any(np.isnan(y_batch)))
            # one-hotを適用
            T = onehot(t_batch)
            # 勾配降下法
            w -= rho * np.dot((y_batch - T).T, x_batch)

        correct_rate = score(X, t, w)
        print "epoch:", epoch, "correct:", correct_rate

        if correct_rate == 1.0:
            break
