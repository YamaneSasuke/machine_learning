# -*- coding: utf-8 -*-
"""
Created on Fri May 27 13:15:20 2016

@author: yamane
"""

import numpy as np
from sklearn import datasets


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))


def onehot(k, num_classes=10):
    t_onehot = np.zeros(num_classes)
    t_onehot[k] = 1
    return t_onehot

# main文
if __name__ == '__main__':

    # Load the digits dataset
    digits = datasets.load_digits()

    # データ・セットの読み込み
    X_raw = digits.data / 16.0
    t = digits.target
    num_examples = len(X_raw)
    X = np.hstack((X_raw, np.ones((num_examples, 1))))
    # ρを定義する(ρ=0.1で良いか判断し，収束しなければ値を変える．)
    rho = 0.5

    # 収束するまで繰り返す
    max_iteration = 1000

    # dim_features次元の重みをnum_classesクラス分用意する
    w = np.random.randn(len(np.unique(t)), X.shape[1])

    # 確率的勾配降下法
    error_history = []
    correct_percent_history = []

    # 正解クラスと予測クラスとの比較
    y = softmax(np.inner(X, w))
    predict_class = np.argmax(y, axis=1)
    num_correct = np.sum(t == predict_class)
    correct_percent = num_correct / float(num_examples) * 100
    print "correct_percent:", correct_percent
    correct_percent_history.append(correct_percent)

    for epoch in range(max_iteration):
        # 入力データXと正解ラベルを取り出す
        for x_i, t_i in zip(X, t):
            # softmax関数を計算
            y_i = softmax(np.dot(x_i, w.T))
            # one-hotを適用
            T = onehot(t_i)
            # 勾配降下法
            w -= rho * np.expand_dims(y_i - T, 1) * x_i

        # 正解クラスと予測クラスとの比較
        y = softmax(np.dot(X, w.T))
        predict_class = np.argmax(y, axis=1)
        num_correct = np.sum(t == predict_class)
        correct_percent = num_correct / float(num_examples) * 100
        print "correct_percent:", correct_percent
        correct_percent_history.append(correct_percent)

        if correct_percent == 100.0:
            break
