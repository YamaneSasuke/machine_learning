# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:38:00 2016

@author: yamane
"""

import numpy as np
from sklearn import datasets


def logistic(z):
    return 1.0 / (1 + np.exp(-z))


# main文
if __name__ == '__main__':
    # Load the digits dataset
    digits = datasets.load_digits(2)

    # データ格納
    X = digits.data
    # 正解ラベル格納
    t = digits.target
    # データ数
    num_examples = len(X)

    # 学習率を定義
    rho = 0.1
    # 最大の繰り返し回数
    max_iteration = 100
    # 重みの初期値
    w = np.random.randn(X.shape[1])

    # 正解クラスと予測クラスとの比較
    y = logistic(np.inner(w, X))
    predict_class = y >= 0.5
    num_correct = np.sum(t == predict_class)
    correct_percent = num_correct / float(num_examples) * 100
    print "correct_percent:", correct_percent

    # すべて正しく認識できるまで繰り返す
    for epoch in range(max_iteration):
        # 入力データXと正解ラベルを取り出す
        for x_i, t_i in zip(X, t):
            # logistic関数を計算
            y_i = logistic(np.inner(w, x_i))
            # 勾配降下法
            w -= rho * (y_i - t_i) * x_i
        # 学習率を小さくする
        # rho *= 0.9

        # 正解クラスと予測クラスとの比較
        y = logistic(np.inner(w, X))
        predict_class = y >= 0.5
        num_correct = np.sum(t == predict_class)
        correct_percent = num_correct / float(num_examples) * 100
        print "correct_percent:", correct_percent
        if correct_percent == 100.0:
            break
