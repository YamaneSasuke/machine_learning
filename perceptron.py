# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:38:00 2016

@author: yamane
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# main文
if __name__ == '__main__':
    # Load the digits dataset
    digits = datasets.load_digits(2)
    # plt.gray()
    # plt.matshow(digits.images[1])
    # plt.show()

    # データ格納
    X = digits.data
    # 正解ラベル格納
    t = digits.target
    t[t == 0] = -1
    num_examples = len(X)

    # ρを定義
    rho = 0.5
    # 最大の繰り返し回数
    max_iteration = 100
    # 重みの初期値
    w = np.random.randn(X.shape[1])

    y = np.sign(np.inner(w, X))
    # 正解の個数
    correct = np.sum(y == t)
    # 正解率
    percent = correct / float(num_examples) * 100
    print percent

    # すべて正しく認識できるまで繰り返す
    for epoch in range(max_iteration):
        # 入力データXと正解ラベルを取り出す
        for x_i, t_i in zip(X, t):
            # 識別関数
            g_i = np.inner(w, x_i)
            # 誤認識が起きた場合、wを修正
            if t_i * g_i < 0:
                w_new = w + rho * x_i * t_i
            else:
                w_new = w
            w = w_new

        y = np.sign(np.inner(w, X))
        # 正解の個数
        correct = np.sum(y == t)
        # 正解率
        percent = correct / float(num_examples) * 100
        print percent
        if percent == 100:
            break
