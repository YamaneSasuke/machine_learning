# -*- coding: utf-8 -*-
"""
Created on Mon Jun 06 18:10:08 2016

@author: yamane
"""

import load_mnist
import numpy as np
from sklearn.cross_validation import train_test_split
import scipy
import matplotlib.pyplot as plt


def softmax_logsumexp(z):
    return np.exp(z - scipy.misc.logsumexp(z, axis=1, keepdims=True))


def onehot(k, num_classes=10):
    assert(k.dtype == np.int32)
    t_onehot = np.zeros((len(k), num_classes))
    for i, k_i in enumerate(k):
        t_onehot[i][k_i] = 1
    return t_onehot


def score(X, t, w1, w2):
    assert(t.dtype == np.int32)
    # 入力変数の線形和
    a_z = np.dot(X, w1.T)

    # 隠れユニットの出力
    Z = np.tanh(a_z)

    # バイアスパラメータを追加
    Z_new = np.hstack((Z, np.ones((len(Z), 1))))

    # 隠れユニットの出力の線形和
    a_y = np.dot(Z_new, w2.T)

    # 出力ユニットの出力
    y = softmax_logsumexp(a_y)

    # 交差エントロピー損失　PRML式(4.108)
    loss = -np.log(y[range(len(X)), t]).mean()

    predict_class = np.argmax(y, axis=1)
    return np.mean(t == predict_class), loss


if __name__ == '__main__':
    x_train, t_train, x_test, t_test = load_mnist.load_mnist()
    # 訓練用データ
    X_raw = x_train / 255.0
    num_train = len(X_raw)
    x = np.hstack((X_raw, np.ones((num_train, 1))))
    t_train = t_train.astype(np.int32)
    t_test = t_test.astype(np.int32)
    X_train, X_valid, T_train, T_valid = train_test_split(x,
                                                          t_train,
                                                          test_size=0.1,
                                                          random_state=10)
    num_train = len(X_train)
    num_valid = len(X_valid)
    num_classes = len(np.unique(T_train))
    num_features = X_train.shape[1]
    # テスト用データ
    X_test_raw = x_test / 255.0
    num_test = len(X_test_raw)
    X_test = np.hstack((X_test_raw, np.ones((num_test, 1))))

    # 超パラメータ
    max_iteration = 300
    batch_size = 100
    rho = 0.01
    # 学習率
    w_scale = 0.01
    dim_hidden = 100

    # num_features次元の重みをnum_classesクラス分用意する
    w1 = w_scale * np.random.randn(dim_hidden, num_features)
    w1[:, -1] = 0  # バイアスパラメータの初期値
    w2 = w_scale * np.random.randn(num_classes, dim_hidden + 1)
    w2[:, -1] = 0  # バイアスパラメータの初期値

    num_batches = num_train / batch_size
    corrects_train = []  # グラフ描画用の配列
    corrects_valid = []  # グラフ描画用の配列
    loss_trains = []
    loss_valids = []
    correct_rate_best = 0
    for epoch in range(max_iteration):
        # 入力データXと正解ラベルを取り出す
        permu = np.random.permutation(num_train)
        for indexes in np.array_split(permu, num_batches):
            x_batch = X_train[indexes]
            t_batch = T_train[indexes]
            this_batch_size = len(indexes)
            T = onehot(t_batch)

            # 入力変数の線形和
            a_z = np.dot(x_batch, w1.T)

            # 隠れユニットの出力
            Z = np.tanh(a_z)

            # バイアスパラメータを追加
            Z_new = np.hstack((Z, np.ones((len(Z), 1))))

            # 隠れユニットの出力の線形和
            a_y = np.dot(Z_new, w2.T)

            # 出力ユニットの出力
            Y = softmax_logsumexp(a_y)

            # w2に関する勾配
            w2_grad = np.dot((Y - T).T, Z_new) / this_batch_size

            z_grad = np.dot((Y - T), w2)

            d_z = z_grad * (np.ones((this_batch_size,
                                     dim_hidden + 1)) - Z_new**2)

            # w1に関する勾配
            w1_grad = np.dot(d_z.T, x_batch) / this_batch_size

            # パラメータを更新
            w1 -= rho * w1_grad[:-1, :]
            w2 -= rho * w2_grad

        correct_rate_train, loss_train = score(X_train, T_train, w1, w2)
        correct_rate_valid, loss_valid = score(X_valid, T_valid, w1, w2)
        if correct_rate_valid > correct_rate_best:
            w1_best = w1
            w2_best = w2
            correct_rate_best = correct_rate_valid
            epoch_best = epoch
        corrects_train.append(correct_rate_train)
        corrects_valid.append(correct_rate_valid)
        loss_trains.append(loss_train)
        loss_valids.append(loss_valid)

        # 正解率、損失を表示
        print "epoch:", epoch
        print "[train] correct:", correct_rate_train
        print "[valid] correct:", correct_rate_valid
        print "[train] loss:", loss_train
        print "[valid] loss:", loss_valid
        print "best_correct", correct_rate_best, "best_epoch", epoch_best

        plt.plot(corrects_train)
        plt.plot(corrects_valid)
        plt.title("correct")
        plt.legend(["train", "valid"], loc="lower right")
        plt.grid()
        plt.show()

        plt.plot(loss_trains, label="train")
        plt.plot(loss_valids, label="valid")
        plt.title("loss")
        plt.legend(loc="upper right")
        plt.grid()
        plt.show()

    # テストデータの結果を表示
    test_correct_rate, test_loss = score(X_test, t_test, w1_best, w2_best)
    print "[test] correct", test_correct_rate
    print "[test] loss", test_loss
