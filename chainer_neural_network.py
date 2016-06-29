# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 15:54:21 2016

@author: yamane
"""

import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers, utils
import chainer.functions as F
import load_mnist
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import time


# 順伝播の計算
def forward(model, X, T, train=True):
    x = Variable(X)
    t = Variable(T)
    h1 = F.dropout(F.relu(model.l1(x)), train=train)
    h2 = F.dropout(F.relu(model.l2(h1)), train=train)
    y = model.l3(h2)

    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)


if __name__ == '__main__':
    x_train, t_train, x_test, t_test = load_mnist.load_mnist()
    # データを0~1に変換
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    # 適切なdtypeに変換
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    t_train = t_train.astype(np.int32)
    t_test = t_test.astype(np.int32)
    # 訓練データを分割
    X_train, X_valid, T_train, T_valid = train_test_split(x_train,
                                                          t_train,
                                                          test_size=0.1,
                                                          random_state=10)
    num_train = len(X_train)
    num_classes = len(np.unique(T_train))
    num_features = X_train.shape[1]

    # 超パラメータ
    max_iteration = 300  # 繰り返し回数
    batch_size = 100  # ミニバッチサイズ
    dim_hidden = 400  # 隠れ層の次元数
    # modelの定義
    model = FunctionSet(l1=F.Linear(num_features, dim_hidden),
                        l2=F.Linear(dim_hidden, dim_hidden),
                        l3=F.Linear(dim_hidden, num_classes))

    # Optimizerの設定
    optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)  # Momentum法
    optimizer.setup(model)

    num_batches = num_train / batch_size

    accuracy_trains = []  # グラフ描画用の配列
    accuracy_valids = []  # グラフ描画用の配列
    loss_trains = []  # グラフ描画用の配列
    loss_valids = []  # グラフ描画用の配列
    accuracy_best = 0

    time_origin = time.time()
    for epoch in range(max_iteration):
        time_begin = time.time()
        # 入力データXと正解ラベルを取り出す
        permu = np.random.permutation(num_train)
        for indexes in np.array_split(permu, num_batches):
            x_batch = X_train[indexes]
            t_batch = T_train[indexes]
            this_batch_size = len(indexes)
            # 勾配を初期化
            optimizer.zero_grads()
            # 順伝播を計算し、誤差と精度を取得
            loss, accuracy = forward(model, x_batch, t_batch)
            # 逆伝搬を計算
            loss.backward()
            optimizer.weight_decay(0.001)  # L2正則化を実行
            optimizer.update()

        time_end = time.time()
        loss_train, accuracy_train = forward(model, X_train, T_train)
        loss_valid, accuracy_valid = forward(model, X_valid, T_valid)

        if accuracy_valid.data > accuracy_best:
            accuracy_best = accuracy_valid.data
            epoch_best = epoch
        accuracy_trains.append(accuracy_train.data)
        accuracy_valids.append(accuracy_valid.data)
        loss_trains.append(loss_train.data)
        loss_valids.append(loss_valid.data)

        # 正解率、損失を表示
        print "epoch:", epoch
        print "time:", time_end - time_begin, "(", time_end - time_origin, ")"
        print "[train] accuracy:", accuracy_train.data
        print "[valid] accuracy:", accuracy_valid.data
        print "[train] loss:", loss_train.data
        print "[valid] loss:", loss_valid.data
        print "best_accuracy:", accuracy_best, "best_epoch", epoch_best
        print "[model1] W:", np.linalg.norm(model.l1.W, axis=0).mean()
        print "[model2] W:", np.linalg.norm(model.l2.W, axis=0).mean()

        plt.plot(accuracy_trains)
        plt.plot(accuracy_valids)
        plt.title("accuracy")
        plt.legend(["train", "valid"], loc="lower right")
        plt.grid()
        plt.show()

        plt.plot(loss_trains)
        plt.plot(loss_valids)
        plt.title("loss")
        plt.legend(["train", "valid"], loc="upper right")
        plt.grid()
        plt.show()

    # テストデータの結果を表示
    loss_test, accuracy_test = forward(model, x_test, t_test)
    print "[test] accuracy:", accuracy_test.data
    print "[test] loss:", loss_test.data
    print "max_iteration:", max_iteration
    print "batch_size:", batch_size
    print "dim_hidden:", dim_hidden
