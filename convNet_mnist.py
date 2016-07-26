# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 16:07:08 2016

@author: yamane

ConvNetでMNIST分類器を作成
"""

import numpy as np
from chainer import cuda, Variable, optimizers, Chain
import chainer.functions as F
import chainer.links as L
import load_mnist
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import time
import copy


# ニューラルネットワークの定義
class Convnet(Chain):
    def __init__(self):
        super(Convnet, self).__init__(
            conv1=L.Convolution2D(1, 40, 5),
            conv2=L.Convolution2D(40, 100, 5),
            l1=L.Linear(1600, 500),
            l2=L.Linear(500, 10),
        )

    def loss_and_accuracy(self, X, T, train):
        x = Variable(X.reshape(-1, 1, 28, 28))
        t = Variable(T)
        h = self.conv1(x)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 2)
        h = self.conv2(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 2)
        h = F.dropout(h, train=train)
        h = self.l1(h)
        h = F.relu(h)
        h = F.dropout(h, train=train)
        y = self.l2(h)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)


def loss_ave_and_accuracy_ave(model, X, T, num_batches, train):
    accuracies = []
    losses = []
    total_data = np.arange(len(X))
    for indexes in np.array_split(total_data, num_batches):
        X_batch = cuda.to_gpu(X[indexes])
        T_batch = cuda.to_gpu(T[indexes])
        loss, accuracy = model.loss_and_accuracy(X_batch, T_batch, train)
        accuracy_cpu = cuda.to_cpu(accuracy.data)
        loss_cpu = cuda.to_cpu(loss.data)
        accuracies.append(accuracy_cpu)
        losses.append(loss_cpu)
    return np.mean(accuracies), np.mean(losses)


if __name__ == '__main__':
    X_train, T_train, X_test, T_test = load_mnist.load_mnist()
    # データを0~1に変換
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    # 適切なdtypeに変換
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    T_train = T_train.astype(np.int32)
    T_test = T_test.astype(np.int32)
    # 訓練データを分割
    X_train, X_valid, T_train, T_valid = train_test_split(X_train,
                                                          T_train,
                                                          test_size=0.1,
                                                          random_state=10)
    num_train = len(X_train)

    X_train_gpu = cuda.to_gpu(X_train)
    T_train_gpu = cuda.to_gpu(T_train)
    X_valid_gpu = cuda.to_gpu(X_valid)
    T_valid_gpu = cuda.to_gpu(T_valid)

    # 超パラメータ
    max_iteration = 100  # 繰り返し回数
    batch_size = 100  # ミニバッチサイズ
    learning_rate = 0.00005  # 学習率
    momentum_rate = 0.9  # Momentum
    decay_rate = 0.001  # L2正則化

    model = Convnet().to_gpu()

    # Optimizerの設定
    optimizer = optimizers.Adam(learning_rate)
    optimizer.setup(model)

    num_batches = num_train / batch_size

    accuracy_trains_history = []  # グラフ描画用の配列
    accuracy_valids_history = []  # グラフ描画用の配列
    loss_trains_history = []  # グラフ描画用の配列
    loss_valids_history = []  # グラフ描画用の配列
    accuracy_best = 0

    time_origin = time.time()
    try:
        for epoch in range(max_iteration):
            time_begin = time.time()
            # 入力データXと正解ラベルを取り出す
            permu = np.random.permutation(num_train)
            for indexes in np.array_split(permu, num_batches):
                x_batch = cuda.to_gpu(X_train[indexes])
                t_batch = cuda.to_gpu(T_train[indexes])
                this_batch_size = len(indexes)
                # 勾配を初期化
                optimizer.zero_grads()
                # 順伝播を計算し、誤差と精度を取得
                loss, accuracy = model.loss_and_accuracy(x_batch,
                                                         t_batch,
                                                         True)
                # 逆伝搬を計算
                loss.backward()
#                optimizer.weight_decay(decay_rate)  # L2正則化を実行
                optimizer.update()

            time_end = time.time()
            epoch_time = time_end - time_begin
            total_time = time_end - time_origin

            # trainデータで損失と精度を計算
            accuracy_train, loss_train = loss_ave_and_accuracy_ave(
                    model, X_train, T_train, num_batches, False)
            accuracy_trains_history.append(accuracy_train)
            loss_trains_history.append(loss_train)
            # validデータで損失と精度を計算
            accuracy_valid, loss_valid = loss_ave_and_accuracy_ave(
                    model, X_valid, T_valid, num_batches, False)
            accuracy_valids_history.append(accuracy_valid)
            loss_valids_history.append(loss_valid)

            if accuracy_valids_history[epoch] > accuracy_best:
                accuracy_best = accuracy_valids_history[epoch]
                epoch_best = epoch
                model_best = copy.deepcopy(model)  # 最善のモデルを確保

            # 正解率、損失を表示
            print "epoch:", epoch
            print "time:", epoch_time, "(", total_time, ")"
            print "[train] accuracy:", accuracy_trains_history[epoch]
            print "[valid] accuracy:", accuracy_valids_history[epoch]
            print "[train] loss:", loss_trains_history[epoch]
            print "[valid] loss:", loss_valids_history[epoch]
            print "best_accuracy:", accuracy_best, "best_epoch", epoch_best
            print "|W1|:", np.linalg.norm(cuda.to_cpu(model.l1.W.data),
                                          axis=0).mean()
            print "|W2|:", np.linalg.norm(cuda.to_cpu(model.l2.W.data),
                                          axis=0).mean()

            plt.plot(accuracy_trains_history)
            plt.plot(accuracy_valids_history)
            plt.title("accuracy")
            plt.legend(["train", "valid"], loc="lower right")
            plt.grid()
            plt.show()

            plt.plot(loss_trains_history)
            plt.plot(loss_valids_history)
            plt.title("loss")
            plt.legend(["train", "valid"], loc="upper right")
            plt.grid()
            plt.show()
    except KeyboardInterrupt:
        print "割り込み停止が実行されました"

    # テストデータの結果を表示
    accuracy_test, loss_test = loss_ave_and_accuracy_ave(
            model_best, X_test, T_test, num_batches, False)
    print "[test] accuracy:", accuracy_test
    print "[test] loss:", loss_test
    print "max_iteration:", max_iteration
    print "batch_size:", batch_size
    print "learning_rate", learning_rate
    print "decay_rate", decay_rate
    print "momentum_rate", momentum_rate
