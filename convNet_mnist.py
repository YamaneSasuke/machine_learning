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
class MLP(Chain):
    def __init__(self, num_features, dim_hidden, num_classes):
        super(MLP, self).__init__(
            conv1=L.Convolution2D(1, 20, 5),
            conv2=L.Convolution2D(20, 50, 5),
            l1=L.Linear(800, 500),
            l2=L.Linear(500, 10),
        )

    def loss_and_accuracy(self, X, T):
        x = Variable(X.reshape(-1, 1, 28, 28))
        t = Variable(T)
        a = self.conv1(x)
        a_z = F.tanh(a)
        z = F.max_pooling_2d(a_z, 2)
        a = self.conv2(z)
        a_z = F.tanh(a)
        z = F.max_pooling_2d(a_z, 2)
        a_y = self.l1(z)
        a_y = self.l2(a_y)
        return F.softmax_cross_entropy(a_y, t), F.accuracy(a_y, t)


def loss_ave_and_accuracy_ave(model, X, T, num_data, num_batches):
    accuracies = []
    losses = []
    total_data = np.arange(num_data)
    for indexes in np.array_split(total_data, num_batches):
        X_batch = cuda.to_gpu(X[indexes])
        T_batch = cuda.to_gpu(T[indexes])
        loss, accuracy = model.loss_and_accuracy(X_batch, T_batch)
        numpy_accuracy = cuda.to_cpu(accuracy.data)
        numpy_loss = cuda.to_cpu(loss.data)
        accuracies.append(numpy_accuracy)
        losses.append(numpy_loss)
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
    num_valid = len(X_valid)
    num_test = len(X_test)
    num_classes = len(np.unique(T_train))  # クラス数を格納
    num_features = X_train.shape[1]  # 特徴数を格納

    X_train_gpu = cuda.to_gpu(X_train)
    T_train_gpu = cuda.to_gpu(T_train)
    X_valid_gpu = cuda.to_gpu(X_valid)
    T_valid_gpu = cuda.to_gpu(T_valid)

    # 超パラメータ
    max_iteration = 300  # 繰り返し回数
    batch_size = 100  # ミニバッチサイズ
    dim_hidden = 400  # 隠れ層の次元数

    # modelの定義
    model = MLP(num_features, dim_hidden, num_classes).to_gpu()

    # Optimizerの設定
    optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)  # Momentum法
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
                loss, accuracy = model.loss_and_accuracy(x_batch, t_batch)
                # 逆伝搬を計算
                loss.backward()
                optimizer.weight_decay(0.001)  # L2正則化を実行
                optimizer.update()

            time_end = time.time()
            epoch_time = time_end - time_begin
            total_time = time_end - time_origin

            # trainデータで損失と精度を計算
            accuracy_trains, loss_trains = loss_ave_and_accuracy_ave(
                    model, X_train, T_train, num_train, num_batches)
            accuracy_trains_history.append(accuracy_trains)
            loss_trains_history.append(loss_trains)
            # validデータで損失と精度を計算
            accuracy_valids, loss_valids = loss_ave_and_accuracy_ave(
                    model, X_valid, T_valid, num_valid, num_batches)
            accuracy_valids_history.append(accuracy_valids)
            loss_valids_history.append(loss_valids)

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
            model_best, X_test, T_test, num_test, num_batches)
    print "[test] accuracy:", accuracy_test
    print "[test] loss:", loss_test
    print "max_iteration:", max_iteration
    print "batch_size:", batch_size
    print "dim_hidden:", dim_hidden