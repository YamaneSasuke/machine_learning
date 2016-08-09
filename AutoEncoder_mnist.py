# -*- coding: utf-8 -*-
"""
Created on Mon Aug 08 13:15:52 2016

@author: yamane
"""

import numpy as np
from chainer import cuda, Variable, optimizers, Chain
import chainer.functions as F
import chainer.links as L
import load_mnist
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt


# ニューラルネットワークの定義
class Autoencoder(Chain):
    def __init__(self, num_features, dim_hidden):
        super(Autoencoder, self).__init__(
            encode=L.Linear(num_features, dim_hidden),
            decode=L.Linear(dim_hidden, num_features),
        )

    def loss_and_output(self, X, T):
        x = Variable(X)
        t = Variable(X)
        h = self.encode(x)
        h = F.relu(h)
        h = self.decode(h)
        y = F.relu(h)
        return F.mean_squared_error(y, t), y


def plot_data(model, X, T, num_batches):
    total_data = np.random.permutation(len(X))
    for indexes in np.array_split(total_data, num_batches):
        X_batch = cuda.to_gpu(X[indexes])
        T_batch = cuda.to_gpu(T[indexes])
        loss, y = model.loss_and_output(X_batch, T_batch)
        y_cpu = cuda.to_cpu(y.data)
        print T[indexes]
        plt.matshow(y_cpu.reshape(28, 28), cmap=plt.cm.gray)
        plt.show()


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
    X_train = X_train[0:1000]
    T_train = T_train[0:1000]
    num_train = len(X_train)
    num_features = X_train.shape[1]

    X_train_gpu = cuda.to_gpu(X_train)
    T_train_gpu = cuda.to_gpu(T_train)
    X_valid_gpu = cuda.to_gpu(X_valid)
    T_valid_gpu = cuda.to_gpu(T_valid)

    # 超パラメータ
    max_iteration = 100  # 繰り返し回数
    batch_size = 1  # ミニバッチサイズ
    learning_rate = 0.00005  # 学習率
    dim_hidden = 400  # 隠れ層の次元数

    model = Autoencoder(num_features, dim_hidden).to_gpu()

    # Optimizerの設定
    optimizer = optimizers.Adam(learning_rate)
    optimizer.setup(model)

    num_batches = num_train / batch_size

    try:
        for epoch in range(max_iteration):
            # 入力データXと正解ラベルを取り出す
            permu = np.random.permutation(num_train)
            for indexes in np.array_split(permu, num_batches):
                x_batch = cuda.to_gpu(X_train[indexes])
                t_batch = cuda.to_gpu(T_train[indexes])
                this_batch_size = len(indexes)
                # 勾配を初期化
                optimizer.zero_grads()
                # 順伝播を計算し、誤差と精度を取得
                loss, y = model.loss_and_output(x_batch, t_batch)
                # 逆伝搬を計算
                loss.backward()
                optimizer.update()

        # 訓練データでの結果を表示
        plot_data(model, X_train, T_train, num_batches)

    except KeyboardInterrupt:
        print "割り込み停止が実行されました"

    # テストデータでの結果を表示
    num_batches = len(X_test) / batch_size
    permu = np.random.permutation(len(X_test))
    plot_data(model, X_test, T_test, num_batches)
