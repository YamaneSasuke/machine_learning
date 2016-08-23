# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 17:31:09 2016

@author: yamane
"""

import numpy as np
from chainer import cuda, Variable, optimizers, Chain
import chainer.functions as F
import chainer.links as L
import load_mnist
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import time


# ネットワークの定義
class ConvnetDeconvnet(Chain):
    def __init__(self):
        super(ConvnetDeconvnet, self).__init__(
            conv1=L.Convolution2D(1, 15, 3, pad=1),
            conv2=L.Convolution2D(15, 30, 3, pad=1),
            deconv1=L.Deconvolution2D(30, 15, 3, pad=1),
            deconv2=L.Deconvolution2D(15, 1, 3, pad=1)
        )

    def forward(self, X):
        x = Variable(X.reshape(-1, 1, 28, 28))
        h = self.conv1(x)
        h1 = F.relu(h)
        h = F.max_pooling_2d(h1, 2,)
        h = self.conv2(h)
        h2 = F.relu(h)
        h = F.max_pooling_2d(h2, 2,)
        h = F.unpooling_2d(h, 2, outsize=h2.data.shape[2:])
        h = self.deconv1(h)
        h = F.relu(h)
        h = F.unpooling_2d(h, 2, outsize=h1.data.shape[2:])
        h = self.deconv2(h)
        y = F.sigmoid(h)
        return y

    def lossfun(self, X, T):
        t = Variable(T.reshape(-1, 1, 28, 28))
        y = self.forward(X)
        loss = F.mean_squared_error(y, t)
        return loss

    def loss_and_output(self, X, T):
        y = cuda.to_cpu(self.forward(X).data)
        loss = cuda.to_cpu(self.lossfun(X, T).data)
        return loss, y

    def predict(self, X):
        return cuda.to_cpu(self.forward(X).data)


def draw_filters(W, cols=20, fig_size=(10, 10), filter_shape=(28, 28),
                 filter_standardization=False):
    border = 2
    num_filters = len(W)
    rows = int(np.ceil(float(num_filters) / cols))
    filter_height, filter_width = filter_shape

    if filter_standardization:
        W = preprocessing.scale(W, axis=1)
    image_shape = (rows * filter_height + (border * rows),
                   cols * filter_width + (border * cols))
    low, high = W.min(), W.max()
    low = (3 * low + high) / 4
    high = (low + 3 * high) / 4
    all_filter_image = np.random.uniform(low=low, high=high,
                                         size=image_shape)
    all_filter_image = np.full(image_shape, W.min(), dtype=np.float32)

    for i, w in enumerate(W):
        start_row = (filter_height * (i / cols) +
                     (i / cols + 1) * border)
        end_row = start_row + filter_height
        start_col = (filter_width * (i % cols) +
                     (i % cols + 1) * border)
        end_col = start_col + filter_width
        all_filter_image[start_row:end_row, start_col:end_col] = \
            w.reshape(filter_shape)

    plt.figure(figsize=fig_size)
    plt.imshow(all_filter_image, cmap=plt.cm.gray,
               interpolation='none')
    plt.tick_params(axis='both',  labelbottom='off',  labelleft='off')
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
    X_test = np.random.permutation(X_test)
    T_test = np.random.permutation(T_test)
    X_train = X_train[0:1000]
    T_train = T_train[0:1000]
    X_valid = X_valid[0:100]
    T_valid = T_valid[0:100]
    X_test = X_test[0:100]
    T_test = T_test[0:100]
    X_train_gpu = cuda.to_gpu(X_train)
    T_train_gpu = X_train_gpu
    X_valid_gpu = cuda.to_gpu(X_valid)
    T_valid_gpu = X_valid_gpu
    X_test_gpu = cuda.to_gpu(X_test)
    T_test_gpu = X_test_gpu

    num_features = X_train.shape[1]

    # 超パラメータ
    max_iteration = 200  # 繰り返し回数
    batch_size = 100  # ミニバッチサイズ
    learning_rate = 0.001  # 学習率
#    decay_rate = 0.001

    model = ConvnetDeconvnet().to_gpu()

    # Optimizerの設定
    optimizer = optimizers.Adam(learning_rate)
    optimizer.setup(model)

    num_batches = len(X_train) / batch_size
    loss_trains_history = []  # グラフ描画用の配列
    loss_valids_history = []  # グラフ描画用の配列

    time_origin = time.time()
    try:
        for epoch in range(max_iteration):
            time_begin = time.time()
            # 入力データXと正解ラベルを取り出す
            permu = np.random.permutation(len(X_train))
            for indexes in np.array_split(permu, num_batches):
                x_batch = cuda.to_gpu(X_train[indexes])
                t_batch = x_batch
                this_batch_size = len(indexes)
                # 勾配を初期化
                optimizer.zero_grads()
                # 順伝播を計算し、誤差と精度を取得
                loss = model.lossfun(x_batch, t_batch)
                # 逆伝搬を計算
                loss.backward()
#                optimizer.weight_decay(decay_rate)  # L2正則化を実行
                optimizer.update()

            time_end = time.time()
            epoch_time = time_end - time_begin
            total_time = time_end - time_origin

            loss_train, y_train = model.loss_and_output(X_train_gpu,
                                                        T_train_gpu)
            loss_valid, y_valid = model.loss_and_output(X_valid_gpu,
                                                        T_valid_gpu)
            loss_trains_history.append(loss_train)
            loss_valids_history.append(loss_valid)
            # 訓練データでの結果を表示
            print "epoch:", epoch
            print "time:", epoch_time, "(", total_time, ")"
            print "[train] loss:", loss_trains_history[epoch]
            print "[valid] loss:", loss_valids_history[epoch]
            plt.plot(loss_trains_history)
            plt.plot(loss_valids_history)
            plt.title("loss")
            plt.legend(["train", "valid"], loc="upper right")
            plt.grid()
            plt.show()
            i = np.random.choice(len(X_train_gpu))
            print "画像の数字:", "[", T_train[i], "]"
            plt.matshow(X_train[i].reshape(28, 28), cmap=plt.cm.gray)
            plt.show()
            plt.matshow(y_train[i].reshape(28, 28), cmap=plt.cm.gray)
            plt.show()

    except KeyboardInterrupt:
        print "割り込み停止が実行されました"

    # テストデータでの結果を表示
    y_test = model.predict(X_test_gpu)
    loss_test = model.lossfun(X_test_gpu, T_test_gpu)
    print "[test] loss:", loss_test
    print "max_iteration:", max_iteration
    print "batch_size:", batch_size
    print "learning_rate", learning_rate
    print "X_test"
    draw_filters(X_test, 10)
    print "predict_test"
    draw_filters(y_test, 10)
