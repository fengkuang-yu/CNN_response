# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   LeetCode.py
@Time    :   2018/11/26 18:52
@Desc    :
"""
import keras
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

np.random.seed(48)

#写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


class Parameters:
    """
    网络的配置超参数
    """
    batch_size = 50
    regularization = 0.0001
    data_size = 20
    time_intervals = 8
    epochs = 5
    learning_rate = 1e-3
    predict_intervals = 0  # 0 denotes 5mins


def train_test_data(param):
    """
    生成训练和测试数据
    :param param: 数据配置参数
    :return: 训练和测试数据集
    """
    data = pd.read_excel('DATA/input_data.xlsx', Sheetname='Sheet1', header=None)
    labels = pd.read_excel('DATA/input_labels.xlsx', Sheetname='Sheet1', header=None)
    test_data = pd.read_excel('DATA/test_data_10days.xlsx', Sheetname='Sheet1', header=None)
    test_labels = pd.read_excel('DATA/test_data_labels_10days.xlsx', Sheetname='Sheet1', header=None)

    # 将数据处理为所需类型
    input_data = data.as_matrix()
    imagesize = param.data_size * param.time_intervals
    column_num = len(input_data)
    sample_num = column_num - param.time_intervals

    # 生成训练样本
    input_data1 = np.zeros((sample_num, imagesize))
    for i in range(sample_num):
        temp = input_data[i:i + param.time_intervals, :]
        input_data1[i, :] = temp.reshape(1, imagesize)

    # 输入训练样本labels
    input_data2 = labels.as_matrix()[8:]

    # 输入10天的测试数据
    test_data_10days = test_data.as_matrix()
    test_sample_num = len(test_data_10days) - param.time_intervals - param.predict_intervals
    input_data3 = np.zeros((test_sample_num, imagesize))
    for i in range(test_sample_num):
        temp = test_data_10days[i:i + param.time_intervals, :]
        input_data3[i, :] = temp.reshape(1, imagesize)

    # 输入10天的测试数据的标签
    input_data4 = test_labels.as_matrix()[8:]
    return [input_data1, input_data2, input_data3, input_data4]


def train(data, param):
    """
    对输入神经网络的数据进行训练和评估
    :param data: 输入训练和测试数据
    :param param: 网络配置的参数
    :return: 返回神经网络的测试数据集表现
    """
    from keras.layers import Dense, Flatten
    from keras.layers.convolutional import Conv2D, MaxPooling2D
    from keras.models import Sequential
    from keras import initializers
    from keras import backend as K
    from keras import regularizers
    x_train, y_train, x_test, y_test = data
    x_train = x_train.reshape((-1, 20, 8, 1))
    y_train = y_train.reshape(-1, 1)
    x_test = x_test.reshape((-1, 20, 8, 1))
    y_test = y_test.reshape(-1, 1)
    # def mape_error(y_true, y_pred):
    #     return K.mean(K.abs(y_pred - y_true)/y_true, axis=-1)
    # model=load_model('E:/LeNet/LeNet-5_model.h5')
    model = Sequential()
    model.add(Conv2D(16, (3, 3), strides=(1, 1), input_shape=(20, 8, 1), padding='same', activation='relu',
                     kernel_initializer=initializers.random_normal(stddev=0.1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu',
                     kernel_initializer=initializers.random_normal(stddev=0.1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(320, activation='relu', kernel_regularizer=regularizers.l2(param.regularization)))
    model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(param.regularization)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_absolute_percentage_error', metrics=['mean_absolute_percentage_error'])
    model.summary()
    history = LossHistory()
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, mode='auto')
    model.fit(x_train, y_train, batch_size=param.batch_size, epochs=param.epochs, shuffle=True,
              validation_split=0.2, callbacks=[history, early_stop], verbose=2)
    model.save('E:/LeNet/LeNet-5_model-{epoch:02d}.h5')
    # [0.10342620456655367 0.9834000068902969]
    loss, accuracy = model.evaluate(x_test, y_test, batch_size=50)

    print(loss, accuracy)
    history.loss_plot('epoch')


if __name__ == '__main__':
    params = Parameters()
    Data = train_test_data(params)
    train(Data, params)

