# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   ANN_train.py
@Time    :   2019/1/1 16:43
@Desc    :
"""

# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   CNN_train.py
@Time    :   2018/12/12 20:18
@Desc    :
"""

import time
import os
import csv
import keras
import numpy as np
import matplotlib.pylab as plt
from keras.layers import Dense
from keras.models import Sequential
from keras import regularizers

np.random.seed(48)


# 写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.val_loss['batch'].append(logs.get('val_loss'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.val_loss['epoch'].append(logs.get('val_loss'))

    def loss_plot(self, loss_type, param):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train')
        if loss_type == 'epoch':
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='validation')
        plt.grid(True)
        plt.xlabel(loss_type, fontsize=14)
        plt.ylabel('loss', fontsize=14)
        plt.legend(loc="upper right", fontsize=14)
        plt.savefig(os.path.join(param.file_path, 'figure\\',
                                 'loss_ST={}_{}_pred_time{}.pdf'.format(param.loop_num,
                                                                        param.time_intervals,
                                                                        5*(param.predict_intervals+1))))


class Parameters:
    """
    网络的配置超参数
    """
    batch_size = 128
    regularization = 1e-3
    loop_num = 4  # 预测使用的空间节点个数
    time_intervals = 4  # 预测使用的时间滞后个数
    epochs = 200
    early_stop_epochs = 20  # 提前中断轮数
    reduce_lr_epochs = 10
    learning_rate = 1e-3  # 初始学习率
    predict_intervals = 0  # 0,1,2,3分别表示5-20分钟预测
    predict_loop = 96  # 96表示159.57号检测线圈
    file_path = r'D:\Users\yyh\Pycharm_workspace\CNN_response_simulation'


def train_test_data(param):
    """
    生成训练和测试数据
    :param param: 数据配置参数
    :return: x_train, x_test, y_train, y_test
    """
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    # 96表示的是159.57号检测线圈的数据
    select_loop = [x for x in range(param.predict_loop - param.loop_num // 2,
                                    param.predict_loop + param.loop_num // 2)]

    def data_pro(data, time_steps=None):
        """
        数据处理，将列状的数据拓展开为行形式

        :param data: 输入交通数据
        :param time_steps: 分割时间长度
        :return: 处理过的数据
        """
        if time_steps is None:
            time_steps = 1
        size = data.shape
        data = np.array(data)
        temp = np.zeros((size[0] - time_steps + 1, size[1] * time_steps))
        for i in range(data.shape[0] - time_steps + 1):
            temp[i, :] = data[i:i + time_steps, :].flatten()
        return temp

    data = pd.read_csv(os.path.join(param.file_path, 'data\\' 'data_all.csv'))
    label = np.array(data.iloc[param.time_intervals + param.predict_intervals:, param.predict_loop]).reshape(-1, 1)
    data = data.iloc[:, select_loop]
    data = data_pro(data, time_steps=param.time_intervals)
    data = data[: -(1 + param.predict_intervals)]
    return train_test_split(data, label, test_size=0.2, shuffle=True, random_state=42)


def train(data, param):
    """
    对输入神经网络的数据进行训练和评估
    :param data: 输入训练和测试数据
    :param param: 网络配置的参数
    :return: 返回神经网络的测试数据集表现
    """
    x_train, x_test, y_train, y_test = data
    x_train = x_train.reshape((-1, param.loop_num * param.time_intervals))
    y_train = y_train.reshape(-1, 1)
    x_test = x_test.reshape((-1, param.loop_num * param.time_intervals))
    y_test = y_test.reshape(-1, 1)
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=param.loop_num * param.time_intervals,
                    kernel_regularizer=regularizers.l2(param.regularization)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error',
                  metrics=['mean_absolute_percentage_error', 'mean_absolute_error'])
    model.summary()
    history = LossHistory()
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=param.early_stop_epochs, mode='min')
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',  # 监控模型的验证损失
                                                  factor=0.3,  # 触发时将学习率乘以0.3
                                                  patience=param.reduce_lr_epochs)
    model.fit(x_train, y_train, batch_size=param.batch_size, epochs=param.epochs, shuffle=True,
              validation_split=0.2, callbacks=[history, early_stop, reduce_lr], verbose=2)
    loss, mape, mae = model.evaluate(x_test, y_test, verbose=0)
    return [mape, mae]


if __name__ == '__main__':
    import tensorflow as tf
    import keras.backend.tensorflow_backend as KTF

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    sess = tf.Session(config=config)
    KTF.set_session(sess)
    params = Parameters()

    params.time_intervals = 4  # 使用多长的时滞来预测
    params.loop_num = 28  # 使用多少空间点数
    params.predict_intervals = 1  # 预测多长的时间间隔0表示5分钟
    params.predict_loop = 94  # 对于道路的哪个节点进行预测

    data = train_test_data(params)
    mape, mae = train(data, params)


