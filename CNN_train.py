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
    drop_rate = 0.25
    early_stop_epochs = 20  # 提前中断轮数
    reduce_lr_epochs = 10
    learning_rate = 1e-4  # 初始学习率
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
    from keras.layers import Dense, Flatten, BatchNormalization, Activation, Dropout
    from keras.layers.convolutional import Conv2D, MaxPooling2D
    from keras.models import Sequential
    from keras import initializers
    from keras import regularizers

    x_train, x_test, y_train, y_test = data
    x_train = x_train.reshape((-1, param.loop_num, param.time_intervals, 1))
    y_train = y_train.reshape(-1, 1)
    x_test = x_test.reshape((-1, param.loop_num, param.time_intervals, 1))
    y_test = y_test.reshape(-1, 1)
    # def mape_error(y_true, y_pred):
    #     return K.mean(K.abs(y_pred - y_true)/y_true, axis=-1)
    # model=load_model('E:/LeNet/LeNet-5_model.h5')
    # convolution 1st layer
    model = Sequential()
    model.add(Conv2D(16, (3, 3), strides=(1, 1), input_shape=(param.loop_num, param.time_intervals, 1),
                     padding='same', activation='relu',
                     kernel_initializer=initializers.random_normal(stddev=0.1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # convolution 2nd layer
    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu',
                     kernel_initializer=initializers.random_normal(stddev=0.1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(param.regularization)))
    model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(param.regularization)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_absolute_percentage_error',
                  metrics=['mean_absolute_percentage_error', 'mean_absolute_error'])
    model.summary()
    history = LossHistory()
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=param.early_stop_epochs, mode='min')
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',  # 监控模型的验证损失
                                                  factor=0.5,  # 触发时将学习率除以5
                                                  patience=param.reduce_lr_epochs)
    model.fit(x_train, y_train, batch_size=param.batch_size, epochs=param.epochs, shuffle=True,
              validation_split=0.2, callbacks=[history, early_stop, reduce_lr], verbose=0)
    loss, mape, mae = model.evaluate(x_test, y_test, verbose=0)
    # model.save(os.path.join(param.file_path, 'model\\',
    #                         'model_ST={}_{}_mape={mape:.3f}_mae={mae:.3f}_time_lag{time}.h5'.format(
    #                             param.loop_num, param.time_intervals, mape=mape, mae=mae,
    #                             time=((1+param.predict_intervals)*5))))
    # history.loss_plot('epoch', param)
    return [mape, mae]


if __name__ == '__main__':
    import tensorflow as tf
    import keras.backend.tensorflow_backend as KTF
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    sess = tf.Session(config=config)
    KTF.set_session(sess)

    # time_start = time.time()
    # predict_loop = [96]  # 选取不同的检测线圈进行预测
    # time_lag = [x for x in range(4, 81, 4)]  # 预测时间变化
    # space = [x for x in range(8, 81, 4)]  # 预测使用的检测线圈数目
    # pred_intervals = [0]  # 预测的时间长度
    params = Parameters()

    params.time_intervals = 8  # 使用多长的时滞来预测
    params.loop_num = 16  # 使用多少空间点数
    params.predict_intervals = 0  # 预测多长的时间间隔0表示5分钟
    params.predict_loop = 86  # 对于道路的哪个节点进行预测

    data = train_test_data(params)
    mape, mae = train(data, params)

    # total = len(predict_loop) * len(time_lag) * len(space) * len(pred_intervals)
    # counter = 0
    # for cur_loop in predict_loop:
    #     params.predict_loop = cur_loop
    #     for cur_intervals in pred_intervals:
    #         params.predict_intervals = cur_intervals
    #         with open(os.path.join(params.file_path,
    #                                'model\\loop{}_res_error{}.csv'.format(params.predict_loop,
    #                                                                      (1 + params.predict_intervals) * 5)),
    #                   'w', newline='') as csvfile:
    #             fieldnames = ['pred_interval', 'space_time', 'MAPE', 'MAE']
    #             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #             writer.writeheader()
    #         for loop_num in space:
    #             for time_intervals in time_lag:
    #                 params.loop_num = loop_num
    #                 params.time_intervals = time_intervals
    #                 Data = train_test_data(params)
    #                 mape, mae = train(Data, params)
    #                 with open(os.path.join(params.file_path,
    #                                        'model\\loop{}_res_error{}.csv'.format(params.predict_loop,
    #                                                                              (1+params.predict_intervals)*5)),
    #                           'a+', newline='') as csvfile:
    #                     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #                     time_space = 'space{}*time{}'.format(params.loop_num, params.time_intervals)
    #                     writer.writerow({'pred_interval': (1+params.predict_intervals)*5,
    #                                      'space_time': time_space, 'MAPE': mape, 'MAE': mae})
    #                 counter += 1
    #                 print('COMPLETED {val:.2%}'.format(val=(counter/total)))
    # time_end = time.time()
    # print(time_end-time_start)
