# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   CNN_train.py
@Time    :   2018/12/12 20:18
@Desc    :
"""

import tensorflow
import os
import keras
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

np.random.seed(48)

# 写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.val_loss['batch'].append(logs.get('val_loss'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.val_loss['epoch'].append(logs.get('val_loss'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.legend(loc="upper right")
        plt.show()


class Parameters:
    """
    网络的配置超参数
    """
    batch_size = 128
    regularization = 1e-3
    loop_num = 2            # 预测使用的空间节点个数
    time_intervals = 8      # 预测使用的时间滞后个数
    epochs = 5
    early_stop_epochs = 20  # 提前中断轮数
    learning_rate = 1e-4    # 初始学习率
    predict_intervals = 0   # 0,1,2,3分别表示5-20分钟预测
    file_path = r'DATA\data_all.csv'
    model_save_path = r'E:\yyh_result_CNN'


def train_test_data(param):
    """
    生成训练和测试数据
    :param param: 数据配置参数
    :param start: 检测线圈的起始
    :param end: 检测线圈的结束
    :return: x_train, x_test, y_train, y_test
    """
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    # 96表示的是159.57号检测线圈的数据
    select_loop = [x for x in range(96 - param.loop_num // 2, 96 + param.loop_num // 2)]
    def data_pro(data, time_steps=None):
        """
        数据处理，将列状的数据拓展开为行形式

        :param data: 输入交通数据
        :param time_steps: 分割时间长度
        :return: 处理过的数据
        :param slide_sep: 是否连续切割数据
        """
        if time_steps is None:
            time_steps = 1
        size = data.shape
        data = np.array(data)
        temp = np.zeros((size[0] - time_steps + 1, size[1] * time_steps))
        for i in range(data.shape[0] - time_steps + 1):
            temp[i, :] = data[i:i + time_steps, :].flatten()
        return temp
    data = pd.read_csv(param.file_path)
    label = np.array(data.iloc[param.time_intervals :,96]).reshape(-1, 1)
    data = data.iloc[:, select_loop]
    data = data_pro(data, time_steps=param.time_intervals)
    data = data[: -1]
    return train_test_split(data, label, test_size=0.2, shuffle=False)


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
    x_train = x_train.reshape((-1, param.loop_num, param.time_intervals, 1))
    y_train = y_train.reshape(-1, 1)
    x_test = x_test.reshape((-1, param.loop_num, param.time_intervals, 1))
    y_test = y_test.reshape(-1, 1)
    # def mape_error(y_true, y_pred):
    #     return K.mean(K.abs(y_pred - y_true)/y_true, axis=-1)
    # model=load_model('E:/LeNet/LeNet-5_model.h5')
    model = Sequential()
    model.add(Conv2D(16, (3, 3), strides=(1, 1), input_shape=(param.loop_num, param.time_intervals, 1),
                     padding='same', activation='relu',
                     kernel_initializer=initializers.random_normal(stddev=0.1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
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
                                                  factor=0.5,  # 触发时将学习率除以10
                                                  patience=10)  # 如果验证损失在10轮内都没有改善，那么就触发这个回调函数
    model.fit(x_train, y_train, batch_size=param.batch_size, epochs=param.epochs, shuffle=True,
              validation_split=0.2, callbacks=[history, early_stop, reduce_lr], verbose=2)
    model.save('E:/LeNet/LeNet-5_model-{epoch:02d}.h5')
    # [0.10342620456655367 0.9834000068902969]
    loss, mape, mae= model.evaluate(x_test, y_test, verbose=2)
    print(loss, mape, mae)
    history.loss_plot('epoch')


if __name__ == '__main__':
    params = Parameters()
    Data = train_test_data(params)
    train(Data, params)
