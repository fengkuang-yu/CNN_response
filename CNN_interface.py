# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   CNN_interface.py
@Time    :   2018/12/12 20:01
@Desc    :
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import time


def data_generate(spatial, temporal):
    """
    根据输入的spatial和temporal产生神经网络的输入数据
    :param spacial: 空间相关点数
    :param temporal: 时间相关点数
    :return: 输入数据+标签
    """
    allRoadData = pd.read_csv(r'DATA\data_all.csv')

    pass


def cnn_interface():
    from keras.datasets import mnist
    from keras.layers import Dense, Flatten
    from keras.layers.convolutional import Conv2D, MaxPooling2D
    from keras.models import Sequential
    from keras.utils.np_utils import to_categorical

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((-1, 28, 28, 1))
    y_train = to_categorical(y_train, 10)
    x_test = x_test.reshape((-1, 28, 28, 1))
    y_test = to_categorical(y_test, 10)

    # model=load_model('E:/LeNet/LeNet-5_model.h5')
    model = Sequential()
    model.add(Conv2D(6, (5, 5), strides=(1, 1), input_shape=(28, 28, 1), padding='valid', activation='relu',
                     kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (5, 5), strides=(1, 1), padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(x_train, y_train, batch_size=100, epochs=1, shuffle=True)
    model.save('E:/LeNet/LeNet-5_model.h5')
    # [0.10342620456655367 0.9834000068902969]
    loss, accuracy = model.evaluate(x_test, y_test, batch_size=100)
    print(loss, accuracy)


if __name__ == '__main__':
    pass
