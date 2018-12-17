# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   hyper_parameter.py
@Time    :   2018/9/28 15:16
@Desc    :
"""


class AnnConfig(object):
    INPUT_NODE = 784
    OUTPUT_NODE = 10

    LAYER1_NODE = 500
    BATCH_SIZE = 100
    LEARNING_RATE_BASE = 0.01
    LEARNING_RATE_DECAY = 0.99
    REGULARIZATION_RATE = 0.0001
    TRAINING_STEPS = 1000
    DISP_PER_TIMES = 100
    MOVING_AVERAGE_DECAY = 0.99
    MODEL_SAVE_PATH = "MNIST_model/ann"
    MODEL_NAME = "mnist_model"


# 对于CNN
class CnnConfig(object):
    INPUT_NODE = 784
    OUTPUT_NODE = 10

    IMAGE_SIZE = 28
    NUM_CHANNELS = 1
    NUM_LABELS = 10
    CONV1_DEEP = 32
    CONV1_SIZE = 5
    CONV2_DEEP = 64
    CONV2_SIZE = 5
    FC_SIZE = 512

    BATCH_SIZE = 50
    LEARNING_RATE_BASE = 0.01
    LEARNING_RATE_DECAY = 0.99
    REGULARIZATION_RATE = 0.0001
    TRAINING_STEPS = 1000
    DISP_PER_TIMES = 100
    MOVING_AVERAGE_DECAY = 0.99
    MODEL_SAVE_PATH = "MNIST_model/cnn"
    MODEL_NAME = "mnist_model"


# 对于LSTM
class LstmConfig(object):
    INPUT_NODE = 784
    IMAGE_SIZE = 28
    OUTPUT_NODE = 10
    HIDDEN_NODE = 128
    STACKED_LAYERS = 1
    BATCH_SIZE = 100
    LEARNING_RATE_BASE = 0.01
    LEARNING_RATE_DECAY = 0.99
    REGULARIZATION_RATE = 0.0001
    TRAINING_STEPS = 1000
    DISP_PER_TIMES = 100
    MOVING_AVERAGE_DECAY = 0.99
    MODEL_SAVE_PATH = "MNIST_model/lstm"
    MODEL_NAME = "mnist_model"


if __name__ == '__main__':
    pass
