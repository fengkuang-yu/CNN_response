import tensorflow as tf
from mnist_final_version.hyper_parameter import *


flags = tf.flags
logging = tf.logging
flags.DEFINE_string("model", "lstm", "A type of model. Possible options are: ann, cnn, lstm.")
flags.DEFINE_string("data_path", 'Data_sets/MNIST_data', "data_path")
FLAGS = flags.FLAGS


def get_config():
    """
    部署神经网络，根据参数确定神经网络类型，匹配对应超参数
    :return: 参数类
    """
    if FLAGS.model == "ann":
        return AnnConfig()
    elif FLAGS.model == "cnn":
        return CnnConfig()
    elif FLAGS.model == "lstm":
        return LstmConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)


config = get_config()


def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def get_bais_variable(shape):
    biases = tf.get_variable("biases", shape, initializer=tf.constant_initializer(0.1))
    return biases


def cnn_inference(input_tensor, train, regularizer):
    """
    卷积神经网络的网络接口
    :param input_tensor: 输入的tensor数据
    :param train: 是否是训练模式，主要是影响到dropout
    :param regularizer: 是否使用正则化
    :return: 返回神经网络的输出结果
    """
    input_tensor_image = tf.reshape(input_tensor, shape=(-1,
                                                         config.IMAGE_SIZE,
                                                         config.IMAGE_SIZE,
                                                         config.NUM_CHANNELS))
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = get_weight_variable(
            [config.CONV1_SIZE, config.CONV1_SIZE, config.NUM_CHANNELS, config.CONV1_DEEP], regularizer=None)
        conv1_biases = get_bais_variable([config.CONV1_DEEP])
        conv1 = tf.nn.conv2d(input_tensor_image, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    with tf.variable_scope("layer3-conv2"):
        conv2_weights = get_weight_variable(
            [config.CONV2_SIZE, config.CONV2_SIZE, config.CONV1_DEEP, config.CONV2_DEEP], regularizer=None)
        conv2_biases = get_bais_variable([config.CONV2_DEEP])
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool2, [-1, nodes])

    with tf.variable_scope('layer5-fc1'):
        fc1_weights = get_weight_variable([nodes, config.FC_SIZE], regularizer=regularizer)
        fc1_biases = get_bais_variable([config.FC_SIZE])
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc2'):
        fc2_weights = get_weight_variable([config.FC_SIZE, config.NUM_LABELS], regularizer=regularizer)
        fc2_biases = get_bais_variable([config.NUM_LABELS])
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit


def ann_inference(input_tensor, regularizer):
    """
    ANN的手写体识别程序
    :param input_tensor: 输入数据
    :param regularizer: 是否使用正则化
    :return: 神经网络计算结果
    """
    with tf.variable_scope('layer1'):
        weights1 = get_weight_variable([config.INPUT_NODE, config.LAYER1_NODE], regularizer=regularizer)
        biases1 = get_bais_variable([config.LAYER1_NODE])
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)

    with tf.variable_scope('layer2'):
        weights2 = get_weight_variable([config.LAYER1_NODE, config.OUTPUT_NODE], regularizer=regularizer)
        biases2 = get_bais_variable([config.OUTPUT_NODE])
        layer2 = tf.matmul(layer1, weights2) + biases2

    return layer2


def lstm_inference(input_tensor, regularizer):
    """
    长短时记忆神经网络接口
    :param input_tensor: 输入数据
    :param regularizer: 正则化
    :return: 返回神经网络计算结果
    """
    input_tensor_image = tf.reshape(input_tensor, [-1, config.IMAGE_SIZE, config.IMAGE_SIZE])
    weights = get_weight_variable([config.HIDDEN_NODE, config.OUTPUT_NODE], regularizer=regularizer)
    biases = get_bais_variable([config.OUTPUT_NODE])

    def lstm():
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(config.HIDDEN_NODE, forget_bias=1.0,
                                                    state_is_tuple=True,
                                                    reuse=tf.get_variable_scope().reuse)
        return lstm_fw_cell

    with tf.variable_scope(None, default_name="Rnn"):
        #    cell = tf.contrib.rnn.MultiRNNCell([cell1, cell2])
        cell = tf.contrib.rnn.MultiRNNCell([lstm() for _ in range(config.STACKED_LAYERS)], state_is_tuple=True)
        output, _ = tf.nn.dynamic_rnn(cell, input_tensor_image, dtype=tf.float32)
        output = tf.transpose(output, [1, 0, 2])
    logit = tf.matmul(output[-1], weights) + biases
    return logit




def res_h(n, val_set,lim, val = 0):
    if n == 0:
        return '0'
    for i in range(len(val_set)):
        i

