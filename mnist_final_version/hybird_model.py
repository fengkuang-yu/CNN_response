# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   hybird_modelhybird_model.py
@Time    :   2018/9/30 9:20
@Desc    :
"""


import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from mnist_final_version import mnist_all_infernece
from mnist_final_version.mnist_all_infernece import config, FLAGS


# 定义训练过程
def train(mnist):
    x = tf.placeholder(tf.float32, [None, config.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, config.OUTPUT_NODE], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(config.REGULARIZATION_RATE)

    # 定义输出数据的placeholder
    if FLAGS.model is 'cnn':
        y = mnist_all_infernece.cnn_inference(x, False, regularizer)
        model_save_path = config.MODEL_SAVE_PATH_CNN
    elif FLAGS.model is 'ann':
        y = mnist_all_infernece.ann_inference(x, regularizer)
        model_save_path = config.MODEL_SAVE_PATH_ANN
    elif FLAGS.model is 'lstm':
        y = mnist_all_infernece.lstm_inference(x, regularizer)
        model_save_path = config.MODEL_SAVE_PATH_LSTM
    else:
        raise Exception('FLAGS.model错误')

    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(config.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        config.LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / config.BATCH_SIZE, config.LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(config.TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(config.BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: xs, y_: ys})

            if i % config.DISP_PER_TIMES == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
        saver.save(sess, os.path.join(model_save_path, config.MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets(FLAGS.data_path, one_hot=True)
    train(mnist)
    tf.reset_default_graph()


if __name__ == '__main__':
    if not FLAGS.data_path:
        raise ValueError("设定的文件路径有误")
    print(FLAGS.data_path)
    config = mnist_all_infernece.get_config()
    main()
