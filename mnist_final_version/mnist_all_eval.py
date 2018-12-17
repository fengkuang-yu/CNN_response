# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 16:02:37 2018

@author: yyh
"""

from tensorflow.examples.tutorials.mnist import input_data

from mnist_final_version.mnist_all_infernece import *
from mnist_final_version.mnist_all_infernece import config, FLAGS

# 每10秒加载一次最新的模型
EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    """
    评价神经网络的性能函数
    :param mnist:
    :return:
    """
    with tf.Graph().as_default() as g:

        x = tf.placeholder(tf.float32, [None, config.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, config.OUTPUT_NODE], name='y-input')
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        if FLAGS.model is 'ann':
            y = ann_inference(x, None)
        elif FLAGS.model is 'cnn':
            y = cnn_inference(x, None)
        elif FLAGS.model is 'lstm':
            y = lstm_inference(x, None)
        else:
            raise Exception('FLAGS.model设置错误')

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(config.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(config.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
            else:
                print('No checkpoint file found')
                return


# 主程序
def main(argv=None):
    mnist = input_data.read_data_sets("Data_sets/MNIST_data", one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    eval_config = get_config()
    main()
