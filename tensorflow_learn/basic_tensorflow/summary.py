"""
将一些数据添加到日志文件中,再使用tensorboard可视化
"""

import tensorflow as tf


class Summary(object):
    def __init__(self):
        self.root_project = '/home/xiaonan/python_project/tensorflow_learn'
        self.log_dir = self.root_project + '/basic_tensorflow/log'

    def simple_example(self):
        # 定义一个简单的计算图,实现向量加法的操作
        a = tf.constant([1.2, 3.4, 2.3], dtype=tf.float32, shape=[3], name='input_a')
        b = tf.Variable(initial_value=tf.random.uniform([3]), name='input_b')
        out = tf.add_n([a, b], name='add')

        # 生成一个写日志的writer,并将当前计算图写入日志
        with tf.compat.v1.summary.FileWriter(logdir=self.log_dir, graph=tf.compat.v1.get_default_graph()) as file:
            pass


if __name__ == '__main__':
    log = Summary()
    log.simple_example()
