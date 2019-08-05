"""
实现各个数据集的加载
"""
import data_load.config as cfg
import scipy.io as sio
import os
import pickle as pk
import numpy as np
import tensorflow as tf


class LoadDataSet(object):
    def __init__(self):
        # 定义数据集目录，文件名称数组，项目根目录
        self.cifar_10_dir = cfg.cifar_10_dir
        self.cifar_10_file_name = cfg.cifar_file_name
        self.svhn_dir = cfg.svhn_dir
        self.svhn_file_name = cfg.svhn_file_name

    def load_svhn(self, file_type, valid_size=None):
        """
        加载svhn数据集
        :param file_type: 文件类型
        :param valid_size: 划分验证集的比例
        :return:
        """
        if file_type is 'train':
            # 训练文件路径
            file_path = os.path.join(self.svhn_dir, self.svhn_file_name[0])
        elif file_type is 'test':
            # 测试文件路径
            file_path = os.path.join(self.svhn_dir, self.svhn_file_name[1])
        elif file_type is 'extra':
            file_path = os.path.join(self.svhn_dir, self.svhn_file_name[2])
        else:
            raise ValueError('不存在你输入的文件类型！')
        # 读取.mat格式的文件
        svhn = sio.loadmat(file_path)
        # 通过字典key获得训练数据、标签
        data = svhn['X'].transpose(3, 0, 1, 2)
        labels = svhn['y']
        return data, labels

    def load_preprocess_svhn_pickle(self, preprocess_file_dir, preprocess_file_name):
        """
        加载pickle打包格式的预处理数据
        :param preprocess_file_dir: 预处理文件路径
        :param preprocess_file_name: 预处理文件名称
        :return: 图像矩阵数据，标签
        """
        # 文件路径
        file_path = os.path.join(preprocess_file_dir, preprocess_file_name)
        # 判断文件是否存在
        if os.path.isfile(file_path) is False:
            raise ValueError('文件不存在！')
        # 打开文件
        with open(file_path, 'rb') as file:
            data = pk.load(file)
        return data[0], data[1]

    def load_preprocess_svhn_tfrecord(self, preproce_file_dir, preprocess_file_name):
        """
        加载预处理后的svhn数据集
        :param preproce_file_dir: 预处理的文件目录
        :param preprocess_file_name: 预处理的文件名称
        :return: 图像矩阵数据，标签
        """
        # 定义图片、标签列表
        images_list = []
        labels_list = []
        # 文件路径
        file_path = os.path.join(preproce_file_dir, preprocess_file_name)

        # 创建TFRecordReader类的实例
        reader = tf.TFRecordReader()

        # 创建一个队列对输入文件列表进行维护
        file_queue = tf.train.string_input_producer([file_path])
        # 使用TFRecordReader.read()函数从文件中读取一个样本，使用.read_up_to()函数一次性读取多个样本
        _, serialized_example = reader.read(file_queue)
        # 使用sparse_single_example()函数解析读取的样本
        features = tf.io.parse_single_example(serialized_example,
                                              features={
                                                  # 使用FixedLenFeature类对属性进行解析
                                                  "image_raw": tf.io.FixedLenFeature([], tf.string),
                                                  "label": tf.io.FixedLenFeature([], tf.int64),
                                                  "num_example":tf.io.FixedLenFeature([], tf.int64)
                                              })

        # decode_raw()用于将字符串解码成图像对应的像素数组
        images = tf.decode_raw(features['image_raw'], tf.uint8)
        # 使用cast()函数进行类型转换
        labels = tf.cast(features['label'], tf.int32)
        num_example = tf.cast(features['num_example'], tf.int32)

        # 进行会话计算
        with tf.compat.v1.Session() as sess:
            # 启动多线程处理输入数据
            coordinator = tf.train.Coordinator()  # 协调器
            threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)  # 开启所有队列线程
            # 获得样本总数
            num_example = sess.run(num_example)
            # 循环出队队列的数据
            for i in range(num_example):
                # 每次循环获得队首数据并添加进列表中
                image, label = sess.run([images, labels])
                images_list.append(image)
                labels_list.append(label)
        return np.array(images_list), np.array(labels_list)

    def batch_and_shuffle_data(self, images, labels, batch_size, shuffle=False):
        """
        获得批次大小的数据
        :param images:
        :param labels:
        :param batch_size:
        :param shuffle:
        :return:
        """


if __name__ == '__main__':
    dataset = LoadDataSet()
    svhn = dataset.load_svhn(file_type='train')



