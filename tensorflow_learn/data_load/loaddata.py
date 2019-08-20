"""
实现各个数据集的加载
"""
import data_load.config as cfg
import scipy.io as sio
import os
import gzip
import struct
import pickle as pk
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import data_load.queue_linked_list as queue
import data_load.preprocessdata as preprocess
import data_load.imageprocess as image_process
from tensorflow.examples.tutorials.mnist import input_data
cfg = cfg.Config()


class LoadDataSet(object):
    def __init__(self):
        # 定义数据集目录，文件名称数组，项目根目录
        self.cifar_10_dir = cfg.cifar_10_dir
        self.cifar_10_file_name = cfg.cifar_file_name
        self.svhn_dir = cfg.svhn_dir
        self.svhn_file_name = cfg.svhn_file_name
        self.mnist_dir = cfg.mnist_dir
        self.mnist_file_name = cfg.mnist_file_name

    def load_svhn_dataset(self, file_type):
        """
        加载svhn数据集
        :param file_type: 文件类型
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

    def load_pickle_data(self, file_dir, file_name):
        """
        加载pickle打包格式的文件数据
        :param file_dir: 文件目录
        :param file_name: 文件名称
        :return: 图像矩阵数据，标签
        """
        # 文件路径
        file_path = os.path.join(file_dir, file_name)
        # 判断文件是否存在
        if os.path.isfile(file_path) is False:
            raise ValueError('文件不存在！')
        # 打开文件
        with open(file_path, 'rb') as file:
            data = pk.load(file)
        return data[0], data[1]

    def load_tfrecord_data(self, file_dir, file_name, multi_files=False, dtype=tf.uint8):
        """
        加载以TFRecord格式保存的数据集
        :param file_dir: 数据存放的文件目录
        :param file_name: 数据的文件名称
        :param mutli_files: boolean,是否进行多文件读取
        :param dtype: 这个属性是用来设置读取后的图像数据转化的格式,因为读取的数据是二进制文件,转化成十进制需要相应的转化格式.
        :return: 图像矩阵数据，标签
        """
        # 定义图片、标签列表
        images_list = []
        labels_list = []
        # 文件路径
        file_path = os.path.join(file_dir, file_name)

        # 创建TFRecordReader类的实例
        reader = tf.TFRecordReader()

        if multi_files:
            # 进行多个文件读取
            files = tf.io.match_filenames_once(file_path)  # 将传入的多个文件整理成文件列表
            file_queue = tf.train.string_input_producer(files)  # 将文件列表以队列的方式进行管理
        else:
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
                                                  # "label": tf.io.FixedLenFeature([], tf.string),
                                                  "num_example": tf.io.FixedLenFeature([], tf.int64)
                                              })

        # decode_raw()用于将字符串解码成图像对应的像素数组
        images = tf.decode_raw(features['image_raw'], dtype)
        # labels = tf.decode_raw(features['label'], tf.uint8)
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
                # print(label)
                images_list.append(image)
                labels_list.append(label)

            # 请求线程停止
            coordinator.request_stop()
            coordinator.join(threads)

        return np.array(images_list), np.array(labels_list)

    def batch_and_shuffle_data(self, images, labels, batch_size, data_queue, shuffle=False):
        """
        获得批次大小的数据并进行洗牌功能
        :param images: 图像数据
        :param labels: 标签数据
        :param batch_size: 批次大小
        :param data_queue: 队列对象,用于创建文件队列
        :param shuffle: 是否打乱批次数据的顺序
        :return:
        """
        # 定义批次图像、标签数据
        batch_images = []
        batch_labels = []

        # 获取batch_size大小的数据
        for i in range(batch_size):
            # 若队列为空,则将总的数据添加到队列中
            if data_queue.is_empty():
                data_queue.data_enqueue(images=images, labels=labels)
            # 从队列中出队数据
            image, label = data_queue.dequeue()
            # 将图像矩阵、标签添加到批次列表中
            batch_images.append(image)
            batch_labels.append(label)

        if shuffle:
            # 定义洗牌后的批次图像、标签数据
            shuffle_batch_images = []
            shuffle_batch_labels = []
            # 根据batch_size生成索引
            index_list = [x for x in range(batch_size)]
            # 使用numpy中随机洗牌函数对索引进行洗牌
            np.random.shuffle(index_list)
            print("洗牌后的索引：{}".format(index_list))
            for index in index_list:
                shuffle_batch_images.append(batch_images[index])
                shuffle_batch_labels.append(batch_labels[index])

            # 将洗牌后的数据赋值给原来变量
            batch_images = shuffle_batch_images
            batch_labels = shuffle_batch_labels

        # 将列表数据转化成N维数组数据
        batch_images, batch_labels =np.array(batch_images), np.array(batch_labels)

        return batch_images, batch_labels

    def multi_thread_load_data(self, file_dir, file_name, dtype=tf.uint8, batch_size=None, capacity=None):
        """
        多线程加载文件数据
        :param file_dir: 文件存在的目录
        :param file_name: 文件名称
        :param dtype: 读取二进制图片数据时,进行十进制转化的类型和位数
        :param batch_size: 批次大小
        :param capacity: 队列缓存的样本数量
        :return:
        """
        # 文件路径
        file_path = os.path.join(file_dir, file_name)
        # 将多个文件生成文件列表
        files = tf.train.match_filenames_once(file_path)
        # 创建多个文件队列,shuffle设置为True,将打乱文件
        files_queue = tf.train.string_input_producer(files, shuffle=True)

        # 实例化读取类,准备读取TFRecord文件
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(files_queue)

        # 解析读取的数据
        features = tf.io.parse_single_example(serialized_example,
                                              features={
                                                  # 使用FixedLenFeature类对属性进行解析
                                                  "image_raw": tf.io.FixedLenFeature([], tf.string),
                                                  "label": tf.io.FixedLenFeature([], tf.int64),
                                                  "num_example": tf.io.FixedLenFeature([], tf.int64)
                                              })

        # decode_raw()用于将字符串解码成图像对应的像素数组
        images = tf.decode_raw(features['image_raw'], dtype)
        # 使用cast()函数进行类型转换
        labels = tf.cast(features['label'], tf.int32)
        num_example = tf.cast(features['num_example'], tf.int32)

        images.set_shape(784)

        # 设置文件队列的最多可以缓存的样本数量
        capacity = capacity + 3*batch_size
        # 使用batch()函数组合成batch
        images_batch, labels_batch = tf.train.batch([images, labels], batch_size=batch_size,
                                                  capacity=capacity)

        # 进行会话计算
        with tf.compat.v1.Session() as sess:
            tf.global_variables_initializer()
            # 启动多线程处理输入数据
            coordinator = tf.train.Coordinator()  # 协调器
            threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)  # 开启所有队列线程
            # 获得样本总数
            num_example = sess.run(num_example)
            # 循环出队队列的数据
            for i in range(3):
                # 每次循环获得队首数据并添加进列表中
                image, label = sess.run([images_batch, labels_batch])
                print(image, label)

            # 请求线程停止
            coordinator.request_stop()
            coordinator.join(threads)

        return image, label

    def load_mnist_dataset(self, image_name, label_name, num_examples=60000):
        """
        读取mnist文件
        :param image_name: 需要读取的图像文件名称
        :param label_name: 需要读取的标签文件名称
        :param num_examples: 需要读取的图像,标签的数量
        :return:
        """
        # 定义图像大小,图像\标签文件路径
        image_size = 28
        image_file_path = os.path.join(self.mnist_dir, image_name)
        label_file_path = os.path.join(self.mnist_dir, label_name)

        # 进行图像文件的读取
        with gzip.open(image_file_path, 'r') as file:
            # 进行16个字节的读取
            file.read(16)
            # 读取相应图片数量的二进制流数据
            buffer = file.read(image_size * image_size * num_examples)
            # 使用frombuffer()函数将二进制数据转化成32位float类型
            images = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
            # 重塑图像的形状
            images = images.reshape(num_examples, image_size, image_size, 1)

        # 进行标签文件的读取
        with gzip.open(label_file_path, 'r') as file:
            # 进行8个字节的读取
            file.read(8)
            # 读取相应图片数量的二进制流数据
            buffer = file.read(num_examples)
            # 使用frombuffer()函数将二进制数据转化成32位int类型
            labels = np.frombuffer(buffer, dtype=np.uint8).astype(np.int32)

        return images, labels


if __name__ == '__main__':
    # 实例化数据集,图像处理类
    dataset = LoadDataSet()
    image_process = image_process.ImageProcess()

    train_data, train_labels = dataset.load_mnist_dataset(dataset.mnist_file_name[0],
                                                          dataset.mnist_file_name[1])
    test_data, test_labels = dataset.load_mnist_dataset(dataset.mnist_file_name[2],
                                                        dataset.mnist_file_name[3],
                                                        num_examples=10000)
    print(train_data.shape, train_labels.shape)
    print(test_data.shape, test_labels.shape)
    image_process.show_image(test_data[0])
    print(test_labels[0])



