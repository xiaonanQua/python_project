"""
预处理数据，包含数据归一化，one-hot向量编码，数据转化成TFRecord格式
"""
import os
import numpy as np
import pickle as pk
import tensorflow as tf
from skimage.transform import resize
import data_load.config as cfg


class PreprocessData(object):
    """
    预处理类
    """
    def __init__(self):
       pass

    def one_hot_encode(self, labels):
        """
        对数据进行one_hot编码，将图像类别标签变成one-hot向量
        :param labels: 需要编码成one-hot向量的数据
        :return:
        """
        # 初始化相应数量的one-hot向量
        encode_labels = np.zeros([len(labels), 10])
        for index, value in enumerate(labels):
            if value == 10:
                encode_labels[index][0] = 1  # 若标签值为10，则第一项设置成1
            else:
                encode_labels[index][value] = 1  # 设置标签值对应的位置为1
        return encode_labels

    def normalize(self, images):
        """
        对数据进行最小、最大归一化处理，将数据转化到0-1范围内
        :param images: 输入的图像数据
        :return:
        """
        # 获得图像矩阵中最小、最大值
        min_value = np.min(images)
        max_value = np.max(images)
        # 进行归一化处理，对图像矩阵进行最小、最大操作
        images = (images-min_value)/(max_value-min_value)
        return images

    def resize_images(self, images, resize_tensor):
        """
        调整单个图像的尺度
        :param images: 图像矩阵
        :param resize_tensor: 调整的尺度
        :return: 调整后的图像
        """
        # 定义调整后的图像列表
        resized_images_list = []
        # 循环调整单个图像的尺度
        for image in images:
            resized_image = resize(image, output_shape=resize_tensor)
            resized_images_list.append(resized_image)
        return np.array(resized_images_list)  # 将调整后的图片列表转化成n维数组

    def save_data_pickle(self, images, labels, save_file_dir, save_file_name):
        """
        将数据保存为pickle格式
        :param images: 保存的图像数据
        :param labels: 保存的标签数据
        :param save_file_dir: 保存的文件目录
        :param save_file_name: 保存的文件名称
        :return:
        """
        # 文件不存在，则创建
        if os.path.exists(save_file_dir):
            os.mkdir(save_file_dir)
        # 合成文件路径
        file_path = os.path.join(save_file_dir, save_file_name)
        # 使用pickle对数据进行保存
        if os.path.isfile(file_path) is False:  # 若该文件不存在，则进行保存
            # 以二进制写入的方式打开文件
            with open(file_path, mode='wb') as file:
                pk.dump((images, labels), file)

    def _int64_feature(value):
        """
        将特征数据加入int列表中
        :param value: 加入的int类型特征
        :return:
        """
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(value):
        """
        将特征数据加入字节列表中
        :param value: 特征数据（字符串）
        :return:
        """
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def save_data_tfrecord(self, images, labels, save_file_dir, save_file_name):
        """
            将图片、标签数据打包成TFRecord格式
            :param images: 图片数字数据
            :param labels: 标签数据
            :param save_file_dir: 打包文件的存放目录
            :param save_file_name: 打包文件的名称
            :return:
            """
        # 文件目录不存在，则创建
        if os.path.exists(save_file_dir) is False:
            os.mkdir(save_file_dir)
        # 文件存放路径
        file_path = os.path.join(save_file_dir, save_file_name + '.tfrecords')
        # 获得打包数据的样本数量
        num_examples = labels.shape[0]
        if images.shape[0] != num_examples:
            raise ValueError("图片数量：{}和标签数量{}不一致".format(images.shape[0], num_examples))
        print("进行{}文件的打包..".format(file_path))

        # 获得TFRecord的写入对象
        writer = tf.io.TFRecordWriter(path=file_path)
        # 循环对每一个数据进行转换
        for index in range(num_examples):
            # 将图片矩阵转化成string格式
            image_to_string = images[index].tostring()
            label_to_string = labels[index].tostring()

            # 定义需要保存的属性字典，并添加到总的属性中
            feature = {'label': self._int64_feature(int(labels[index])),
                       'image_raw': self._bytes_feature(image_to_string),
                       'num_example': self._int64_feature(num_examples)}
            features = tf.train.Features(feature=feature)
            # 定义Example类，将前面定义的属性字典写入这个数据结构中
            example = tf.train.Example(features=features)
            # 使用SerializeToString()函数将协议内存块序列化为一个字符串，并写入到TFRecord文件中
            writer.write(example.SerializeToString())
        print("pack end...")
        # 关闭写入流
        writer.close()

    def divide_valid_data(self, images, labels, valid_size):
        """
        从训练集中划分出一定比例的验证集
        :param images: 图像数据
        :param labels: 标签数据
        :param valid_size: 验证集比例
        :return:
        """
        # 计算出验针集的长度
        valid_len = int(len(images)*valid_size)
        # 训练集
        train_data = images[:-valid_len]
        train_labels = labels[:-valid_len]
        # 验证集
        valid_data = images[-valid_len:]
        valid_labels = images[-valid_len:]

        return train_data, train_labels, valid_data, valid_labels




