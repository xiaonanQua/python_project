"""
实现将数据集打包成TFRecord格式，并进行读取TFRecord格式的文件
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import pickle as pk
from skimage.transform import resize


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


def convert_to_tfrecord(images, labels, save_file_dir, save_file_name):
    """
    将图片、标签数据打包成TFRecord格式
    :param images: 图片数字数据
    :param labels: 标签数据
    :param save_file_dir: 打包文件的存放目录
    :param save_file_name: 打包文件的名称
    :return:
    """
    # 文件存放路径
    file_path = os.path.join(save_file_dir, save_file_name+'.tfrecords')
    # 获得打包数据的样本数量
    num_examples = labels.shape[0]
    if images.shape[0] != num_examples:
        raise ValueError("图片数量：{}和标签数量{}不一致".format(images.shape[0], num_examples))

    # 图片的宽度、高度、深度
    image_width = images.shape[1]
    image_height = images.shape[2]
    image_depth = images.shape[3]

    print("进行{}文件的打包..".format(file_path))
    # 获得TFRecord的写入对象
    writer = tf.io.TFRecordWriter(path=file_path)

    # resize_images = []
    # for i in range(num_examples):
    #     image = resize(images[i], [227, 227, 3])
    #     resize_images.append(image)
    # images = np.array(resize_images)
    # print("重新调整尺度后的图像形状{}".format(images.shape))

    # 循环对每一个数据进行转换
    for index in range(num_examples):
        # 将图片矩阵转化成string格式
        image_to_string = images[index].tostring()
        # 定义需要保存的属性字典，并添加到总的属性中
        feature = {'width': _int64_feature(image_width),
                   'height': _int64_feature(image_height),
                   'depth': _int64_feature(image_depth),
                   'label': _int64_feature(int(labels[index])),
                   'image_raw':_bytes_feature(image_to_string)}
        features = tf.train.Features(feature=feature)
        # 定义Example类，将前面定义的属性字典写入这个数据结构中
        example = tf.train.Example(features=features)
        # 使用SerializeToString()函数将协议内存块序列化为一个字符串，并写入到TFRecord文件中
        writer.write(example.SerializeToString())
    print("pack end...")
    # 关闭写入流
    writer.close()


def read_tfrecords(file_dir, file_name):
    """
    读取tfrecords格式的文件
    :param file_dir: 文件目录
    :param file_name: 文件名称
    :return:images,labels,[width, shape, depth]
    """
    # 文件路径
    file_path = os.path.join(file_dir, file_name)

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
                                           "width": tf.io.FixedLenFeature([], tf.int64),
                                           "height": tf.io.FixedLenFeature([], tf.int64),
                                           "depth": tf.io.FixedLenFeature([], tf.int64)
                                       })

    # decode_raw()用于将字符串解码成图像对应的像素数组
    images = tf.decode_raw(features['image_raw'], tf.uint8)
    # 使用cast()函数进行类型转换
    labels = tf.cast(features['label'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    height = tf.cast(features['height'], tf.int32)
    depth = tf.cast(features['depth'], tf.int32)

    return images, labels, [width, height, depth]


def load_cifar_10_batch(file_dir, file_name):
    """
    加载数据集批次数据
    :param file_dir:数据集目录
    :param file_name: 数据集文件名称
    :return:
    """
    # 文件路径
    file_path = os.path.join(file_dir, file_name)
    # 打开文件
    with open(file_path, mode='rb') as file:
        # 以‘latin1’编码形式，获得数据是data数据格式
        batch = pk.load(file, encoding='latin1')
    features = batch['data'].reshape(-1, 3, 32, 32).transpose(0, 3, 2, 1)
    labels = batch['labels']
    return np.array(features), np.array(labels)


def main():
    # 主函数

    # 数据集目录、项目根目录
    dataset_dir = '/home/xiaonan/Dataset/cifar-10'
    project_dir = '/home/xiaonan/python_project/tensorflow_learn'
    # TFRecord保存路径
    file_dir = project_dir + '/basic_tensorflow/tfrecord/'
    if os.path.exists(file_dir) is False:
        os.mkdir(file_dir)
    # 数据集文件名称
    file_name = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_3', 'data_batch_4', 'data_batch_5']

    # 获得数据集数据
    train_data, train_labels = load_cifar_10_batch(dataset_dir, file_name[0])
    print(train_data.shape, train_labels.shape)
    # 将数据转化成tfrecord格式
    convert_to_tfrecord(train_data, train_labels, file_dir, file_name[0])

    with tf.Session() as sess:
        # 读取TFRecord格式的数据
        images, labels, shape = read_tfrecords(file_dir, file_name[0] + '.tfrecords')
        # 启动多线程处理输入数据
        coordinator = tf.train.Coordinator()  # 协调器
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)  # 开启所有队列线程
        for i in range(len(train_data)):
            image, label, shape = sess.run([images, labels, shape])
            print(label)
            print(shape)
            print(image.shape)
            break


if __name__ == '__main__':
    main()


