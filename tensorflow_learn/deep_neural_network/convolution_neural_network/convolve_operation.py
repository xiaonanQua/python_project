"""
使用python实现一系列卷积运算
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import os


class ConvOp(object):
    """
    使用python实现常规卷积操作
    """
    def __init__(self):
        self.data_path = None
        self.data = None
        self.data_array = []

    def acquire_image_data(self):
        """
        获取cifar10图像数据
        :param data_path:数据路径
        :return: 数组形式，包含训练集，测试集，标签名称
        """
        train_data = []
        test_data = []
        label_name = []
        for i in range(len(self.data_array)):
            # 获得每条文件路径
            file_path = os.path.join(self.data_path, self.data_array[i])
            # 读取文件，以二进制读取数据
            with open(file_path, mode='rb') as file:
                data = pk.load(file, encoding='latin1')  # 装载pickle封装的数据，编码设置成latin1，为了获得data格式
            if i < 5:
                train_data.append(data)
            elif i == 5:
                test_data.append(data)
            else:
                label_name.append(data)
        return train_data, test_data, label_name

    def horizontal_op(self):
        """
        获取图像水平边缘
        :return:
        """
        pass


if __name__ == '__main__':
    conv = ConvOp()
    conv.data_path = '/home/xiaonan/Dataset/cifar-10/'
    conv.data_array = ['data_batch_1', 'data_batch_2', 'data_batch_3',
                       'data_batch_4', 'data_batch_5', 'test_batch', 'batches.meta']
    train_data, test_data, label_name =conv.acquire_image_data()
    print(train_data[0]['labels'])
    print(test_data[0]['data'].shape)
    print(label_name[0])