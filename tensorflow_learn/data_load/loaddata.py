"""
实现各个数据集的加载
"""
import data_load.config as cfg
import scipy.io as sio
import os


class LoadDataSet(object):
    def __init__(self):
        # 定义数据集目录，文件名称数组，项目根目录
        self.cifar_10_dir = cfg.cifar_10_dir
        self.cifar_10_file_name = cfg.cifar_file_name
        self.svhn_dir = cfg.svhn_dir
        self.svhn_file_name = cfg.svhn_file_name

    def load_svhn(self, train=True, value=True):
        """
        加载svhn数据集
        :param train=True
        :param value=True
        :return:
        """




