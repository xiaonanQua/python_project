"""
配置文件路径
"""
import os


class Config(object):

    def __init__(self):
        # 数据集根目录、项目根目录
        self.root_dataset = '/home/xiaonan/Dataset/'
        self.root_project = '/home/xiaonan/python_project/tensorflow_learn/'

        # cifar-10数据集目录、文件名称
        self.cifar_10_dir = self.root_dataset + 'cifar-10/'
        self.cifar_file_name = ['batches.meta', 'data_batch_1',
                           'data_batch_2', 'data_batch_3',
                           'data_batch_4', 'data_batch_5', 'test_batch']

        # svhn数据集目录、文件名称
        self.svhn_dir = self.root_dataset + 'svhn/'
        self.svhn_file_name = ['train_32.mat', 'test_32.mat', 'extra_32.mat']

        # mnist数据集目录,文件名称
        self.mnist_dir = self.root_dataset + 'mnist/'
        self.mnist_file_name = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz']

        # 数据保存的目录、(svhn、cifar,mnist)批次保存目录
        self.save_data_dir = self.root_project + 'data_load/save_data/'
        self.save_svhn_batch_dir = self.save_data_dir + 'svhn/batch/'

        self.save_mnist_dir = self.root_project + 'mnist/'

        self._init()

    def _init(self):
        # 若文件夹不存在，则创建
        if os.path.exists(self.save_data_dir) is False:
            os.mkdir(self.save_data_dir)
        elif os.path.exists(self.save_svhn_batch_dir) is False:
            os.mkdir(self.save_svhn_batch_dir)
