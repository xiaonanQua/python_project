"""
配置文件路径
"""
import os
# 数据集根目录、项目根目录
root_dataset = '/home/xiaonan/Dataset/'
root_project = '/home/xiaonan/python_project/tensorflow_learn/'

# cifar-10数据集目录、文件名称
cifar_10_dir =  root_dataset + 'cifar-10/'
cifar_file_name = ['batches.meta', 'data_batch_1',
                   'data_batch_2', 'data_batch_3',
                   'data_batch_4', 'data_batch_5', 'test_batch']

# svhn数据集目录、文件名称
svhn_dir = root_dataset + 'svhn/'
svhn_file_name = ['train_32.mat', 'test_32.mat', 'extra_32.mat']

# 预处理保存的路径
preprocess_file_dir = root_project + 'sava_data/'

# 若文件夹不存在，则创建
if os.path.exists(preprocess_file_dir) is False:
    os.mkdir(preprocess_file_dir)