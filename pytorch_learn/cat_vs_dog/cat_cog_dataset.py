"""
重写数据集类，写猫狗数据集对象
"""

import torch
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class CatDogDataset(Dataset):
    """
    继承父类数据集,重写__len__和__getitem__方法
    """
    def __init__(self, file_list, file_dir, mode='train', transform=None):
        """
        构造函数
        :param file_list: 猫狗文件的列表
        :param file_dir: 猫狗数据集存在目录
        :param mode: 训练模式还是测试模式
        :param transform: 对图像数据进行一系列操作对象
        """
        self.file_list = file_list
        self.file_dir = file_dir
        self.mode = mode
        self.transform = transform
        # 若是训练模式，则设置标签
        if self.mode is 'train':
            if 'dog' in self.file_list[0]:
                # 图像为狗，则设置标签为1
                self.label = 1
            else:
                # 设置成猫
                self.label = 0

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # 打开图片
        image = Image.open(os.path.join(self.file_dir, self.file_list[idx]))
        if self.transform:
            # 对图像进行相应的操作
            image = self.transform(image)
        if self.mode == 'train':  # 返回训练数据
            image = image.numpy()  # 将图像矩阵转化成numpy类型
            return image.astype('float32'), self.label
        else: # 返回测试数据
            image = image.numpy()
            return image.astype('float32'), self.label




