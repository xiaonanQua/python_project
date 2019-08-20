"""
提供图像数据处理方法:1.查看图像数据
"""
import matplotlib.pyplot as plt
import matplotlib.pylab as lab
import numpy as np


class ImageProcess(object):
    def __init__(self):
        pass

    def show_image(self, image_matrix):
        """
        使用matplotlib显示图像矩阵,包括灰度图像和RGB图像
        :param image_matrix: 一张图像矩阵,格式是(高度, 宽度, 颜色通道)
        :param label: 标签数据,和图像相对应
        :return:
        """
        # 颜色通道
        num_channel = image_matrix.shape[2]
        # 若是灰度图像,则需要进行挤压
        if num_channel == 1:
            # 对灰度图像进行挤压,使得形状格式为(高度, 宽度)
            image_matrix = np.asarray(image_matrix).squeeze()
            plt.imshow(image_matrix, interpolation='none', cmap=lab.gray())
        else:
            # 显示图片
            plt.imshow(image_matrix)
        plt.show()