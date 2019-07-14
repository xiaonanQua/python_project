"""
压缩感知：用l1优先进行断层重建。使用lasso回归
"""
import numpy as np
from scipy import sparse
from scipy import ndimage
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt


def _weights(x, dx=1, orig=0):
    """
    设置权重
    :param x:
    :param dx:
    :param orig:
    :return:
    """
    x = np.ravel(x)  # 将多维的矩阵x连续排列成一维矩阵
    floor_x = np.floor((x - orig)/dx).astype(np.int64)  # 将计算结果进行取整，并转化成int类型
    alpha = (x - orig - floor_x*dx) / dx  # 计算出alpha系数
    return np.hstack((floor_x, floor_x+1)), np.hstack((1 - alpha, alpha))