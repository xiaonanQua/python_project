"""
对于一个假设的线性函数，使用梯度下降法最小化其损失函数
"""

import numpy as np
import matplotlib.pyplot as plt

# 定义训练、测试的数据和标签
train_data = (((5, 2, 3), 15), ((6, 5, 9), 25), ((11, 12, 13), 41), ((1, 1, 1), 8), ((11, 12, 13), 41))
test_data = (((515, 22, 13), 555), ((61, 35, 49), 150))
parameter_vector = [2, 4, 1, 5]
m = len(train_data)  # 训练数据的长度
LEARNING_RATE = 0.009  # 学习率


def get_cost_derivative(index):
    """
    得到损失函数关于theta的偏导数值，损失函数使用均方损失函数
    :param index:
    :return:
    """


def run_gradient_descent():
    global parameter_vector
    # 调整这些值，为预测的输出设置一个容差值
    absolute_error_limit = 0.00002
    relative_error_limit = 0
    j=0
    while True:
        j+=1
        temp_parameter_vector = [0, 0, 0, 0]  # 临时参数向量
        for i in range(0, len(parameter_vector)):
            pass
