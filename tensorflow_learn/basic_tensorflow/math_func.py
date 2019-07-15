"""
可视化一些常见的数学函数
"""
import numpy as np
import matplotlib.pyplot as plt
import math


def standard_logistic_func():
    """
    标准logistic函数的实现及可视化
    :return:
    """
    x = np.arange(-10, 10)
    y = []
    for i in x:
        value = 1./(1+math.exp(-i))
        y.append(value)
    print(y)
    plt.plot(x, y, label='logistic')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def logistic_func(data, L, x_0, k):
    """
    常规logistic函数
    :param data:x的范围
    :param L:最大值
    :param x_0:中心点
    :param k:曲线的倾斜度
    :return:
    """
    y = []  # 保存logistic值
    for x in data:
        value = L / (1 + math.exp(-k*(x-x_0)))
        y.append(value)
    print(y)
    plt.plot(data, y)
    plt.show()


def logistic_and_tanh_func(data):
    """
    logistic函数和tanh函数的可视化
    :param data: 数据
    :return:
    """
    # 初始化变量
    logistic = []
    tanh = []
    # 计算logistic函数和tanh函数的值
    for x in data:
        logistic.append(1/(1+math.exp(-x)))
        tanh.append((math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x)))
    plt.title('logistic and tanh graph')
    plt.plot(data, logistic, label='logistic')
    plt.plot(data, tanh, label='tanh')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def hard_logistic_and_tanh(data):
    """
    hard_logistic和hard_tanh函数的可视化
    :param data:数据
    :return:
    """
    # 初始化参数
    logistic = []
    hard_logistic = []
    tanh = []
    hard_tanh = []

    # 计算函数值
    for x in data:
        logistic.append(1/(1+math.exp(-x)))
        hard_logistic.append(max(0, min(0.25*x+0.5, 1)))
        tanh.append((math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x)))
        hard_tanh.append(max(-1,min(x,1)))
    # 可视化函数值
    plt.title('hard_logistic and hard_tanh graph')
    plt.plot(data, logistic, label='logistic')
    plt.plot(data, hard_logistic, label='hard_logistic')
    plt.plot(data, tanh, label='tanh')
    plt.plot(data, hard_tanh, label='hard_tanh')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def relu_func(data, param):
    """
    修正线性单元ReLU函数及相应的改进函数：Leaky ReLU(带泄漏的ReLU）和Parametric ReLU(带参数的ReLU)
    :param data:数据
    :param param: 模拟Parametric ReLU函数的可学习的参数
    :return:
    """
    # 初始化参数
    relu = []
    leaky_relu = []
    par_relu = []
    # 计算函数
    for i, x in data:
        relu.append(max(x, 0))
        leaky_relu.append(x, 0.01*x)
        par_relu.append(max(x, param[i]*x))


if __name__ == "__main__":
    data = np.linspace(-10, 10, 100)
    param = np.linspace(0, 1, 100, endpoint=False)
    param = np.random.rand(100)
    # print(data)
    print(param)
    # standard_logistic_func()
    # logistic_func(data, L=1, x_0=-1, k=1)
    # logistic_and_tanh_func(data)
    hard_logistic_and_tanh(data)