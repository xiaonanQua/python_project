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
    i = 0
    # 计算函数
    for x in data:
        relu.append(max(x, 0))
        leaky_relu.append(max(x, 0) + 0.01*min(x, 0))
        par_relu.append(max(x, 0) + param[i]*min(x, 0))
        i = i + 1
    # 可视化函数图像
    plt.title('relu、leaky_relu and par_relu graph')
    plt.plot(data, relu, label='ReLU')
    plt.plot(data, leaky_relu, label='LeakyReLU')
    plt.plot(data, par_relu, label='ParReLU')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def elu_func(data):
    """
    指数线性单元
    :param data:数据
    :return:
    """
    # 参数初始化
    elu = []
    # 计算函数值
    for x in data:
        elu.append(max(x,0)+min(0, 0.02*(math.exp(x)-1)))
    # 实现函数图像
    plt.title('ELU graph')
    plt.plot(data, elu, label='ELU')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def softplus_func(data):
    """
    softplus是修正函数的平滑版本，其导数刚好是logistic
    :param data:数据
    :return:
    """
    # 初始化参数
    softplus = []
    # 进行函数计算
    for x in data:
        softplus.append(math.log(1+math.exp(x)))
    # 可视化函数图像
    plt.title('Softplus graph')
    plt.plot(data, softplus, label='Softplus')
    plt.legend()
    plt.show()


def swish_func(data):
    """
    swish函数是一种自门控的激活函数，swish(x)=xγ(βx),其中γ是Logistic函数，β为可学习或者固定参数
    :param data:
    :return:
    """
    gama = [0, 0.5, 1, 50]  # 定义超参数
    for g in gama:
        swish = []  # 存储函数值
        for x in data:
            logistic = 1/(1+math.exp(-g*x))
            swish.append(x*logistic)
        plt.plot(data, swish, label='β:{}'.format(g))
    plt.title('Swish graph')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data = np.linspace(-10, 10, 100)
    param = np.random.rand(100)
    # print(data)
    # print(param)
    # standard_logistic_func()
    # logistic_func(data, L=1, x_0=-1, k=1)
    # logistic_and_tanh_func(data)
    # hard_logistic_and_tanh(data)
    # relu_func(data, param)
    # elu_func(data)
    # softplus_func(data)
    swish_func(data)