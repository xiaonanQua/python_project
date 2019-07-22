"""
用基本python实现线性回归
"""

import requests
import numpy as np


def collect_dataset():
    """收集CSGO的数据集
    那数据集包含一个选手ADR和评分
    :return:从链接中获得数据集，转化成矩阵,形状为1000*2
    """
    # 使用http获取链接数据
    response = requests.get('https://raw.githubusercontent.com/yashLadha/' +
                            'The_Math_of_Intelligence/master/Week1/ADRvs' +
                           'Rating.csv')
    # 将获取的数据（字符串）存储到列表中
    lines = response.text.splitlines()
    data = []  # 保存数据（float）
    for item in lines:
        item = item.split(',')  # 将单个字符串以‘,’为分隔符进行划分
        data.append(item)  # 附加数据
    data.pop(0)  # 将第一个数据（标签）出栈
    dataset = np.matrix(data)  # 将list转化成1000*2的矩阵
    return dataset


def run_steep_gradient_descent(data_x, data_y, len_data, alpha, theta):
    """
    运行特征向量并且更新相应的特征向量
    :param data_x:输入的数据集
    :param data_y:包含与每个数据输入相关联的输出
    :param len_data:数据集的长度，数据大小
    :param alpha:学习率
    :param theta:特征向量（我们模型的权重）
    :return:更新特征，使用当前特征-alpha×特征梯度
    """
    n = len_data
    print(data_x.transpose())
    prod = np.dot(theta, data_x.transpose())  # theta和x的乘积
    print(prod.shape, data_y.shape, data_y.transpose().astype(float).shape)
    prod -= data_y.transpose().astype(float)

    print(prod)


def run_linear_regression(data_x, data_y):
    """
    对数据集实现线性回归
    :param data_x:包含数据集
    :param data_y:包含输出（结果向量）
    :return:最佳拟合直线特征（特征向量）
    """
    iterations = 100000  # 训练次数
    alpha = 0.000155  # 学习率

    no_features = data_x.shape[1]  # 特征类别
    len_data = data_x.shape[0] - 1  # 数据长度
    theta = np.zeros((1, no_features))  # 参数初始化
    # 利用梯度下降学习参数theta和平方误差和
    for i in range(0, iterations):
        theta = run_steep_gradient_descent(data_x, data_y, len_data, alpha, theta)
        break
    print(theta)
    print(no_features)
    print(len_data)


def main():
    """主函数"""
    data = collect_dataset()
    len_data = data.shape[0]  # 矩阵的形状是个元组形式，第一项代表数据长度
    data_x = np.c_[np.ones(len_data), data[:, :-1]].astype(float)  # 把数据的第一列的前len-1个数切割出来，并转化成float型
    data_y = data[:, -1]  # 切割保留每一行的最后一个数
    theta = run_linear_regression(data_x, data_y)


if __name__ == '__main__':
    main()