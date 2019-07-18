"""
使用python实现简单的线性回归
线性回归模型是统计方法中关于给定的独立自变量和因变量之间的关系。
在本次实验中，我们称因变量为映射、自变量为简单的特征。
为了提供线性回归的基本理解，我们使用最基本的线性回归，即简单的线性回归。
简单线性回归是使用一个单独特征预测一个映射的方法。总的来说， 那两个变量是线性相关的。
因此，我尝试发现一个线性函数，作为特征或者自变量x的函数，尽可能准确地预测y值（映射）
"""
import numpy as np
import matplotlib.pyplot as plt


def estimate_coef(x, y):
    n = np.size(x)  # 特征数量
    m_x, m_y = np.mean(x), np.mean(y)  # 计算特征和映射的平均值
    # 计算x,y的交叉偏差
    ss_xy = np.sum(x*y) - n*m_y*m_x
    # 计算x的平方偏差和
    ss_xx = np.sum(x*x) - n*m_x*m_x
    # 计算回归系数
    b_1 = ss_xy / ss_xx
    b_0 = m_y - b_1*m_x

    return (b_0, b_1)


def plot_regression_line(x,y, b):
    # 绘制真实数据的散点图
    plt.scatter(x, y, color='m', marker='o', s=30)
    y_pred = b[1]*x + b[0]
    plt.plot(x, y_pred, color='g')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def main():
    # 数据
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9,10, 12])
    # 显示散点图
    # plt.scatter(x, y)
    # plt.show()
    # 优化系数
    b = estimate_coef(x, y)
    print('估计的系数：b_0:{},b_1:{}'.format(b[0], b[1]))
    # 绘制回归线
    plot_regression_line(x, y, b)


if __name__ == '__main__':
    main()