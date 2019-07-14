'''
逻辑回归
'''
import tensorflow as tf
import pandas as pd
import sklearn.model_selection as sk
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


def liner_regression_diabetes():
    # 加载糖尿病(diabetes)数据集
    diabetes = datasets.load_diabetes()
    # 使用仅仅一个特征
    diabetes_x = diabetes.data[:, np.newaxis, 2]
    # 划分x数据为训练集和测试集
    diabetes_x_train = diabetes_x[:-20]  # 数据从0开始到倒数第20个为训练集
    diabetes_x_test = diabetes_x[-20:]  # 倒数第20个开始到最后为测试集
    # 划分y数据为训练集和测试集
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]

    # print(diabetes)
    # print(diabetes_x)
    # print(diabetes_x_train)
    # print('dd')
    print(diabetes_x_test)
    print(diabetes_y_test)

    # 创建线性模型对象
    regr = linear_model.LinearRegression()
    # 使用训练集来训练(拟合)模型
    regr.fit(diabetes_x_train, diabetes_y_train)
    # 使用测试集做出预测
    diabetes_y_pred = regr.predict(diabetes_x_test)
    # 回归系数
    print("回归系数：{}".format(regr.coef_))
    # 均方误差
    print("mean squared error:%.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    # 解释变量得分，1是完美预测
    print('变量得分：%.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

    # 使用plot绘画结果图形
    plt.scatter(diabetes_x_test, diabetes_y_test, color='black')  # 将数据集的测试集以散点图的形式展现
    plt.plot(diabetes_x_test, diabetes_y_pred, color='blue', linewidth=3)
    plt.xticks()
    plt.yticks()

    plt.show()


def liner_regression_example():
    """
    线性回归示例
    :return:
    """
    x = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])  # 生成4×2的矩阵数据
    y = np.dot(x, np.array([1, 2])) + 3  # y = 1*x_1 + 2*x_2 +3
    reg = linear_model.LinearRegression().fit(x, y)  # 线性拟合
    print(x.shape)
    print(np.array([1, 2]))
    print(y)
    print(reg.score(x,y))
    print(reg.coef_)
    print(reg.intercept_)
    print(reg.predict(np.array([[3, 5]])))


def ridge_regression():
    """
    展示了优化器岭回归系数的共线性
    :return:
    """
    # x是一个10*10的矩阵，1×10+10*1=10*10
    x = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
    y = np.ones(10)

    # 计算路径
    n_alphas = 200
    alphas = np.logspace(-10, -2, n_alphas)  # 产生岭回归的系数

    coefs = []
    for a in alphas:
        ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
        ridge.fit(x,y)
        coefs.append(ridge.coef_)

    # 显示结果
    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Ridge coefficients as a function of the regularization')
    plt.axis('tight')
    plt.show()

# tf.compat.v1.enable_eager_execution()  # 使程序能够更快的运行
# csv_path = 'data.csv'  # csv文件路径
# def grade_map(grade1, grade2, label):
#     '''
#     csv文件数据映射成字典形式
#     :param grade1: 分数1
#     :param grade2: 分数2
#     :param label: 标签
#     :return: 字典数据
#     '''
#     return {'grade1': grade1, 'grade2': grade2, 'label': label}
# # 读取csv数据
# dataset = tf.data.experimental.CsvDataset(csv_path,  # csv文件路径
#                                           # 规定每列的类型
#                                           record_defaults=[tf.float32, tf.float32, tf.int32],
#                                           select_cols=[0, 1, 2],  # 选择的列
#                                           field_delim=',',  # 分割符
#                                           header=True)  # 当csv文件中有头标签，当进行解析时进行跳过
# 对数据进行进一步的处理
# dataset = dataset.filter(lambda grade1, grade2, label: label < 1)
# dataset = dataset.map(map_func=grade_map)
# dataset = dataset.batch(1)


if __name__ == '__main__':
    # 显示数据
    # for element in dataset:
    #     tf.print(element)
    # liner_regression_example()
    # liner_regression_diabetes()
    ridge_regression()