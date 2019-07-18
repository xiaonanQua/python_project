"""
多元线性回归

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split

# 装载boston数据集
boston = datasets.load_boston(return_X_y=False)
# 定义特征矩阵X和响应向量y
X = boston.data
y = boston.target
# 划分X和y的训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=1)
print(X.shape,y.shape,x_train.shape,x_test.shape,y_train.shape,y_test.shape)

