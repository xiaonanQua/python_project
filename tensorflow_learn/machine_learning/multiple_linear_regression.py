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

# 创建线性回归对象
reg = linear_model.LinearRegression()
# 使用训练集训练（拟合）模型
reg.fit(x_train, y_train)

# 输出信息
print('训练集、测试集形状：', X.shape,y.shape, x_train.shape, x_test.shape,y_train.shape, y_test.shape)
print('回归系数（regression coefficients）：', reg.coef_, '回归系数数量：{}'.format(len(reg.coef_)))
# 显示方差分数（variance score），1代表完美的预测
print("方差分数：{}".format(reg.score(x_test, y_test)))

# 绘画偏差
# 设置画图风格
plt.style.use('fivethirtyeight')
# 绘制训练集中偏差的散点图
plt.scatter(reg.predict(x_train), reg.predict(x_train) - y_train, color='green', s=10, label='Train data')
# 绘制测试集中偏差的散点图
plt.scatter(reg.predict(x_test), reg.predict(x_test) - y_test, color='blue', s=10, label='test data')
# 绘制水平分割线
plt.hlines(y=0, xmin=0, xmax=50, colors='red', linewidth=2)
plt.legend(loc='upper right')
plt.title('Residual errors')
plt.show()

