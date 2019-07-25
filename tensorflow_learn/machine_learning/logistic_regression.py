"""
使用python实现logistic回归
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# 装载数据
dataset = pd.read_csv('data/User_Data.csv')
# 开始去预测一个用户是否购买商品，我们需要去发现年龄和评估的价格之间的关系。其中用户ID和性别对于发现商品不是重要的因素。
# 划分数据、标签，转化成数组
x = dataset.iloc[:, [2, 3]].values  # 数据400*2,400个数据，每个数据两个特征
y = dataset.iloc[:, 4].values    # 标签400
# 将数据划分成训练集和测试集，划分比例0.75,
train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.75, random_state=0)
# 数据特征进行缩放，因为age和salary值位于不同的数值范围内。若不对其进行缩放，那么当模型找到数据空间中数据点的最近邻点时，
# salary特征会主导age特征
sc_x = StandardScaler()
train_x = sc_x.fit_transform(train_x)
test_x = sc_x.transform(test_x)  # 选用另一种缩放方法是为了使得训练集和测试集的数据不同
# 训练Logistic回归模型
classifier = LogisticRegression(random_state=0)
classifier.fit(train_x, train_y)  # 训练（拟合）模型
# 对于训练后的模型进行预测
y_pred = classifier.predict(test_x)
# 使用混淆矩阵(误差矩阵)对预测的结果进行性能度量
cm = confusion_matrix(test_y, y_pred)
print("confusion_matrix:\n{}".format(np.matrix([['True Positive', 'False Positive'], ['False Negative', 'True Negative']])))
print(cm)
print('Precision:{}={}'.format('TP/(TP+FP)', cm[0][0]/(cm[0][0]+cm[0][1])))
print('Recall:{}={}'.format('TP/(TP+FN)', cm[0][0]/(cm[0][0]+cm[0][1])))
# 进行准确率的估计
acc = accuracy_score(test_y, y_pred)
print('Accuracy:{}'.format(acc))

# 可视化


