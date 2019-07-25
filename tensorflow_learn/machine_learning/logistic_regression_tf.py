"""
使用tensorflow实现线性回归
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

# 使用pandas读取csv文件
data = pd.read_csv('data/lris_dataset.csv', header=None)
# 获得特征向量和数据标签
features = data.iloc[:, 1:3].values
labels = data.iloc[:, -1:].values
# 划分出积极、消极的数据
features_positive = np.array([features[i] for i in range(len(features)) if labels[i] == 1])
features_negative = np.array([features[i] for i in range(len(features)) if labels[i] == 0])
# 可视化积极、消极数据
plt.scatter(features_positive[:, 0], features_positive[:, 1], color='red', label='Positive')
plt.scatter(features_negative[:, 0], features_negative[:, 1], color='blue', label='Negative')
plt.xlabel('feature_1')
plt.ylabel('feature_2')
plt.legend()
plt.show()

# 使用OneHot编码器
one_hot = OneHotEncoder()
# 编码特征、标签
one_hot.fit(features)
x = one_hot.transform(features).toarray()
one_hot.fit(labels)
y = one_hot.transform(labels).toarray()
# 设置超参数
learning_rate, epochs = 0.0035, 500
m, n = x.shape

# 在进行onehot编码之后特征有n列
train_x = tf.placeholder(tf.float32, [None, n], name='input_x')
train_y = tf.placeholder(tf.float32, [None, 2], name='input_y')
# 学习参数：权重w和偏置b
weight = tf.Variable(initial_value=tf.random.normal(shape=[n, 2], stddev=0.1), name='weight')
bias = tf.Variable(tf.zeros([2]))

# logistic回归
y_pred = tf.nn.sigmoid(tf.add(tf.matmul(train_x, weight), bias))
# 交叉熵损失函数
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=train_y)
# 梯度下降优化损失函数
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# 开启会话
with tf.Session() as sess:
    # 初始化所有变量
    tf.compat.v1.global_variables_initializer().run()
    # 存储每批次的损失函数值和精度值
    loss_history, accuracy_history = [], []
    # 循环训练所有批次
    for epoch in range(epochs):
        # 运行优化器并输送数据
        sess.run(optimizer, feed_dict={train_x: x, train_y: y})
        # 计算当前批次的损失函数
        l = sess.run(loss, feed_dict={train_x: x, train_y: y})
        # 计算当前批次的准确率
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(train_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # 存储loss和精确度到历史中
        loss_history.append(sum(sum(l)))
        accuracy_history.append(accuracy.eval({train_x: x, train_y: y}) * 100)
        # 每100批次显示一次
        if epoch % 100 == 0 and epoch != 0:
            print("Epoch:"+str(epoch)+", loss:"+str(loss_history[-1]))
    # 保存权重和偏置值
    weight = sess.run(weight)
    bias = sess.run(bias)
    # 最终精确度
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(train_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("\nAccuracy:", accuracy_history[-1])

# 绘制损失函数图形
plt.plot(list(range(epochs)), loss_history, label='Loss', color='blue')
plt.plot(list(range(epochs)), accuracy_history, label='Accuracy', color='red')
plt.xlabel('epochs')
plt.ylabel('y')
plt.legend()
plt.title('training data')
plt.show()