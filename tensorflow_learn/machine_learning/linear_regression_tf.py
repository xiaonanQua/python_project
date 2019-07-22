"""
使用tensorflow实现线性回归
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 为了制定可预测的随机数，我们将为numpy和tensorflow定义固定的种子
np.random.seed(101)
tf.compat.v1.set_random_seed(101)

# 为训练线性回归模型生成一些随机数据
# 在0到50之间有50个数据点
x = np.linspace(0, 50, 50)
y = np.linspace(0, 50, 50)
# print(x, "数量:{}".format(len(x)))
# print(y, '数量：{}'.format(len(y)))
# 为随机线性数据添加噪音
x_2 = np.random.uniform(-4, 4, 50)  # 数据是均值分布
y_2 = np.random.uniform(-4, 4, 50)  # 数据是均值分布
train_x = x + x_2
train_y = y + y_2
n = len(x)  # 数据点的数量
# print(train_x, "数量:{}".format(len(train_x)))
# print(train_y, '数量：{}'.format(len(train_y)))

# 可视化数据
# plt.scatter(train_x, train_y)
# plt.show()

# 设置占位符x，y以至于在整个训练过程将训练样本x，y输送到优化器中
x = tf.compat.v1.placeholder('float', name='input_x')
y = tf.compat.v1.placeholder('float', name='input_y')

# 声明两个可训练的tensorflow变量，分别为权重和偏置值，并使用高斯分布来初始化变量
w = tf.Variable(np.random.normal(scale=0.1), name='weights')
b = tf.Variable(np.random.normal(scale=0.1), name='bias')
# w = tf.Variable(np.random.randn(), name='weights')
# b = tf.Variable(np.random.randn(), name='bias')

# 设置超参数
learning_rate = 0.01  # 学习率
training_epochs = 1000  # 训练批次

# 预测
y_pred = tf.add(tf.multiply(x, w), b)
# 均方差损失函数
loss = tf.reduce_sum(tf.compat.v1.pow(y_pred - y, 2)) / (2*n)
# 梯度下降优化器
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# 执行session
with tf.compat.v1.Session() as sess:
    # 初始化所有变量
    tf.compat.v1.global_variables_initializer().run()
    # 循环所有训练批次
    for epoch in range(training_epochs):
        for x_, y_ in zip(train_x, train_y):  # 打包所有数据，每次训练，训练所有数据
            # 使用输送字典输送每一个数据点给优化器
            sess.run(optimizer, feed_dict={x: x_, y: y_})
        # 在50批次之后显示结果
        if (epoch+1) % 50 == 0:
            # 计算一次批次的损失
            single_loss = sess.run(loss, feed_dict={x: train_x, y: train_y})
            print("epoch:{}, loss:{}, weight:{}, bias:{}".format(epoch+1,
                                                                 single_loss,
                                                                 sess.run(w),
                                                                 sess.run(b)))
    # 保存一些必要值，用于会话外使用
    train_loss = sess.run(loss, feed_dict={x: train_x, y: train_y})
    weight = sess.run(w)
    bias = sess.run(b)

# 计算预测值
predictions = weight * train_x + bias
print(predictions)
print("Train loss:{}, weight:{}, bias:{}".format(train_loss,
                                                 weight,
                                                 bias))
# 绘制结果
plt.scatter(train_x, train_y, color='blue', label='Original data')
plt.plot(train_x, predictions, label='Fitted line')
plt.title('Linear Regression Result')
plt.legend()
plt.show()
