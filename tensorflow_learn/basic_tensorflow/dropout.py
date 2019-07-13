"""
Dropout:用于解决过拟合问题。在训练时，将神经网络某一层的神经单元数据（不包括输出层的单元数据）随机丢弃一部分。
使用Dropout集合方法需要训练的是从原始网络去掉一些不属于输出层的单元后形成的子网络。
丢弃某些单元并不是指真的构建出这种结构的网络。为了能有效地删除一个单元，我们只需要将这个单元的输出乘以0即可。
可以将每次的单元丢弃都理解为是对特征的一种再采样，这种做法就等于创造许多新的随机样本，以增大样本量、减少特征量的方式防止过拟合。
"""

import tensorflow as tf

# 定义用于dropout处理的数据
x = tf.Variable(tf.ones([10, 10]))

# 定义dro作为dropout处理时的keep_prob参数
dro = tf.placeholder(tf.float32)

# 定义一个dropout操作
y = tf.nn.dropout(x, keep_prob=dro)  # x中每个元素被保留的概率是keep_prob,被保留下来的元素会被乘以1/keep_prob,没有被保留的元素乘以0

# 定义Session()进行计算
with tf.Session() as sess:
    # 所有变量进行初始化
    tf.global_variables_initializer().run()
    print(sess.run(y, feed_dict={dro: 0.8}))  # 大部分情况下对于输入单元，一般会将keep_prob设置为0.8,对于隐藏单元一般会将keep_prob设为0.5
