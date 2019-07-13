"""
损失函数的使用
"""

import tensorflow as tf

# 交叉熵的使用
y = tf.constant([1.0, 0, 0, 0], dtype=tf.float32)  # 真实值
y_ = tf.constant([0.2, 0.8, 0, 0], dtype=tf.float32)  # 预测值

# 自定义损失函数的变量
a = tf.constant([1.0, 2.0, 3.0, 4.0])
b = tf.constant([6.0, 5.0, 4.0, 3.0])

# 构建交叉熵，衡量预测值和真实值之间的分布距离
cross_entropy = -tf.reduce_mean(y * tf.clip_by_value(y_, 1e-10, 1.0))  # clip_by_value()将张量中的数值限制在0.1-0.7的范围内

with tf.Session() as sess:
    print(sess.run(cross_entropy))
    # 自定义损失函数
    print(tf.greater(a, b).eval())  # 使用greater比较相同维度两个张量各个位置元素的大小
    print(tf.where(tf.greater(a, b), a, b).eval())  # where是一个选择函数，第一个参数是选择条件，bool值类型，若为True，则选择第二个参数对应的位置

