"""
简单实现全连接的前向传播算法
全连接与稀疏连接指的是神经网络模型中相邻两层单元间的连接方式。
使用全连接，网络当前单元与上一层的每一个单元都存在连接。
使用稀疏连接方式，网络当前层的单元只与网络上一层的部分单元连接。
前向传播算法描述了前馈神经网络的计算过程，计算前向传播算法需要三部分信息：
（1）神经网络的输入，这个输入是经过提取的特征向量数据。
（2）神经网络的连接结构，通过连接结构可以确定运算关系。
（3）每个神经元中的参数
"""
import tensorflow as tf

# 定义输入数据，常量
x = tf.constant([1.0, 2.3], dtype=tf.float32, shape=[1, 2])

# 使用随机正态分布函数初始化权重参数w1和w2变量，其中w1是2×3,w2是3×1的矩阵
w1 = tf.Variable(tf.random_normal(shape=[2, 3], stddev=1.0, seed=1), name='w1')
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1.0, seed=1), name='w2')

# 使用tf.zeros()函数初始化偏置项b1和b2,其中b1是1×3的矩阵，b2是1×1的矩阵
b1 = tf.Variable(tf.zeros([1, 3]), name='b1')
b2 = tf.Variable(tf.zeros([1]), name='b2')

# 初始化所有定义的变量
init_op = tf.global_variables_initializer()

# 进行前向运算,使用Relu激活函数进行去线性化，进行非线性转化
a = tf.nn.relu(tf.matmul(x, w1) + b1)
y = tf.nn.relu(tf.matmul(a, w2) + b2)

# 使用session执行运算
with tf.Session() as sess:
    sess.run(init_op)  # 初始化变量
    print(sess.run(y))  # 输出前向传播的运算结果
