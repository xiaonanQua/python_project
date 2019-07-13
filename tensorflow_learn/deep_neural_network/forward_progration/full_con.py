"""
定义四层全连接神经网络带L2正则化损失函数的功能
"""
import tensorflow as tf
import numpy as np

# 定义训练的轮数
training_steps = 3000
# 定义进行训练的数据和标签
data = []
label = []
# 迭代200次生成数据、标签,属于二分类问题
for i in range(200):
    x1 = np.random.uniform(-1, 1)
    x2 = np.random.uniform(0, 2)
    # 对x1和x2进行判断，如果产生的点落在半径为1的圆内，则标签为0,否则标签为1
    if x1**2 + x2**2 <=1:
        # 添加数据、标签
        data.append([np.random.normal(x1, 0.1), np.random.normal(x2, 0.1)])
        label.append(0)
    else:
        # 添加数据标签
        data.append([np.random.normal(x1, 0.1), np.random.normal(x2, 0.1)])
        label.append(1)

# 使用hstack()函数用于在水平方向将数据堆起来，reshape()函数的参数-1表示行列进行翻转
data = np.hstack(data).reshape(-1, 2)
label = np.hstack(label).reshape(-1, 1)



def hidden_layer(input_data, weight1, bias1, weight2, bias2, weight3, bias3):
    """
    定义前向传播的隐藏层
    :param input_data:输入数据
    :param weight1: 权重1
    :param bias1: 偏置值1
    :param weight2:
    :param bias2:
    :param weight3:
    :param bias3:
    :return: 经过2层隐藏层进行的计算
    """
    layer1 = tf.nn.relu(tf.matmul(input_data, weight1) + bias1)  # 第一层，输入层
    layer2 = tf.nn.relu(tf.matmul(layer1, weight2) + bias2)  # 第二层隐藏层
    return tf.matmul(layer2, weight3) + bias3


# 定义输入数据、标签占位符
x = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='input_data')
y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='input_label')

# 定义权重和偏置值变量
weight1 = tf.Variable(tf.truncated_normal([2, 10], stddev=0.1))
bias1 = tf.Variable(tf.constant([0.1], shape=[10]))
weight2 = tf.Variable(tf.truncated_normal([10, 10], stddev=0.1))
bias2 = tf.Variable(tf.constant([0.1], shape=[10]))
weight3 = tf.Variable(tf.truncated_normal([10, 1], stddev=0.1))
bias3 = tf.Variable(tf.constant([0.1], shape=[1]))

# 计算数据数组长度
data_size = len(data)
# 得到前向传播结果
y_ = hidden_layer(x, weight1, bias1, weight2, bias2, weight3, bias3)

# 自定义的损失函数，平方均差的损失函数，衡量计算值与实际值之间的差距
error_loss = tf.reduce_mean(tf.pow(y_ - y, 2)) / data_size

weight_decay = tf.constant(0.01)
l2_loss = weight_decay * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])

loss = error_loss + l2_loss
# tf.add_to_collection('loss', error_loss)  # 将平方差的损失函数值加入集合中
#
# # 对权重参数实现l2正则化，防止过拟合。正则化优化不是直接优化损失函数，而是优化权重参数并加入损失函数中
# regularization = tf.nn.l2_loss(weight1) + tf.nn.l2_loss(weight2) + tf.nn.l2_loss(weight3)
# tf.add_to_collection('loss', regularization)  # 将正则化的参数加入集合中
#
# # 获取总的loss，通过get——collection()函数获取指定集合中所有个体，此处获取所有损失值和正则化值
# # 再使用add_n()进行加和运算
# loss = tf.add_n(tf.get_collection('loss'))

# 定义优化器来优化网络中的参数，也就是权重和偏置值，来使损失值不断下降。
# 其中优化器中的学习率用来每次优化学习的幅度或者速度,此处设置的学习率是0.01,实际要大于
train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

# 启动session进行网络训练
with tf.Session() as sess:
    # 初始化全局所有定义的变量
    tf.global_variables_initializer().run()
    # 进行3000轮训练
    for i in range(30000):
        # 进行训练操作，输送训练的数据、标签
        sess.run(train_op, feed_dict={x: data, y:label})
        # 每隔300轮输出一次loss值
        if i % 300 == 0:
            loss_value = sess.run(loss, feed_dict={x: data, y:label})
            print("After %d steps, loss:%f" % (i, loss_value))



