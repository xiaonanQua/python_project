"""
使用mnist数据集
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/xiaonan/Dataset/mnist/", one_hot=True)

# 设置数据、标签占位符
x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x_input')
y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y_input')

# 超参数设置
batch_size = 100  # 设置每一轮训练的批次（batch）大小为100
learning_rate = 0.8  # 优化算法中的学习率
learning_rate_decay = 0.999  # 学习率的衰减
max_steps = 30000  # 最大训练步数
training_step = tf.Variable(0, trainable=False)  # 定义保存训练步骤的变量，默认设置变量不放入计算图中

# 设置权重、偏置项等参数
# 设置隐藏层的参数
weight1 = tf.Variable(initial_value=tf.truncated_normal([784, 500], stddev=0.1))
bias1 = tf.Variable(initial_value=tf.constant(value=0.1,shape=[500]))
# 设置输出层的参数
weight2 = tf.Variable(tf.truncated_normal([500, 10], stddev=0.1))
bias2 = tf.Variable(tf.constant(0.1, shape=[10]))


def hidden_layer(input_data, weight1, bias1, weight2, bias2, layer_name):
    """
    定义神经网络模型的隐藏层和输出层，激活函数使用Relu,实现前向传播计算
    :param input_data: 输入的数据
    :param weight1: 权重
    :param bias1: 偏置项
    :param weight2:
    :param bias2:
    :param layer_name:层名称
    :return: 输出层的分布值
    """
    layer1 = tf.nn.relu(tf.matmul(input_data, weight1) + bias1)  # 先使用线性模型，再使用relu去线性化
    return tf.matmul(layer1, weight2) + bias2


# 获得经过神经网络前向传播后的y_值
y_ = hidden_layer(x, weight1, bias1, weight2, bias2, 'y_output')

# 为了在采用随机梯度下降算法训练神经网络时提高最终模型在测试数据上的表现，TensoFlow提供了一种在变量上使用滑动平均
# 的方法，通常称之为滑动平均模型。
# 滑动平均算法会对每一个变量的影子变量（shadow_variable）进行维护，影子变量的初始值为相应变量的初始值。
# 若一个变量进行变化，其影子变量的值按照shadow_variable(n+1) = decay*shadow_variable(n)+(1-decay)*variable(n+1)
# 其中decay为衰减率，decay决定了滑动平均模型的更新速度

# 初始化一个滑动平均类，衰减率为0.99
# 为了使模型更新的更快，类中设置了num_updates参数，此处参数的值设置为训练轮数
averages_class = tf.train.ExponentialMovingAverage(0.99, num_updates=training_step)
# 定义一个更新变量滑动平均值的操作需要向滑动平均类的apply()提供一个参数列表
# train_variable()函数返回集合图上Graph.TRAINING_VARIABLES中的元素，这个集合就是所有没有指定
# trainable_variable=False的参数
averages_op = averages_class.apply(tf.trainable_variables())
# 再次计算经过神经网络前向传播得到的y值，这里使用了滑动平均值，滑动平均值只是一个影子变量
average_y = hidden_layer(x, averages_class.average(weight1),
                         averages_class.average(bias1),
                         averages_class.average(weight2),
                         averages_class.average(bias2), 'average_y')

# 使用sparse_softmax_cross_entropy_with_logits()将输入的样本划分为某一类，即图片只有一个数字类似情况
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y, 1), logits=y_)
# 使用l2正则化损失
regularization = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
# 总损失
loss = tf.reduce_sum(cross_entropy) + regularization

# 使用反向传播算法，即使用优化算法队loss中参数进行优化更新
learning_rate = tf.train.exponential_decay(learning_rate, training_step,
                                           mnist.train.num_examples/batch_size,
                                           learning_rate_decay)  # 用指数衰减设置学习率
train_op = tf.train.\
    GradientDescentOptimizer(learning_rate=learning_rate).\
    minimize(loss, global_step=training_step)  # 使用随机梯度下降优化算法来优化交叉熵损失和正则化损失，达到参数更新学习
# 在训练这个模型时，每过一遍即需要通过反向传播来更新网络中的参数又需要更新每一个参数的滑动平均值
# control_dependencies()用于完成这样的一次性多次操作
with tf.control_dependencies([train_op, averages_op]):
    train_op = tf.no_op(name='train')
# train_op = tf.group(train_op, averages_op)  此代码的功能与上式相同

