"""
简单的前馈神经网络，2层隐藏层
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 读取mnist对象
mnist = input_data.read_data_sets("/home/xiaonan/Dataset/mnist/", one_hot=True)

# 数据、标签占位符
x = tf.placeholder(tf.float32, shape=[None, 784], name='x_input')
y = tf.placeholder(tf.float32, shape=[None, 10], name='y_input')

# 设置超参数
batch_size = 200
learning_rate = 0.1
learning_rate_decay = 0.999
max_steps = 30000  # 最大训练步数

# 设置权重、偏置项、当前训练步骤参数
weight1 = tf.Variable(initial_value=tf.random_normal([784, 500], stddev=0.1, dtype=tf.float32))
bias1 = tf.Variable(tf.constant(0.1, shape=[500]))
weight2 = tf.Variable(initial_value=tf.random_normal([500, 10], stddev=0.1, dtype=tf.float32))
bias2 = tf.Variable(tf.constant(0.1, shape=[10]))
current_train_step = tf.Variable(0, trainable=False)  # 当前训练步骤，trainable=False表示变量不加入计算图中

# 网络结构
layer1 = tf.nn.relu(tf.matmul(x, weight1) + bias1)
y_ = tf.matmul(layer1, weight2) + bias2

# 使用交叉熵作为Loss函数衡量预测标签与真实标签之间的差距
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_, labels=tf.argmax(y, 1))
# 使用梯度下降进行loss的优化
train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
# 统计正确的预测，为精确度的计算
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 进行网络的训练
with tf.Session() as sess:
    # 初始化全局变量
    tf.global_variables_initializer().run()
    # 准备验证、测试数据
    validate_feed ={x:mnist.validation.images, y:mnist.validation.labels}
    test_feed = {x:mnist.test.images, y:mnist.test.labels}

    # 进行训练
    for i in range(1, max_steps+1):
        if i % 1000 == 0:  # 若进行一千次训练，则验证训练精度
            validate_accuracy = sess.run(accuracy, feed_dict=validate_feed)
            print('训练{}轮,验证精度：{}'.format(i, validate_accuracy*100))
        xs, ys = mnist.train.next_batch(batch_size=batch_size)
        sess.run(train_op, feed_dict={x: xs, y: ys})

    # 测试精度
    test_accuracy = sess.run(accuracy, feed_dict=test_feed)
    print("测试精度：{}".format(test_accuracy*100))
