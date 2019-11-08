"""
Variale()函数是Variable类的构造函数，在初始化这个变量时，可以使用随机数来初始化这个变量。
常见产生随机数的方法有：
1、正太分布：tf.random_normal(shape,mean,stddev,seed,name)
2、正太分布，若随机出来的值偏离平均值超过两个标准差，将重新随机分配。tf.truncated_normal(shape,mean,stddev,seed,name)
3、平均分布：tf.random_uniform(shape,minval,maxval,dtype,seed,name)
4、Gamma分布：tf.random_gamma(shape,alpha,beta,dtype,seed,name)

另外，也可以使用常数来初始化变量。
常见产生常数的方法：
1、zeros():产生全0的数组
2、ones()：产生全1的数组
3、fill():产生一个值为给定数字的数组
4、constant():产生一个给定值的常量

注意：定义变量Variable()函数,需要调用其initializer属性进行初始化，但是随着变量之多且每个变量都需要初始化，
所有可以使用tf.global_variables_initializer().run()进行初始化且自动处理变量之间的依赖关系。
"""

import tensorflow as tf

# 定义常量，用作输入
x = tf.constant([[1.0, 2.0]], tf.float32, name='x')
# 定义变量，当作参数。设置随机种子，可以保证每次得到的结果都是一样的
w1 = tf.Variable(tf.truncated_normal([2, 3], stddev=1, seed=2), name='w1')
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))
# tf.assign(w1, w2, validate_shape=False)
# 进行矩阵相乘
y = tf.matmul(x,w1)
y2 = tf.matmul(y, w2)

# 使用get_variable()创建变量
# w3 = tf.get_variable('w1', shape=(2,3), initializer=tf.constant_initializer(1.0))

# 使用上下文环境设置变量范围,使用variable_scope()变量空间管理变量，对于使用Variable()和
# get_variable()函数创建的变量，都会在生成的变量名称前添加变量空间名称前缀。
with tf.variable_scope('one'):
    w4 = tf.get_variable('w1', (2,3), initializer=tf.truncated_normal_initializer())
    print(w4)

# 变量空间的嵌套
with tf.variable_scope('two'):
    # 使用get_variable_scope()函数获取当前的变量空间
    print(tf.get_variable_scope().reuse)
    with tf.variable_scope('three'):
        w5 = tf.get_variable('w1', shape=[2, 3], dtype=tf.float32, initializer=tf.random_normal_initializer())
        print(w5.name)
    w6 = tf.get_variable('w1', [2,3], initializer=tf.constant_initializer(1.0))
    print(w6.name)

# name_scope()的使用，在其内部使用get_variable()函数生成变量名称不会添加变量空间名称，而相反
# 使用Variable()函数会添加变量空间名称前缀。
with tf.name_scope('four'):
    w7 = tf.get_variable('w1', shape=[2, 3])
    print(w7.name)
    w8 = tf.Variable(tf.random_uniform([2, 3], minval=2, maxval=6), name='w8')
    print(w8.name)

with tf.Session(confi) as sess:
    # 初始化所有变量
    tf.global_variables_initializer().run()
    print(sess.run(y2))  # 因为随机种子的设置，每次结果都相同
#    print(sess.run(w3))
    print(sess.run(w4))

