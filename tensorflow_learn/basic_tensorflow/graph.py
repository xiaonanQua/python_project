"""
定义：tensorflow的计算过程可以表示为一个计算图（Computation Graph），也称为有向图，在计算图上可以直观地看到数据的计算流程。
操作节点：计算图上的每一个操作可以看成一个节点，每一个节点都可以有任意的输入和输出。
边连接：若一个运算的输入是另一个运算的输出，则这两个运算存在依赖关系，两者之间通过边（Edge）相互连接。
依赖控制：有一类的边不存在数据流动，而是起着依赖控制的作用。也就是起始节点执行后再执行目标节点，来达到条件控制的目的。
需要注意：不同计算图上的变量不能共享，也就是不能调用其他计算图上的变量。对于自己定义的计算图通常设置成默认。
"""

import tensorflow as tf

# 使用Graph()函数创建一个计算图
g1 = tf.Graph()
with g1.as_default():  # 将定义的计算图设置成默认的计算图
    # 声明形状为[2],初始值为1的变量
    a = tf.get_variable(name='a', shape=[2], initializer=tf.ones_initializer())
    # 声明形状为[2],初始值为0的变量
    b = tf.get_variable(name='b', shape=[2], initializer=tf.zeros_initializer())

g2 = tf.Graph()  # 创建另一个计算图
with g2.as_default():
    # 声明形状为[2],初始值为0的变量
    a = tf.get_variable(name='a', shape=[2], initializer=tf.zeros_initializer())
    # 声明形状为[2],初始值为1的变量
    b = tf.get_variable(name='b', shape=[2], initializer=tf.ones_initializer())


# 使用上下文管理可以使创建的session不用手动关闭
with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()  # 初始化计算图中的所有变量
    with tf.variable_scope("",reuse=tf.AUTO_REUSE):
        print(sess.run(tf.get_variable('a')))
        print(sess.run(tf.get_variable('b')))

# 运行第二个计算图
with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("",reuse=True):
        print(sess.run(tf.get_variable('a')))
        print(sess.run(tf.get_variable('b')))
