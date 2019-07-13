"""
placeholder机制，用于在会话运行时动态提供数据。
目的：解决如何在有限的输入节点上实现高效地接受大量数据的问题。
"""
import tensorflow as tf

# 定义占位符并计算
x = tf.placeholder(tf.float32, shape=(2), name='x')
y = tf.placeholder(tf.float32, (2), name='y')
result = x + y

# 定义另一组占位符并计算
a = tf.placeholder(tf.float32, shape=[2], name='a')
b = tf.placeholder(tf.float32, shape=(3,2), name='b')
result2 = a + b
with tf.Session() as sess:
    # 执行计算图，fetches:计算后的结果，feed_dict:输入的数据
    sess.run(fetches=result, feed_dict={x:[1.0, 2.0], y:[3.0, 4.0]})
    print(result)
    print(sess.run(fetches=result2, feed_dict={a:[1.0, 2.0], b:[[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]}))
    print(result2)