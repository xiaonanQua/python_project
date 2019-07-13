"""
测试文件
"""
import tensorflow as tf
import numpy as np

a = [0.1, 0.3, 0.5]
b = [0.7, 0.8, 0.4]

c = tf.argmax(a, 1)
d = tf.argmax(b, 1)

e = tf.equal(c, d)

with tf.Session() as sess:
    print(sess.run(c))
    print(sess.run(d))
    print(sess.run(e))

