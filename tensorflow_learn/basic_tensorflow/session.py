"""
session使用
"""
import tensorflow as tf

# 定义常量
x = tf.constant([2.0, 3.0], tf.float32, name='x')
y = tf.constant([3.0, 4.0], tf.float32, name='y')
# 计算两个常量之和
result = x + y
# 定义session
sess = tf.Session()
with sess.as_default():  # 将session设置成默认的
    # 两种相同计算方法
    print(sess.run(result))
    print(result.eval(session=sess))
    sess.close() # 关闭
# 使用InteractiveSession()方法将session已经设置成默认的，不需手动设置。
sess = tf.InteractiveSession()
print(sess.run(result))

# 第二种
with tf.Session(tf.ConfigProto(log_device_placement='true', allow_soft_placement='true')) as sess:
    tf.global_variables_initializer().run()
    print(sess.run(result))