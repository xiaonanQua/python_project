"""
实验tensorflow中的队列操作
"""
import tensorflow as tf

# 创建先进先出队列，同时指定队列可以保存的数据数量和数据类型
Queue = tf.queue.FIFOQueue(capacity=2, dtypes=tf.int32)

# 初始化队列
init_queue = Queue.enqueue_many(([1, 2],))

# 开启会话计算
with tf.compat.v1.Session() as sess:
    # 初始化队列
    sess.run(init_queue)
    # 出队
    print(sess.run(Queue.dequeue()))
    # 入队
    sess.run(Queue.enqueue([3]))
    # 出队所有
    for i in range(6):
        print(sess.run(Queue.dequeue()))