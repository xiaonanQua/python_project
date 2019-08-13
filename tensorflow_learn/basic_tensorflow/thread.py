"""
使用多线程处理输入的数据
使用Thread类创建5个线程,每个线程都只执行thread_op()函数操作,这些线程全部交由Coordinator类来管理.
下面创建了5个线程,也就是在同一个时间有5个线程被执行,这时需要考虑不同线程之间的协作了,无论5个线程中的哪一个线程发出
request_stop()操作请求,这5个线程的should_stop()函数返回值都被置为True.
"""

import tensorflow as tf
import numpy as np
import threading
import time


def thread_op(coordinator, thread_id):
    """
    线程操作
    :param coordinator: 协调器,用于协调多个线程并停止
    :param thread_id: 线程id
    :return:
    """
    # 若线程没有停止,进行操作
    while coordinator.should_stop() is False:
        # 若随机数小于0.1,则停止请求停止所有线程
        if np.random.rand() < 0.1:
            print("stoping thread id:{}".format(thread_id))
            coordinator.request_stop()
        else:
            # 打印当前线程
            print("working on thread id:{}".format(thread_id))
        # 每次循环暂停2s
        time.sleep(2)


def use_queuerunner():
    """
    使用QueueRunner类创建多个线程
    :return:
    """
    # 创建先进先出队列
    queue = tf.queue.FIFOQueue(capacity=100, dtypes=tf.float32)
    # 入队操作,每次入队10个随机数值
    enqueue = queue.enqueue([tf.random.normal([10])])

    # 使用QueueRunner创建10个线程进行队列入队操作
    qr = tf.train.QueueRunner(queue=queue, enqueue_ops=[enqueue]*10)
    # 将定义过的QueueRunner加入到计算图的GraphKeys.QUEUE_RUNERS集合中
    tf.train.add_queue_runner(qr)

    # 定义出队操作
    out = queue.dequeue()

    # 开启会话进行图计算
    with tf.compat.v1.Session() as sess:
        # 使用Coordinator来协同启动的线程
        coordinator = tf.train.Coordinator()

        # 调用start_queue_runners()函数来启动所有线程,并通过参数coord指定一个Coordinator来处理线程同步停止
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
        # 打印全部的结果
        for i in range(10):print(sess.run(out))
        # 请求所有线程停止,将Coordinator加入线程中等待所有线程退出
        coordinator.request_stop()
        coordinator.join(threads)


if __name__ == '__main__':
    # 使用python自带的threading库创建线程
    # 实例化Coordinator类
    coordinator = tf.train.Coordinator()
    # 使用线程类的线程函数创建5个线程
    threads = [threading.Thread(target=thread_op, args=(coordinator, i)) for i in range(5)]
    # 启动创建的5个线程
    for thread in threads:
        thread.start()
    # 将Coordinator类加入到线程并等待所有线程退出
    coordinator.join(threads)

    # 使用QueueRunner类创建线程
    use_queuerunner()