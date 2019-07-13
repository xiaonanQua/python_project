"""
任务：基于列表实现队列功能，包括入队，出队操作。
介绍：队列是一种抽象数据结构（Abstract Data Type）,遵循先进先出的原则（First IN First Out,FIFO）.
     在堆栈中只需一个指针top去指向栈顶，而在队列中需要front,rear指向对头和队尾。一般可以使用列表和链表来
     实现队列。
应用：计算机的模拟，CPU的作业调度，外围设备联机并发处理系统，图遍历的广度优先搜索法。
"""


class QueueList(object):
    def __init__(self, len):
        self.max_len = len  # 队列大最大长度
        self.queue = [None] * self.max_len  # 定义初始队列
        self.front = -1  # 定义队首指针
        self.rear = -1  # 定义队尾指针

    def enqueue(self, data):
        """
        入队,需要更新队尾指针
        :param data: 入队数据
        :return:
        """
        if self.rear == self.max_len -1:  # 队满
            print("队列已满，无法入队")
        else:
            self.rear += 1
        self.queue[self.rear] = data  # 入队

    def dequeue(self):
        """
        出队，需要更新队首指针
        :return: 返回出队的数据
        """
        if self.front == self.rear:  # 队空,需要添加一个空指针
            print("队空，无法出队")
        else:
            self.front = self.front + 1
            data = self.queue[self.front]
            return data

    def show_queue(self):
        cur = self.front + 1  # 当前队首指针
        while cur <= self.rear:
            print("{}".format(self.queue[cur]), end=' ')
            cur += 1
        print()


if __name__ == '__main__':
    queue_list = QueueList(5)  # 定义队列对象
    print("请选择操作：")
    print("1:入队，2：出队，3：查看队列信息")
    while True:
        input_op = int(input("请选择操作："))
        if input_op == 1:
            data = input("请输入需要入队的数据：")
            queue_list.enqueue(data)
        elif input_op == 2:
            data = queue_list.dequeue()
            print("出队数据：{}".format(data))
        elif input_op == 3:
            queue_list.show_queue()
