"""
循环队列,使用列表实现循环队列。
"""
from queue import queue_list as queue


class CircularQueue(queue.QueueList):
    """
    循环队列继承父类列表队列，并重写队头指针，队尾指针的属性，入队、出队的方法。
    类继承构造函数，调用顺序是：先调用子类的构造函数，再调用父类的构造函数。
    """
    def __init__(self, len):
        queue.QueueList.__init__(self,len)
        self.max_len = len
        self.front = (self.front+1) % self.max_len  # 重写队头指针,self.front=0
        self.rear = (self.rear+1) % self.max_len  # 重写队尾指针,self.rear=0

    def enqueue(self, data):
        """
        重写进队方法，实现循环队列。
        :param data:进队数据
        :return:
        """
        if (self.rear+1) % self.max_len == self.front:  # 队满条件,牺牲一个单元来判断
            print("队满，无法入队")
            return None
        else:
            self.queue[self.rear] = data  # 保存数据
            self.rear = (self.rear + 1) % self.max_len  # 指向下一个空值的位置

    def dequeue(self):
        """
        重写出队方法,实现循环队列
        :return: 出队数据
        """
        if self.front == self.rear:  # 判断队空
            print("队空，无法出队")
            return None
        else:
            data = self.queue[self.front]  # 出队数据
            self.front = (self.front+1) % self.max_len  # 指向下一个位置
        return data

    def show_queue(self):
        """
        重写显示队列方法
        :return:
        """
        cur = self.front  # 保存当前队头指针
        while cur != self.rear:  # 循环判断条件，队头与队尾不重合
            print("{}".format(self.queue[cur]), end='')
            cur = (cur+1) % self.max_len


if __name__ == '__main__':
    queue_list = CircularQueue(5)  # 定义队列对象
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