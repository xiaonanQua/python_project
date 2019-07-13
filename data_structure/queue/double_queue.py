"""
双向队列：允许队列两端中的任意一端都具备添加与删除的功能，
        且不论是左右两端的队列，队首域队尾指针都是朝队列中央来移动。
两种情况：(1)数据只能一端加入，两端进行删除。(2)两端进行数据加入，一端进行删除。
"""
from queue import queue_linked_list as queue  # 导入链表队列


class BothWayQueue(object):
    """
    双向队列，继承链表队列。重写入队、出队方法。
    """
    def __init__(self):
        self.front=self.rear=None  # 重写头节点、尾节点

    def dequeue_both(self, action):
        """
        实现两端出队
        :param action:选择在那一端出队
        :return:出队数据
        """
        if self.front is not None and action == 1:  # 若队头不为空，且选择的是队头出队，则出队
            if self.front == self.rear:  # 若队头等于队尾，则将队尾设置成None，队空
                self.rear = None
            # 保存出队数据，更新队头
            data = self.front.data
            self.front = self.front.next
            return data
        elif self.rear is not None and action == 2:  # 若队尾不为空，且选择的是队尾出队，则出队
            cur_node = self.front  # 保存当前队头指针值
            data = self.rear.data  # 保存出队数据
            # 遍历到队尾的前一个结点,且队头的下一个结点不为空
            while cur_node.next is not self.rear and cur_node.next is not None:
                cur_node = cur_node.next
            cur_node.next = None  # 将前一个节点的指向设置成None
            self.rear = cur_node  # 更新队尾节点
            # 若只剩下一个节点，则先取出数据。将队头和队尾都设置成None。
            if self.front == self.rear:
                self.front = None
                self.rear = None
            return data
        elif self.rear == self.front:
            print("队空，无法出队")
            return None

    def enqueue_both(self, data, action):
        """
        实现两端进队
        :param data:入队数据
        :param action: 选择在那一端入队
        :return:
        """
        # 构建新节点
        new_node = queue.Node()
        new_node.data = data
        # 若头结点和尾结点都为空，则队头、队尾都设为第一个结点
        if self.rear is None or self.front is None:
            self.front = new_node
            self.rear = new_node
        elif self.front is not None and action == 1:  # 从队头入队
            new_node.next = self.front
            self.front = new_node
        elif self.rear is not None and action == 2:  # 从队尾入队
            self.rear.next = new_node
            self.rear = new_node

    def enqueue(self, data):
        """
        入队
        :param data: 入队数据
        :return:
        """
        # 构建新结点
        new_node = queue.Node()
        new_node.data = data
        # 若头结点和尾结点都为空，则队头、队尾都设为第一个结点
        if self.front is None or self.rear is None:
            self.front = new_node
            self.rear = new_node
        else:  # 从队尾入队
            self.rear.next = new_node
            self.rear = new_node  # 更新队尾数据

    def dequeue(self):
        """
        出队
        :return:出队数据
        """
        if self.front is None:
            print("队空，无法出队")
            return None
        else:
            data = self.front.data  # 出队数据
            if self.front == self.rear:  # 若队头等于队尾，则取出数据，将队头、队尾设置成None
                self.front=self.rear=None
            else:
                self.front = self.front.next  # 更新队头
        return data

    def show_queue(self):
        """
        显示当前队列
        :return:
        """
        cur_node = self.front  # 当前节点
        while cur_node is not None:
            print("{}".format(cur_node.data))
            cur_node = cur_node.next


if __name__ == "__main__":
    both_queue = BothWayQueue()
    print("请选择操作：")
    print("1:入队，2：出队，3：查看队列信息,4:两端入队,5:两端出队")
    while True:
        input_op = int(input("请选择操作："))
        if input_op == 1:
            data = input("请输入需要入队的数据：")
            both_queue.enqueue(data)
        elif input_op == 2:
            data = both_queue.dequeue()
            print("出队数据：{}".format(data))
        elif input_op == 3:
            both_queue.show_queue()
        elif input_op == 4:
            data = input("请输入需要入队的数据：")
            choice = int(input("请选择1：队头，2：队尾入队:"))
            both_queue.enqueue_both(data, choice)
        elif input_op == 5:
            choice = int(input("请选择1：队头，2：队尾出队:"))
            data = both_queue.dequeue_both(choice)
            print("出队数据：{}".format(data))



