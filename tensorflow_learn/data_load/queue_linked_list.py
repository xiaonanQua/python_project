"""
任务：使用链表实现队列操作，入队，出队，查看队列信息操作
除了定义类的节点外，还需要定义front,rear的指针。
"""


class Node:
    """
    定义队列节点
    """
    def __init__(self):
        self.image = None  # 图像矩阵数据
        self.label = None  # 标签数据
        self.next = None  # 指向下一个节点


class QueueLink:
    """
    定义队列链表对象，实现入队，出队，查看队列信息的功能
    """
    def __init__(self):
        self.front = self.rear = Node() # 定义首、尾结点，并指向头结点

    def enqueue(self, image, label):
        """
        将数据入队，更新队尾指针。链表没有堆满的判断。
        :param data: 入队数据
        :return: None
        """
        # 创建新节点对象，保存图像矩阵、标签数据
        new_node = Node()
        new_node.image = image
        new_node.label = label
        self.rear.next = new_node  # 新节点入队
        self.rear = new_node  # 更新当前尾节点

    def dequeue(self):
        """
        将数据出队，更新队首指针。有队空判断。
        :return: 返回出队的数据
        """
        if self.front == self.rear:  # 判断队空
            self.rear = self.front  # 队尾指向头结点
            print("队空，无法出队")
            return None
        else:  # 队不为空
            # 出队数据
            image = self.front.next.image
            label = self.front.next.label
            self.front.next = self.front.next.next  # 更新队头，头结点的指向
            return image, label

    def show_queue(self):
        """
        显示当前队列的信息
        :return: None
        """
        cur = self.front.next  # 当前队首
        while cur is not None:
            print("{}".format(cur.data), end=' ')
            cur = cur.next
        print()

    def is_empty(self):
        """
        判断队列是否为空
        :return:
        """
        if self.front.next is None:  # 队列为空
            return True
        else:
            return False

    def data_enqueue(self, images, labels):
        """
        将数据集数据进队
        :param images: 图像矩阵样本
        :param labels: 标签样本
        :return:
        """
        print("将数据压入队列中....")
        # 初始化队头，队尾
        self.front=self.rear=Node()
        # 获取样本总数
        num_examples = labels.shape[0]
        if num_examples != images.shape[0]:
            raise ValueError('图像矩阵样本和标签样本数量不一致')
        # 循环将数据进队
        for i in range(num_examples):
            self.enqueue(image=images[i], label=labels[i])
        print("将{}数量的样本数据压入队列".format(num_examples))


if __name__ == '__main__':
    queue_link = QueueLink()
    print("请选择操作：")
    print("1:入队，2：出队，3：查看队列信息，4：队列是否为空：")
    while True:
        input_op = int(input("请选择操作："))
        if input_op == 1:
            data = input("请输入需要入队的数据：")
            queue_link.enqueue(data)
        elif input_op == 2:
            data = queue_link.dequeue()
            print("出队数据：{}".format(data))
        elif input_op == 3:
            queue_link.show_queue()