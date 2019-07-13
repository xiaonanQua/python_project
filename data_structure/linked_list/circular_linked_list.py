"""
任务：基于学生信息实现环形链表的创建、添加、删除、查看功能
原因：在单向链表中，维持表头信息很重要。因为一旦缺失链表的表头，整个链表都会遗失且浪费空间。
    若是将尾节点的指针指向链表头部而不是None，则整个链表会形成单方向的环形结构，也就不用担心表头丢失的问题。
    并且可以从任意节点遍历其他节点。
优点：可以从任何一个节点开始遍历所有节点,且回收整个链表花费的时间是固定的，和长度无关。
缺点：需要多出一个链接空间，插入一个节点需要改变两个链接
应用：环形链表通常应用于内存工作区和输入、输出缓冲区
"""


class student:
    """
    定义学生类，用于节点的生成
    """
    def __init__(self):
        self.num = 0  # 学号
        self.name = ''  # 姓名
        self.score = 0  # 成绩
        self.next = None  # 指针


def create_circular_list(data):
    """
    构建带有尾节点的环形链表
    :param data: 构建链表的数据
    :return: 带有尾节点的环形链表
    """
    head = student()  # 定义第一个结点
    cur = head  # 当前节点
    for i in range(len(data)):  # 根据数据的长度设置遍历次数
        # 添加信息
        cur.num = data[i][0]
        cur.name = data[i][1]
        cur.score = data[i][2]
        new_node = student()  # 构建尾节点
        cur.next = new_node  # 当前节点指向尾节点
        cur = new_node  # 更新当前结点
    cur.next = head  # 将尾节点指向第一个节点
    return head  # 返回头节点位置的链表


def show_circular_list(cir_link):
    """
    显示环形链表的信息
    :param cir_link: 环形链表
    :return: None
    """
    head = cir_link  # 头节点
    cur = head  # 当前节点设置在第一个结点上
    print("学号\t姓名\t成绩")
    while cur.next is not head:  # 循环遍历条件：当前结点是头结点则结束
        print("{}\t{}\t{}".format(cur.num, cur.name, cur.score))
        cur = cur.next


def insert_node(cir_link, new_node, posit=None):
    """
    在带有尾节点的环形链表中添加新节点。在特定的位置posit之后添加，默认在第一节点前添加。
    主要有两种插入的情况：（1）由于此链表是有尾节点，所以在第一个节点前插入时，则尾节点的指向需要更新一下。
    若是带有头节点的链表，在最后节点处插入时，则需要指向头节点处。
    （2）在除了头、尾节点处插入时，也就是正常的插入操作。
    :param cir_link: 环形链表
    :param new_node: 新结点
    :param posit: 添加的位置
    :return: 添加后的链表
    """
    head = cir_link  # 获得第一个节点
    cur = head  # 当前节点

    # 若插入位置为空，默认添加到第一个节点之前
    if posit is None:
        new_node.next = head  # 新节点指向第一个节点
        # 循环迭代到尾节点
        while cur.next is not head:
            cur = cur.next
        cur.next = new_node  # 尾节点指向新节点
        return new_node
    else:  # 若插入的位置不为空，则将新节点插入到posit节点之后
        while cur.next is not head:  # 判断当前节点指向的节点不是第一个节点
            # 判断插入节点的位置
            if cur.num == posit:
                new_node.next = cur.next  # 新节点指向当前节点的下一个节点
                cur.next = new_node  # 当前节点指向下一个节点
            cur = cur.next  # 指向下一个节点
        return head


def delete_node(cir_link, posit):
    """
    删除特定位置的节点。
    这和插入一样，也存在两种情况。（1）由于是带有尾节点的链表，若是删除首个节点，则尾节点需要重新指向。
    （2）若删除的节点是其他位置，则只需要将删除节点的前一个节点连接后面的节点即可。
    :param cir_link: 环形链表
    :param posit: 删除的位置
    :return: 删除后的链表
    """
    head = cir_link  # 第一个节点
    cur = head  # 当前节点
    # 进行节点的删除操作
    if posit == head.num:  # 判断删除位置是否是第一个节点
        # 将当前节点遍历到尾节点
        while cur.next is not head:
            cur = cur.next
        head = head.next  # 删除第一个节点
        cur.next = head  # 尾节点指向删除后链表的第一个节点
    else:  # 除了首节点，其他位置的删除
        pre = None  # 前一个节点
        while cur.next is not head:  # 遍历整个链表
            if cur.num == posit:  # 找到删除的位置
                pre.next = cur.next  # 前一个节点指向当前节点的下一个节点
            else:
                pre = cur  # 保存当前节点
            cur = cur.next  # 指向下一个节点
    return head


def concat_link(link1, link2):
    """
    连接两个环形链表。
    由于两个链表都带有尾结点，所以需要将link1的尾节点删除，前一个节点指向link2的首个节点。link2的尾
    节点需要指向link1的第一个结点
    :param link1:环形链表1
    :param link2: 环形链表2
    :return: 连接后的环形链表
    """
    head_node1 = link1  # 链表1的首个节点
    head_node2 = link2  # 链表2的首个节点
    pre_node = None  # 前一个节点
    cur_node = head_node1  # 链表1的当前节点
    # 链表1遍历到尾节点
    while cur_node.next is not head_node1:
        pre_node = cur_node  # 保存前一个节点
        cur_node = cur_node.next  # 指向下一个节点
    pre_node.next = head_node2  # 尾节点的前一个节点指向链表2的首个节点
    cur_node = head_node2  # 更换当前节点为链表2的首个节点
    # 循环遍历到链表2的尾节点
    while cur_node.next is not head_node2:
        cur_node = cur_node.next
    cur_node.next = head_node1  # 将链表2的尾节点指向链表1的首个节点
    return head_node1  # 返回链表1的表头


if __name__ == '__main__':
    data = [[1001, 'xiaonan', 294], [1002, 'xiaoshuai', 314], [1003, 'xiaopan', 358]]
    print("请选择你需要的操作：")
    print("1:创建循环链表,2:添加,3:删除,4:查看\n5:连接,-1:退出")
    while True:
        try:
            choice = int(input("请选择一个操作:"))
        except ValueError:
            choice = -1
        if choice is 1:
            head = create_circular_list(data)  # 创建环形链表
            print("创建成功！")
        elif choice is 2:
            # 定义新节点对象
            new_node = student()
            try:
                new_node.num = int(input("请输入学号："))
                new_node.name = input("请输入姓名：")
                new_node.score = int(input("请输入成绩："))
            except ValueError:
                print("输入的数据不合法")
            try:
                posit = int(input("请输入需要插入的位置（学号）："))
            except ValueError:
                posit = None
            # 插入新节点
            head = insert_node(head, new_node, posit)
            print("插入成功")
        elif choice is 3:
            try:
                posit = int(input("请输入删除的学号："))
            except ValueError:
                print("请重新输入")
            head = delete_node(head, posit)
            print("删除成功")
        elif choice is 4:
            show_circular_list(head)
        elif choice is 5:
            # 创建两个链表
            cir_link1 = create_circular_list(data)
            cir_link2 = create_circular_list(data)
            # 连接两个链表
            head = concat_link(cir_link1, cir_link2)
            print("连接成功")
        elif choice is -1:
            break