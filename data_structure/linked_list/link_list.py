"""
基于员工信息实现单向链表的创建，添加、删除、查看、反转、连接功能
"""
import sys

class employee:
    """
    定义员工类，包含员工号、姓名、薪水、下一个结点
    """
    def __init__(self):
        self.num = 0
        self.salary = 0
        self.name = ''
        self.next = None


def delete_node(link_list, num):
    """
    根据结点删除链表中的结点
    :param link_list: 链表
    :param num: 删除的员工号
    :return: 删除结点后的链表
    """
    head = link_list # 获得链表的第一个结点
    count = 0  # 统计删除了次数
    pre = None  # 删除的前一个结点
    # 遍历出删除的结点
    while head is not None:
        if head.num == num:  # 找出要删除的结点
            print("删除的结点为：{}，{}，{}".format(head.num, head.name, head.salary))
            pre.next = head.next  # 断开删除的结点
            count = count + 1  # 记录删除
        pre = head  # 保存当前结点
        head = head.next  # 指向下一个结点
    if count == 0:
        print("不存在此要删除的结点")
    return link_list  # 返回链表


def create_linked_list():
    """
    创建带有头结点的单链表
    :return:单链表
    """
    # 初始化数据
    employee_list = [[1001, 'xiaonan', 32221], [1002, 'xiaoshuai', 32223], [1003, 'xiaopan', 33333]]
    print("创建的数据：")
    # 原来数据
    for i in range(len(employee_list)):
        for j in range(3):
            print(employee_list[i][j], end=' ')
        print('\n')
    # 构建头结点
    head = employee()
    if not head:
        print("Error,内存分配失败")
        sys.exit(0)
    # 当前结点，用于连接整个结点
    ptr = head
    # 构建链表
    for i in range(0, 3):
        new_node = employee()  # 构建新结点
        # 将列表中的值赋给对应新节点的属性中
        new_node.num = employee_list[i][0]
        new_node.name = employee_list[i][1]
        new_node.salary = employee_list[i][2]
        new_node.next = None
        ptr.next = new_node  # 连接新节点
        ptr = ptr.next  # 更新当前结点
    print("创建成功")
    return head


def show_linked_list(link_list):
    """
    输出链表中的数据
    :param link_list: 需要输出的链表数据
    :return: None
    """
    head = link_list.next  # 获取第一个结点
    print("员工信息：\n员工号\t姓名\t薪水")
    # 遍历链表
    while head is not None:
        print("{}\t{}\t{}".format(head.num, head.name, head.salary))
        head = head.next  # 指向下一个结点


def insert_node(linked_list, new_node, num_position=None):
    """
    在已经构建的链表中插入结点，在某个员工号之前进行插入
    :param linked_list: 单链表
    :param new_node: 插入的结点
    :param num_position: 在某个员工号位置之前进行插入,默认在链表的尾结点进行插入
    :return: 插入后的链表
    """
    head = linked_list  # 获得链表的表头
    pre = None  # 保存当前结点的前一个结点
    count = 0  #记录插入是否成功
    # 遍历整个链表，找出插入的结点
    while head is not None:
        if head.num == num_position:
            pre.next = new_node  # 将当前将结点的前一个结点指向新结点
            new_node.next = head  # 新结点指向当前结点
            count = count + 1  # 记录插入
        pre = head  # 保存当前结点记录
        head = head.next  # 指向下一个结点
    if count is 0:
        print("位置信息不正确，新节点插在尾结点之后")
        pre.next = new_node  # 尾结点后插入新节点
        print("插入成功")
    else:
        print("插入成功")
    return linked_list


def invert_linked_list(linked_list):
    """
    反转链表的顺序
    :param linked_list: 链表
    :return: 反转后的链表
    """
    head = linked_list.next  # 头结点
    pre = None  # 当前结点的前结点
    while head is not None:  # 训练遍历条件
        temp = pre  # 保存上一个结点
        pre = head  # 保存当前结点
        head = head.next  # 指向下一个结点,这里的位置不能跳动，因为pre.next重新指向上一结点，这会导致当前结点也会指向上一个结点，无法进行接下来的反转
        pre.next = temp  # 指向上一个结点，达到两个结点反转
    head = employee()  # 创建头结点
    head.next = pre  # 连接反转后的结点
    return head

def concat_list(link1, link2):
    """
    连接两个链表
    :param link1: 链表1
    :param link2: 链表2
    :return: 连接后的链表
    """
    head1 = link1  # 链表1的头结点
    head2 = link2  # 链表2的头结点
    while head1.next is not None:  # 判断next是否为空
        head1 = head1.next  # 遍历到尾结点
    head1.next = head2.next  # 将head2的第一个结点赋给head1
    return link1

if __name__ == '__main__':

    # 循环选择操作
    print("请选择你需要的操作：")
    print("1:插入,2:删除,3:查看,4:创建单链表,\n5:反转,6：连接两链表,-1:退出\n")
    while (True):
        input_op = int(input("请选择:"))
        if input_op == -1:
            break
        elif input_op == 1:
            # 输入新节点的信息
            new_node = employee()
            print("请输入新节点和插入位置，默认是尾结点插入：")
            new_node.num = int(input("请输入员工号："))
            new_node.name = input("请输入姓名：")
            new_node.salary = int(input("请输入薪水："))
            # 对插入位置进行异常捕获，若输入不合法则默认在尾结点后进行插入
            try:
                position = int(input("请输入插入位置"))
            except ValueError:
                position = None
            # 插入结点
            head=insert_node(head,new_node,position)
        elif input_op == 2:
            # 删除的员工号
            position = int(input("请输入删除的员工号"))
            # 删除数据
            head = delete_node(head)
        elif input_op == 3:
            # 显示链表信息
            show_linked_list(head)
        elif input_op == 4:
            # 创建链表信息
            head = create_linked_list()
        elif input_op == 5:
            # 反转链表的信息
            head = invert_linked_list(head)
        elif input_op == 6:
            # 创建两个链表
            head1 = create_linked_list()
            head2 = create_linked_list()
            head = concat_list(head1, head2)


