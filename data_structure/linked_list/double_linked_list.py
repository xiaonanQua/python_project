"""
任务：基于员工信息构建双向链表、添加、删除、查看等功能
优点：每个节点中都有两个指针，分别指向当前节点的前后节点，能够快速找到前后节点。同时可以从任意一个节点出发查找其他
    节点，不需要经过反转或对比节点等处理。
缺点：由于双向链表每个节点有两个指针，所以在添加和删除时需要花费时间调整指针。每个节点含有两个指针，浪费空间
实现：双向链表的数据结构包含：左指针、中间数据、右指针。双向链表可以构成环形双向链表，也可以不构成。
    若实现环形双向链表，通常需要添加一个头指针，其数据字段不存放数据，左指针指向链表的尾节点，右指针指向链表
    的首节点。
"""

class employee:
    """
    定义员工类，构建员工双向节点
    """
    def __init__(self):
        self.num = 0  # 员工号
        self.name = ''  # 姓名
        self.salary = 0  # 薪水
        self.pre = None  # 前指针
        self.next = None  # 后指针


def create_double_link(data):
    """
    构建带有头指针的环形双向链表
    :param data: 构建链表的数据
    :return: 构建好的环形双向链表
    """
    head_node = employee()  # 头指针
    cur_node = head_node  # 当前节点

    # 循环数据、构建环形双向链表
    for i in range(len(data)):
        # 构建新节点的信息
        new_node = employee()
        new_node.num = data[i][0]
        new_node.name = data[i][1]
        new_node.salary = data[i][2]
        # 新节点的前节点指针指向当前节点
        new_node.pre = cur_node
        # 当前节点的后节点指针指向新节点，即将节点连接到链表中
        cur_node.next = new_node
        # 更新当前节点
        cur_node = new_node
    # 尾节点的后节点指针指向头指针
    cur_node.next = head_node
    # 头指针的前节点指针指向尾节点
    head_node.pre = cur_node
    return head_node  # 返回带有头指针的链表


def insert_node(double_link, new_node, posit):
    """
    在环形双向链表中添加节点。
    插入操作：在某个节点之后进行插入或者某个节点之前插入。
    :param double_link: 环形双向链表
    :param new_node: 新节点
    :param posit: 插入的位置
    :return: 插入后的链表
    """
    head_node = double_link  # 头指针
    cur_node = head_node.next  # 当前节点,首节点
    # 循环遍历整个链表
    while cur_node is not head_node:
        # 判断当前节点是否为尾节点
        if cur_node.num == posit:
            # 在当前节点后插入操作（从后往前）
            # cur_node.next.pre = new_node  # 当前节点的后一个节点的前指针指向新节点
            # new_node.next = cur_node.next  # 新节点的后指针指向当前节点的后一个节点
            # new_node.pre = cur_node  # 新节点的前指针指向当前节点
            # cur_node.next = new_node  # 当前节点的后指针指向新节点
            # 在当期节点前插入操作（从前往后）
            cur_node.pre.next = new_node  # 当前节点的前一个节点的后指针指向新节点
            new_node.pre = cur_node.pre  # 新节点的前指针指向当前节点的前一个节点
            new_node.next = cur_node  # 新节点的后指针指向当前节点
            cur_node.pre = new_node  # 当前节点的前指针指向新节点
        cur_node = cur_node.next  # 更新当前节点
    return head_node


def delete_node(link, posit):
    """
    删除posit位置的节点。
    :param link: 环形双向链表
    :param posit: 删除位置（员工号）
    :return: 删除后的链表
    """
    head_node = link  # 头指针
    cur_node = head_node.next  # 当前节点，首节点
    # 循环遍历整个链表
    while cur_node.next is not head_node:
        if  cur_node.num == posit:  # 删除条件
            cur_node.pre.next = cur_node.next  # 当前节点的前一个节点的后指针指向当前节点的后一个节点
            cur_node.next.pre = cur_node.pre  # 当前节点的后一个节点的前指针指向当前节点的前一个节点
        cur_node = cur_node.next  # 更新当前节点
    return head_node  # 得到删除后的链表

def show_double_link(link):
    """
    遍历环形双向链表
    :param link: 环形双向链表
    :return: None
    """
    head_node = link  # 头指针
    cur_node = head_node.next  # 首节点
    # 循环遍历整个链表
    print("员工号\t姓名\t分数")
    while cur_node is not head_node:
        print("{}\t{}\t{}".format(cur_node.num, cur_node.name, cur_node.salary))
        cur_node = cur_node.next


if __name__ == "__main__":
    # 定义员工数据
    data = [[1001, 'xiaonan', 294], [1002, 'xiaoshuai', 378], [1003, 'xiaopan', 399]]
    print("请选择操作：")
    print("1:创建环形双向链表,2：添加,3：删除,4：查看\n-1:退出")
    # 循环选择操作
    while True:
        try:
            choice = int(input("请选择一个操作："))
        except ValueError:
            print("请输入正确的值")
        if choice is 1:
            # 创建环形双向链表
            head_node = create_double_link(data)
            print("创建成功")
        elif choice is 2:
            # 输入新节点的信息
            new_node = employee()
            print("请输入新节点和插入位置：")
            new_node.num = int(input("请输入员工号："))
            new_node.name = input("请输入姓名：")
            new_node.salary = int(input("请输入薪水："))
            # 对插入位置进行异常捕获
            try:
                position = int(input("请输入插入位置："))
            except ValueError:
                position = None
            # 插入结点
            head_node = insert_node(head_node, new_node, position)
            print("添加成功")
        elif choice is 3:
            # 对插入位置进行异常捕获
            try:
                position = int(input("请输入删除位置："))
            except ValueError:
                position = None
            head_node = delete_node(head_node, position)
            print("删除成功")
        elif choice is 4:
            # 查看链表
            show_double_link(head_node)
        elif choice is -1:
            break

