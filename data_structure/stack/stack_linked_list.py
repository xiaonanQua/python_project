"""
用链表实现堆栈功能。
优缺点：使用链表可以动态改变链表的长度，但是设计的算法较为复杂。
"""


class node:
    def __init__(self):
        self.data = 0  # 存放的数据
        self.next = None  # 指向下一个节点


# 声明栈顶并初始化
global top
top = None


def isEmpty():
    """
    判断是否为空
    :return:
    """
    if top is None:
        return 1
    else:
        return 0


def push(data):
    """
    入栈
    :param data: 入栈节点
    :return:
    """
    global top
    new_node = node()  # 构建新节点
    new_node.data = data  # 数据
    new_node.next = top  # 新节点指向栈顶元素
    top = new_node  # 新节点成为堆栈的顶端


def pop():
    """
    出栈
    :return:
    """
    global top
    if isEmpty() == 1:
        print("栈为空,无法出栈")
        return -1
    else:
        cur_node = top  # 保存栈顶节点
        top = top.next  # 栈顶指针指向下一个节点
        temp = cur_node.data  # 提取出栈的数据
        return temp


# 主程序
print("请选择操作：")
print("1:入栈，2：出栈，3：退出")
while True:
    try:
        choice = int(input("请选择操作"))
    except ValueError:
        print("请重新输入")
    if choice is 1:
        data = input("输入值：")
        push(data)
        print("入栈成功")
    elif choice is 2:
        data = pop()
        print("出栈数据：{}".format(data))
    elif choice is 3:
        break

