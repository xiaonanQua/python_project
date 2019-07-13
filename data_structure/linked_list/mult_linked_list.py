"""
实现多项式链表功能，并实现两个多项式进行相加的方法
"""

class experssion:
    """
    定义多项式表达式结点
    """
    def __init__(self):
        self.coef = 0  # 系数
        self.exp = 0  # 指数
        self.next = None  # 指针


def create_link(factors):
    """
    创建多项式链表
    :param factors: 创建的参数，多项式的系数和指数
    :return: 链表
    """
    head = experssion()  # 头结点
    cur = head  # 当前结点
    # 循环遍历，将数据添加到链表中
    for i in range(len(factors)):
        # 生成新结点
        new_node = experssion()
        new_node.coef = factors[i][0]
        new_node.exp = factors[i][1]
        # 当前结点指向新结点
        cur.next = new_node
        # 更新当前结点
        cur = new_node
    return head  # 返回链表


def show_link(linked_list):
    """
    查看链表信息
    :param linked_list: 链表
    :return: None
    """
    head = linked_list.next  # 第一个结点
    print("表达式：", end='')
    while head is not None:
        if head.coef is not 0 and head.exp is not 0:
            print("{}X^{}+".format(head.coef, head.exp), end='')
        elif head.coef is not 0 and head.exp is 0:
            print("{}".format(head.coef))
        head = head.next


def plus_link(link1, link2):
    """
    多项式相加，即链表的连接
    :param link1: 多项式链表1
    :param link2: 多项式链表2
    :return: 相加后的链表
    """
    # 获得两个链表的头结点
    exp1 = link1.next
    exp2 = link2.next
    factor = []  # 保存系数、指数数据
    i = 0  # 用于更新列表索引
    # 循环遍历，进行多项式相加
    while exp1 is not None:
        if exp1.exp == exp2.exp:  # 两个表达式的指数相等，进行系数相加并指向下一个结点
            factor.append([exp1.coef + exp2.coef,exp1.exp])  # 将数据添加到列表中
            exp1 = exp1.next  # exp1指向下一个结点
            exp2 = exp2.next  # exp2指向下一个结点
            i = i + 1  # 更新索引
        elif exp1.exp > exp2.exp:  # 若exp1当前的指数比exp2的指数大，则说明exp2后续没有比exp1大的指数
            factor.append([exp1.coef,exp1.exp])  # 将数据添加到列表中
            exp1 = exp1.next  # exp1指向下个结点
            i = i + 1  # 更新索引
        elif exp1.exp < exp2.exp:  # 若exp1当前的指数比exp2的指数小，则说明后续exp2有可能比exp1相等的指数，所以应保存exp2数据并迭代exp2
            factor.append([exp2.coef, exp2.exp])  # 将数据添加到列表中
            exp2 = exp2.next  # exp2指向下个结点
            i = i + 1  # 更新索引
    print(factor)
    return create_link(factor)  # 根据系数创建表达式


if __name__ == "__main__":
    # 两个表达式的系数列表
    factor1 = [[3, 3], [4, 1], [2, 0]]
    factor2 = [[6, 3], [8, 2], [6, 1], [9, 0]]
    # 生成两个多项式,相加后的多项式
    experssion1 = create_link(factor1)
    experssion2 = create_link(factor2)
    plus_experssion = plus_link(experssion1, experssion2)
    # 查看表达式信息
    print("A:", end='')
    show_link(experssion1)
    print("B:", end='')
    show_link(experssion2)
    print("C:", end='')
    show_link(plus_experssion)

