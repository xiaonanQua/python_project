"""
算术表达式
"""


class Node:
    """
    定义链表节点对象。用于形成栈链表和队列链表
    """
    def __init__(self):
        self.data = None  # 保存数据
        self.next = None  # 指向下一个节点的指针


class Stack:
    """
    定义链表栈的对象，实现栈的入栈、出栈、栈空、栈满操作。
    """
    def __init__(self):
        self.top = Node()  # 栈顶

    def is_empty(self):
        """
        判断栈是否为空
        :return:
        """
        if self.top is None:  # 判断栈底是否为空
            return -1
        else:
            return 1

    def push(self, data):
        """
        入栈操作
        :param data:入栈数据
        :return:
        """
        # 新节点
        new_node = Node()
        new_node.data = data  # 存入数据
        # 判断栈是否为空
        if self.top.data is None:
            # 将第一个节点保存到栈底和栈顶中
            self.top = new_node  # 保存到栈顶中,用于链表的连接
        else:  # 其他节点的入栈
            new_node.next = self.top  # 新节点的入栈，让新节点的指针指向当前栈顶以此达到栈的先进后出效果
            self.top = new_node  # 更新当前栈顶

    def pop(self):
        """
        出栈操作
        :return: 出栈数据，若栈空则出None
        """
        if self.is_empty() == -1:
            print("栈空、无法出栈。")
            return None
        else:
            data = self.top.data  # 保存将要出栈的数据
            self.top = self.top.next  # 出栈，更新栈顶
            return data

    def show_stack(self):
        """
        查看当前栈对象里的数据
        :return:
        """
        while self.top is not None:  # 循环遍历条件，栈顶不为空
            print("{}".format(self.top.data), end=' ')
            self.top = self.top.next  # 指向下一个节点


class Queue:
    """
    定义队列对象，实现入队，出队，查看队列信息，队空操作。
    """
    def __init__(self):
        self.front = Node()  # 队首
        self.rear = Node()  # 队尾

    def is_empty(self):
        """
        判断队列是否为空
        :return: -1:代表为空，1代表不为空
        """
        if self.front is None:  # 判断队首是否为空
            return -1
        else:
            return 1

    def enqueue(self, data):
        """
        入队操作,首先将传入的数据转换成新的节点，然后接入队列中
        :param data:入队的数据
        :return:
        """
        # 构建新节点
        new_node = Node()
        new_node.data = data

        # 让新节点入队
        if self.front.data is None:  # 判断队列是否为空
            self.rear = new_node  # 新节点赋值给为节点
            self.front = self.rear  # 将尾节点赋值给首节点，使首节点能够连起来
        else:
            self.rear.next = new_node  # 队尾指针指向新节点
            self.rear = new_node  # 更新队尾

    def dequeue(self):
        """
        出队操作,按照先进先出的规则。
        :return: 出队节点的数据
        """
        if self.is_empty() == -1:  # 判断队列是否为空
            print("队空，无法出队！")
        else:
            data = self.front.data  # 保存数据
            self.front = self.front.next  # 出队，更新当前队
        return data

    def show_queue(self):
        """
        显示队列信息
        :return:
        """
        while self.is_empty() != -1:
            print("{}".format(self.front.data), end='')
            self.front = self.front.next

    def back_origin(self):
        """
        返回队列的队首
        :return:
        """
        pass


def judge_op(op):
    """
    用于判断是操作数或者是运算符。
    :param op:操作数或者运算符
    :return:
    """
    if op == '(' or op == '^' or op == '*' or op == '/' or op == '+' or op == '-':
        return 1
    else:
        return -1


def judge_op_priority(stack_node, infix_node):
    """
    判断堆栈里的运算符和中序表达式队列中的运算符的优先级。
    :param stack_node: 堆栈里的运算符节点
    :param infix_node: 中序表达式中的运算符节点
    :return: 比较结果,1:表示表达式优先级比栈顶高,0:表示表达式优先级低于或者等于栈顶
    """
    if stack_node is None:
        stack_op = None
    else:
        stack_op = stack_node.data
    if infix_node is None:
        infix_op = None
    else:
        infix_op = infix_node.data
    print(stack_op, infix_op)
    infix_priority = [''] * 9  # 中序表达式运算符优先级
    stack_priority = [''] * 8  # 堆栈中运算符优先级
    index_i = index_s = 0 # 中序表达式和堆栈的索引

    # 初始化中序表达式运算符优先级
    infix_priority[0] = None; infix_priority[1] = ')'
    infix_priority[2] = '+'; infix_priority[3] = '-'
    infix_priority[4] = '*'; infix_priority[5] = '/'
    infix_priority[6] = '^'; infix_priority[7] = ' '
    infix_priority[8] = '('
    # 堆栈中运算符优先级
    stack_priority[0] = None; stack_priority[1] = '('
    stack_priority[2] = '+'; stack_priority[3] = '-'
    stack_priority[4] = '*'; stack_priority[5] = '/'
    stack_priority[6] = '^'; stack_priority[7] = ' '

    # 计算堆栈运算符在堆栈中的优先级
    while stack_priority[index_s] != stack_op:
        index_s += 1
    # 计算中序运算符在中序中的优先级
    while stack_priority[index_i] != infix_op:
        index_i += 1

    # 判断两个运算符的优先级
    if int(index_i/2) > int(index_s/2):
        return 1
    else:
        return 0


def infix_to_postfix():
    """
    中序表示法转化成后序表示法。
    规则：(1)使用队列结构保存中序表达式，再按照队列先进先出的规则依次读取每个字符（token）;
         (2)使用队列结构保存后序表达式，如果从中序表达式中读取的是操作数，则直接压入后序表达式的队列。
         (3)使用堆栈结构保存运算符。当从中序表达式中读取的运算符优先级比堆栈中栈顶运算符优先级要高，则直接
            入栈。若读取的运算符优先级低于或者等于栈顶运算符优先级，就把栈顶运算符出栈，直到栈顶运算符的优先级
            低于读取的运算符优先级或者栈空，才能让读取的运算符入栈。
         (4)当从中序表达式中读取的运算符是')'时,则弹出堆栈中的运算符，直到弹出'('为止。在堆栈中’（‘的优先级
            最小，在栈外则最大。
    :return:
    """
    infix = Queue()  # 定义中序表达式队列对象
    input_exper = str(input("输入表达式："))
    # 将表达式数据保存到中序表达式队列中
    for i in range(len(input_exper)):
        infix.enqueue(input_exper[i])
    # 显示中序表达式
    # infix.show_queue()
    postfix = Queue()  # 定义后序表达式队列对象
    stack_op = Stack()  # 定义操作符堆栈
    while infix.front is not None:  # 循环迭代条件，判断中序表达式队列是否为空
        if infix.front.data == ')':  # 若队头数据是'(',则弹出堆栈中的运算符，直到遇到’）‘为止。
            while stack_op.top is not None and stack_op.top.data != '(':  # 遍历堆栈直到遇到’）‘
                postfix.enqueue(stack_op.pop())  # 将从堆栈中弹出的运算符入后序表达式队列中
            stack_op.pop()  # 将’('出栈
        elif judge_op(infix.front.data) == 1:  # 若是运算符，则对比中序运算符和栈顶运算符
            if stack_op.is_empty() == -1:  # 若堆栈中为空，则将运算符入栈
                stack_op.push(infix.dequeue())
            else:
                # 若表达式运算符比栈顶低，且堆栈不为空，将运算符从栈顶退栈，压入后序队列中
                while judge_op_priority(stack_op.top, infix.front) == 0 or stack_op.top is not None:
                    postfix.enqueue(stack_op.pop())
        else:  # 若读取的操作数，则直接压入队列中
            postfix.enqueue(infix.dequeue())
    while stack_op.top is not None:  # 若堆栈不为空，弹出运算符压入后序队列
        postfix.enqueue(stack_op.pop())
    print("shuchu")
    postfix.show_queue()


    #     # 判断读取数据是操作数还是运算符，1：代表运算符，-1：代表操作数。
    #     if judge_op(infix.front.data) == 1 and infix.front.data != ')':
    #         if stack_op.top is None:
    #             stack_op.push(infix.front.data)
    #             infix.dequeue()
    #             continue
    #         print(infix.front.data)
    #         # 判断表达式运算符和堆栈栈顶运算符的优先级，若高于栈中运算符则直接入栈
    #         if judge_op_priority(stack_op.top.data, infix.front.data) == 1:
    #             stack_op.push(infix.front.data)  # 入栈
    #             infix.dequeue()  # 出队
    #         else:  # 低于或者等于栈中运算符
    #             # 弹出栈顶运算符，压入后序表达式队列中，直到栈外的运算符优先级高于栈顶运算符
    #             while judge_op_priority(stack_op.top.data, infix.front.data) == 0:
    #                 postfix.enqueue(stack_op.pop())
    #             # 将栈外运算符压入栈顶，中序队列出队压入栈顶运算符
    #             stack_op.push(infix.front.data)
    #             infix.dequeue()
    #     # 判断中序队列中迭代到‘）’的情况
    #     elif judge_op(infix.front.data) ==1 and infix.front.data == ')':
    #         # 弹出栈顶运算符直到遇到‘(’
    #         while stack_op.top.data is not '(':
    #             postfix.enqueue(stack_op.pop())  # 将栈顶运算符压入后序表达式
    #         stack_op.pop()  # 删除堆栈中‘(’
    #     else:
    #         # 将中序队列中操作数压入后序队列中a
    #         postfix.enqueue(infix.front.data)
    #         infix.dequeue()
    # print("{},{},{}".format(infix.show_queue(), stack_op.show_stack(), postfix.show_queue()))


if __name__ == "__main__":
    infix_to_postfix()