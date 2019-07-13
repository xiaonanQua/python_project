"""
任务：用列表实现堆栈的操作。
简介：堆栈（Stack）是一组相同数据类型的组合，有“先进先出”（First In Last Out，FILO）的特征。
    所有操作都在栈顶进行操作。
操作列表：创建、入栈、出栈、栈空、栈满。
优缺点：使用列表实现堆栈相对简单，堆栈本身大小是变动的，而列表大小只能事先规划和声明好，这样会浪费大量空间。
应用：(1)二叉树和森林的遍历，如前序遍历（PreOrder）中序遍历（InOrder）;
     (2)计算机中央处理单元（CPU）的中断处理（Interrupt Handling）;
     (3)图的深度优先搜索法（DFS）;
     (4)堆栈计算机（Stack Computer）,采用空地址指令，指令没有操作数，只有出栈（pop）和入栈（push）两个指令
     (5)递归算法的实现。进行递归循环时，也就是不断入栈操作。当递归进行返回时，依序从栈顶取出值，
        回到原来执行递归前的状态，进行执行下去;
     (6)算术表达式的转换和求值，如中序法转换成后序法;
     (7)调用子程序和返回处理操作。如在执行调用的子程序之前，需先将返回地址入栈，再进行子程序的处理。等到
        子程序处理完成时，从栈中弹出返回地址;
     (8)编译错误处理。当编译程序发生错误或者警告时，将程序所在的地址入栈，再显示错误或者警告信息;
"""

MAXSTACK = 100  # 定义最大栈容量
global stack
stack = [None] * MAXSTACK  # 堆栈的数组声明
print(stack)
top = -1  # 栈顶


def push(data):
    """
    入栈操作
    :param data: 入栈数据
    :return:
    """
    global top
    global MAXSTACK
    global stack
    if top > MAXSTACK - 1:
        print("栈已满，无法入栈")
    else:
        top = top + 1
        stack[top] = data


def isEmpty():
    """
    判断栈是否为空
    :return: boolean
    """
    if top == -1:
        return True
    else:
        return False


def pop():
    """
    出栈
    :return:
    """
    global top
    global stack
    if isEmpty():
        print("栈为空，无法出栈")
    else:
        print("出栈：{}".format(stack[top]))
        top = top - 1


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
    elif choice is 2:
        pop()
    elif choice is 3:
        break



