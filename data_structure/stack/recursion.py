"""
定义：递归算法，一个函数或子程序是由自身所定义或调用。
条件：（1）一个反复调用的执行过程。（2）一个跳出执行过程的出口。
应用情况：一般在选择结构和重复结构中可以用递归来进行编写。
三种递归情况：
（1）直接递归（Direct Recursion）：在递归函数中允许调用自身函数。
                            def fun(...):
                                if ...:
                                    fun(..)
(2)间接递归（Indirect Recursion）:在递归函数中调用其他递归函数，再从其他递归函数调用回原来的递归函数。
                    def fun1(...):          def fun2(..):
                        if ..:                  if ..:
                            fun2(..)                fun1(..)
(3)尾递归（Tail Recursion）：就是程序的最后一条指令为递归调用，即每次调用后再回到前一次调用后执行的
                            第一行指令就是return，不需要再进行任何计算工作。
"""


def factorial(i):
    """
    递归实现阶乘，考虑循环调用的过程和跳出循环的条件。
    :param i: 阶乘的数
    :return:
    """
    if i == 0:
        return 1  # 跳出递归的条件，从这里开始进行出栈的操作。
    else:
        result = i * factorial(i - 1)  # 反复执行的递归过程，也就是将获得结果不断入栈
    return result


def factorial_for(n):
    """
    用for循环计算0！～n！的和。
    :param n:
    :return:
    """
    sum = 0
    for i in range(0, n+1):  # n+1能够使i取到n的值
        x = 1
        for j in range(i, 0, -1):  # r[j]= i + -1*j
            x = x * j  # 阶乘
        sum = sum + x  # 累加
    return sum


def factorial_recursion(n):
    """
    用递归计算0!~n!的和
    :param n:
    :return:
    """
    sum = 0
    for i in range(n+1):
        sum = sum + factorial(i)  # 每个阶乘都使用递归操作
    return sum


def fab(n):
    """
    递归实现斐波那契数列：第0项为0，第一项为1,其他后续项的值是前两项的数值和。
    :param n: 数列的长度
    :return: 第n项数列的值
    """
    if n == 0:
        return 0
    elif n == 1 or n == 2:
        return 1
    else:
        return (fab(n-1) + fab(n-2))


def hanoi(n, p1, p2, p3):
    """
    用递归实现汉诺塔游戏，规则：(1)直径较小的盘子永远只能置于直径较大的盘子上。
                          (2)盘子可以任意地从任何一个木桩移动到其他的木桩上。
                          (3)每一次只能移动一个盘子，而且只能从最上面开始移动。
                          (4)总的移动次数最少，所有盘子都移动到木桩3上。
    通过不断的画图实践总结出步骤：
                          (1)将n-1个盘子从木桩1移动到木桩2
                          (2)将第n个最大的盘子从木桩1移动到木桩3上
                          (3)将n-1个盘子从木桩2移动到木桩3
                          以此进行递归操作
    :param n: 移动盘子的数目
    :param p1: 木桩1
    :param p2: 木桩2
    :param p3: 木桩3
    :return:
    """
    if n == 1:  # 递归出口
        print("盘子从{}移动到{}".format(p1, p3))
    else:
        hanoi(n-1, p1, p3, p2)
        print("盘子从{}移动到{}".format(p1, p3))
        hanoi(n-1, p2, p1, p3)


if __name__ == "__main__":
    result = factorial(5)  # 5的阶乘，递归实现
    print("递归阶乘结果：{}".format(result))
    result2 = factorial_for(5)  # 5的阶乘和
    print("阶乘和的结果：{}".format(result2))
    result3 = factorial_recursion(5)  # 递归实现5的阶乘和
    print("阶乘和递归结果：{}".format(result3))
    # 斐波那契每一项值
    print("斐波那契每一项值：")
    for i in range(10):
        print("fab({})={}".format(i, fab(i)))
    # 汉诺塔问题
    print("输出汉诺塔的移动步骤：")
    hanoi(4, 1, 2, 3)

