"""
任务：基于堆栈思想实现老鼠走迷宫问题。
走迷宫规则：(1)一次只能走一格
         (2)遇到墙无法前进，需退一步寻找其他路是否可走
         (3)走过的路不再走第二次
实现细节：(1)使用二维数组设计迷宫规则：MAZE[row][col],row*col=10*12
                                (1)MAZE[i][j]=1,表示墙壁，无法通过
                                (2)MAZE[i][j]=0,表示通道，可以通过
                                (3)MAZE[i][j]=2,表示老鼠走过的路径
                                (4)MAZE[1][1]表示入口，MAZE[8][10]表示出口
        (2)用MAZE[x][y]表示老鼠的位置,老鼠有四个方向,东:MAZE[x][y+1],西：MAZE[x][y-1],
           南：MAZE[x+1][y],北：MAZE[x-1][y]
        (3)走迷宫的流程：可以使用链表来记录走过的位置,并设置走过位置的数组信息为2,同时将位置信息压入堆栈中
            才可以进行往前走。若走到死胡同且没有走到终点，则向后退一格直到回到岔路口后再选择其他路。由于每次
            新加入的位置必定会在堆栈的顶端，所以栈顶顶端所指的位置就是老鼠的位置，重复判断老鼠的位置直到找到出口。
"""
import time


class Node:
    """
    定义老鼠所处位置信息的类，也是位置节点
    """
    def __init__(self, x, y):
        self.x = x  # x轴
        self.y = y  # y轴
        self.next = None  # 指向下一个节点


class PathRecord:
    """
    定义路径信息的堆栈，保存老鼠走过的路径。
    入栈：走过的位置信息压入栈中
    出栈：走到死路时，进行后退操作。从栈中弹出上次走过的位置信息。
    """
    def __init__(self):
        self.tail = None  # 栈底节点，用于栈遍历
        self.top = None  # 栈顶节点，用于弹出位置信息

    def is_empty(self):
        return self.tail == None  # 判断栈是否为空

    def push(self, x, y):
        """
        将老鼠走的位置信息进行入栈，位置信息使用节点进行保存。
        :param x: x轴
        :param y: y轴
        :return:
        """
        new_node = Node(x, y)  # 行走的新位置，转化成链表节点形式
        if self.tail is None:  # 第一次行走
            # 保存老鼠第一次行走的位置
            self.tail = new_node
            self.top = new_node
        else:
            self.top.next = new_node  # 当前栈顶的指针指向新节点，将节点入栈
            self.top = new_node  # 更新栈顶

    def pop(self):
        """
        将老鼠走过的位置信息进行出栈。
        :return:
        """
        if self.tail is None:
            print("已经回到起点了")
            return
        pre_node = self.tail  # 前节点，首先从栈底开始。用于保存栈顶的前一个节点。
        # 循环遍历到栈顶的前一个位置
        while pre_node.next is not self.top:
            pre_node = pre_node.next
        # 将栈顶指针指向的节点赋给前一个节点指针的指向，也就是将pre_node.next=None.
        pre_node.next = self.top.next
        self.top = pre_node  # 更新栈顶，将原先栈顶节点出栈


def judge_exit(maze,x,y,ex,ey):
    """
    判断是否已经到达出口。
    :param maze: 迷宫数组
    :param x: 当前x轴
    :param y: 当前y轴
    :param ex: 出口x轴
    :param ey: 出口y轴
    :return: 0:表示未到达，1:表示已到达
    """
    if x == ex and y == ey:
        if maze[x-1][y]==1 or maze[x+1][y]==1 or maze[x][y-1]==1 or maze[x][y+1]==2:
            return 1
        if maze[x-1][y]==1 or maze[x+1][y]==1 or maze[x][y-1]==2 or maze[x][y+1]==1:
            return 1
        if maze[x-1][y]==1 or maze[x+1][y]==2 or maze[x][y-1]==1 or maze[x][y+1]==1:
            return 1
        if maze[x-1][y]==2 or maze[x+1][y]==1 or maze[x][y-1]==1 or maze[x][y+1]==1:
            return 1
    return 0


def maze():
    """
    主程序,绘制迷宫图，走迷宫。
    :return:
    """
    exit_x = 8  # 出口的x轴
    exit_y = 10  # 出口的y轴
    # 定义二维数组迷宫,规格：10*12
    maze = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], \
            [1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], \
            [1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1], \
            [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1], \
            [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1], \
            [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1], \
            [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1], \
            [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1], \
            [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1], \
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    # 获得路径对象和起点位置
    path = PathRecord()
    x = 1
    y = 1

    # 显示迷宫地图
    print("迷宫地图：")
    show_maze(maze)

    print("开始走迷宫：")
    # 开始走迷宫
    while x <= exit_x and y <=exit_y:  # 判断（x,y）的坐标是否是终点坐标
        if maze[x][y] == 2:  # 若进行回退操作，则将回退的路径设置成3
            maze[x][y] = 3
        else:
            maze[x][y] = 2  # 将当前位置的信息设置成已经走过
        # 对当前位置的东、西、南、北四个方向进行判断，寻找出路。
        if maze[x-1][y] == 0:  # 对西方向进行判断，0表示可以前进。
            path.push(x, y)  # 将当前的位置信息（走过）压入栈中
            x -= 1  # 更正x的位置,向西的方向前进一格。
        elif maze[x+1][y] == 0:  # 对东方向进行判断
            path.push(x,y)
            x += 1  # 更新x的位置，向东的方向前进一格
        elif maze[x][y+1] == 0:  # 对北方向进行判断
            path.push(x, y)
            y -= 1  # 更新y的位置，向北方向前进一格
        elif maze[x][y-1] == 0:  # 向南方向进行判断
            path.push(x, y)
            y += 1  # 更新y的位置，向南方向前进一格
        elif judge_exit(maze, x, y, exit_x, exit_y) == 1:  # 判断是否到达出口，而不是死胡同。
            show_maze(maze)
            break
        else:  # 当没有方向可走时，此刻老鼠走到死胡同中，且没有到达出口。
            maze[x][y] = 3
            # 进行回退操作，从栈顶节点中获得回退的位置信息
            x = path.top.x  # 栈顶中回退的x轴
            y = path.top.y  # 栈顶中回退的y轴
            path.pop()  # 将栈顶节点进行出栈操作，更新栈顶节点

        # 实时显示走过的路径
        show_maze(maze)
        time.sleep(1)  # 程序执行延迟1s


def show_maze(maze):
    """
    显示走过的路径
    :param maze:迷宫数组
    :return:
    """

    for i in range(10):
        for j in range(12):
            print(maze[i][j], end='')
        print()


if __name__ == "__main__":
    maze()
