"""
任务：基于堆栈思想实现八皇后问题。
规则：皇后可以吃掉同行、同列、对角方向的棋子。当新皇后加入时，需要考虑所处的位置不存在同行、同列、对角方向的皇后，
    否则会被旧皇后吃掉。
    利用这个规则，将其应用在4×4的棋盘上就称为4-皇后问题;应用在8x8的棋盘上就称8-皇后问题;应用在N×N的棋盘上就称为
    N-皇后问题。
实现细节：以八皇后问题为例:(1)在棋盘中放置新皇后,若这个位置不会被之前放置的皇后吃掉，就将新皇后的位置压入堆栈中。
                     (2)但是，如果放置新皇后的该行或该列的八个位置都没有办法放置新皇后，此时就必须从堆栈中
                     弹出前一个皇后的位置，并在该行或该列重新寻找另一个新的位置来放，再将该位置压入堆栈中，
                     这也就是回溯算法的应用。
"""

import time

class Queen:
    """
    定义皇后堆栈，用列表实现堆栈。
    """
    def __init__(self, N):
        self.n = N  # N皇后
        self.queen = [None] * N  # 保存N个皇后的堆栈
        self.number = 0  # 计算共有几组解

    def check_attack(self, row, col):
        """
        检测在(row,col)位置是否会受到攻击。
        :param row:行位置
        :param col:列位置
        :return: 返回检测结果。若遭受攻击，则返回值为1,否则返回为0。
        """
        i = 0  # 遍历索引
        flag = 0  # 标记是否存在攻击的危险，0代表没有，1代表受到攻击。
        offset_row=offset_col = 0  # 对角线上的行与列
        while flag != 1 and i < col:
            offset_col = abs(i - col)
            offset_row = abs(self.queen[i] - row)
            # 判断两个皇后是否在同一行或在同一对角线上
            if self.queen[i] == row or offset_row == offset_col:
                flag = 1
            i = i + 1
        return flag

    def show_result(self):
        """
        显示结果
        :return:
        """
        self.number += 1  # 增加一次
        for i in range(self.n):
            for j in range(self.n):
                if self.queen[j] == i:
                    print('1', end='')
                else:
                    print('0', end='')
            print()
        print('\t')

    def choose_position(self, col):
        """
        选择插入位置，并判断插入的位置是否符合规则
        :param col: 需要插入的列位置
        :return:
        """
        row = 0  # 定义行位置，从第一行开始
        # 循环遍历n行，每行找到可以放置的位置
        while row < self.n:
            if self.check_attack(row, col) != 1:
                self.queen[col] = row
                if col == 7:
                    self.show_result()
                    #time.sleep(1)
                else:
                    self.choose_position(col+1)
            row = row + 1


if __name__ == "__main__":
    queen = Queen(8)
    queen.choose_position(0)
    print("共有{}个解".format(queen.number))