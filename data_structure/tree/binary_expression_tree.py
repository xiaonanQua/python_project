"""
二叉运算树。
建立二叉运算树遵循两种规则：
（1）考虑表达式中运算符的结合性与优先级，再适当地加上括号，其中树叶一定是操作数，父节点是运算符。
（2）再从最内层的括号逐步向外，利用运算符做树根，左边操作数当左子树，右边操作数当右子树，其中优先级最低作为此二叉运算数的树根。

"""


class Node(object):
    """
    声明节点类。
    """
    def __init__(self):
        self.data = None
        self.left = None
        self.right = None


class BinaryTree:
    """
    二叉树
    """
    def __init__(self, data):
        self.root = None  # 根结点
        for i in range(len(data)):
            self.insert_node(data[i])

    def insert_node(self, data):
        """
        添加节点到二叉树中
        :param data: 添加数据
        :return:
        """
        cur_node = self.root  # 当前结点
        # 构建新节点
        new_node = Node()
        new_node.data = data

        # 若是根节点为空，则先添加
        if self.root is None:
            self.root = new_node
        else:
            while cur_node is not None:
                if cur_node.data < data:
                    cur_node = cur_node.right  # 指向右节点
                elif cur_node.data > data:
                    cur_node = cur_node.left  # 指向左节点
            if cur_node is None:
                cur_node = new_node  # 将新结点添加


class ExpressionTree(BinaryTree):
    def __init__(self, info, index):
        """
        二叉运算树类,继承链表二叉树类
        :param info:创建的信息
        :param index: 索引
        """
        self.root = self.create(info, index)

    def create(self, seq, index):
        if index >= len(seq):  # 递归出口，索引大于整体数组长度
            return None
        else:
            # 构建新节点
            new_node = Node()
            new_node.data = seq[index]
            new_node.left = self.create(seq, index*2)  # 建立左子树
            new_node.right = self.create(seq, index*2+1)  # 建立右子树
            return new_node

    def preorder(self, node):
        if node is not None:
            print("{}".format(node.data), end=' ')
            self.preorder(node.left)
            self.preorder(node.right)

    def inorder(self, node):
        if node is not None:
            self.inorder(node.left)
            print("{}".format(node.data), end=' ')
            self.inorder(node.right)

    def postorder(self, node):
        if node is not None:
            self.postorder(node.left)
            self.postorder(node.right)
            print(node.data, end=' ')

    def condition(self, operator, num1, num2):
        """
        实现算术运算
        :param operator: 运算符
        :param num1: 左操作数
        :param num2: 右操作数
        :return: 运算的结果
        """
        if operator == '*':
            return num1 * num2
        elif operator == '/':
            return num1 / num2
        elif operator == '+':
            return num1 + num2
        elif operator == '-':
            return num1 - num2
        elif operator == '%':
            return num1 % num2
        else:
            return -1

    def count_experssion(self, node):
        """
        计算二叉运算树的值
        :param node: 根节点
        :return:
        """
        # 定义左右子树的值
        left_num = 0
        right_num = 0
        # 递归调用的出口条件
        if node.left is None and node.right is None:
            return node.data # 将节点字符值转化成数值
        else:
            left_num = self.count_experssion(node.left)  # 计算左子树表达式的值
            right_num = self.count_experssion(node.right)  # 计算右子树表达式的值
            return self.condition(node.data, left_num, right_num)


if __name__ == '__main__':
    information = [' ', '+', '*', '%', 6, 3, 9, 5]
    exp = ExpressionTree(information, 1)
    print("中序表达式：", end=' ')
    exp.inorder(exp.root)
    print('')
    print('前序表达式：', end=' ')
    exp.preorder(exp.root)
    print()
    print('后序表达式：', end=' ')
    exp.postorder(exp.root)
    print()
    print('计算表达式的值：', end=' ')
    print(exp.count_experssion(exp.root))