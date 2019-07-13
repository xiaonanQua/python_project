"""
使用链表实现二叉树，使用链表来表示二叉树的好处是对于节点的增加与删除相当容易，缺点是很难找到父节点。
实现二叉树链表的创建，前序遍历，中序遍历，后序遍历。
"""


class Node:
    """
    树的节点
    """
    def __init__(self):
        self.data = None  # 节点数据
        self.left = None  # 左节点
        self.right = None # 右节点


class BinaryTreeLink(object):
    """
    二叉树链表
    """
    def __init__(self):
        self.root = None  # 根节点

    def create_binary_tree(self, data):
        """
        创建二叉树链表
        :param data:用于创建二叉树的数据
        :return:
        """
        # 构建新结点
        new_node = Node()
        new_node.data = data
        # 若根结点为空，则将新结点填入
        if self.root is None:
            self.root = new_node
        else:
            cur_node = self.root  # 当前节点
            # 若当前结点为空则结束
            while cur_node is not None:
                father_node = cur_node  # 保存父结点
                if data < cur_node.data:  # 若输入数据小于根节点的数据，更新当前结点为左子树
                    cur_node = cur_node.left
                else:  # 若输入数据大于根节点的数据，更新当前结点为右子树
                    cur_node = cur_node.right
            # 获得添加数据的父结点，判断新结点加入左结点还是右结点的位置
            if data > father_node.data:
                father_node.right = new_node
            else:
                father_node.left = new_node

    def inorder(self, cur_node):
        """
        递归实现，中序遍历。从左向右先访问左子树，再访问根结点，最后访问右子树。
        :param cur_node:当前结点
        :return:
        """
        if cur_node is not None:
            self.inorder(cur_node.left)  # 遍历左子树
            print("{}".format(cur_node.data), end=' ')
            self.inorder(cur_node.right)  # 遍历右子树

    def preorder(self, cur_node):
        """
        递归实现，前序遍历。从左向右先访问根结点，再访问左子树，最后访问右子树。
        :param cur_node:当前结点
        :return:
        """
        if cur_node is not None:
            print("{}".format(cur_node.data), end=' ')
            self.preorder(cur_node.left)  # 遍历左子树
            self.preorder(cur_node.right)  # 遍历右子树

    def postorder(self, cur_node):
        """
        递归实现，后序遍历。从左向右先访问左子树，再访问右子树，最后访问根结点。
        :param cur_node: 当前结点
        :return:
        """
        if cur_node is not None:
            self.postorder(cur_node.left)  # 遍历左子树
            self.postorder(cur_node.right)  # 遍历右子树
            print("{}".format(cur_node.data), end=' ')

    def search(self, data):
        """
        根据键值查找节点。
        二叉树是根据左节点<树根<右节点的规则建立起来。在进行二叉树遍历时，根据键值的对比，遍历相应的节点，直到
        节点为None，表示遍历结束。
        :param data: 查找结点的键值
        :return: 查找的节点，或者None
        """
        cur_node = self.root  # 当前节点
        while cur_node is not None:  # 若当前节点不为空，则遍历
            # 当前节点数据和要查找的数据相等，返回节点
            if cur_node.data is data:
                return cur_node
            elif cur_node.data < data:  # 当前节点数据小于查找数据，向右子树遍历。
                cur_node = cur_node.right  # 更新当前节点
            elif cur_node.data > data:  # 当前节点数据大于查找数据，向左子树遍历
                cur_node = cur_node.left  # 更新当前节点
        return cur_node

    def insert_node(self, data):
        """
        向二叉树中插入节点。
        首先判断二叉树中是否存在要插入的数据。若存在，则无法插入。
        :param data: 插入的数据
        :return: None表示插入失败
        """
        cur_node = self.root  # 当前结点
        # 构建新节点
        new_node = Node()
        new_node.data = data

        # 循环遍历,找到插入的位置
        while cur_node is not None:
            pre_node = cur_node  # 保存前节点
            if cur_node.data is data:
                break
            elif cur_node.data > data:
                cur_node = cur_node.left  # 更新当前节点为左子树
            elif cur_node.data < data:
                cur_node = cur_node.right  # 更新当前节点为左子树

        # 判断插入数据在二叉树是否存在
        if cur_node is None:  # 若为None，则可以插入
            if pre_node.data < data:  # 插入右子树
                pre_node.right = new_node
            else:
                pre_node.left = new_node  # 插入左子树
        else:  # 插入失败
            return None


if __name__ == '__main__':
    data = [5, 6, 24, 8, 12, 3, 17, 1, 9]
    binary_tree = BinaryTreeLink()
    print("原始数据：")
    for i in range(len(data)):
        print("{}".format(data[i]), end=' ')
        binary_tree.create_binary_tree(data[i])
    print('')
    print("链表数据：")
    cur_node = binary_tree.root
    print("前序遍历：")
    binary_tree.preorder(cur_node)
    print('')
    print("中序遍历：")
    binary_tree.inorder(cur_node)
    print('')
    print("后序遍历：")
    binary_tree.postorder(cur_node)
    print('')
    print('查找特定节点：')
    node = binary_tree.search(24)
    print('当前结点数据：{}，左子树：{}，右子树：{}'.format(node.data, node.left.data, node.right))
    print('插入结点：')
    binary_tree.insert_node(2)
    binary_tree.inorder(cur_node)

