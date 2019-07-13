"""
任务：使用数组来实现二叉树的存储方式。
二叉树介绍：一个由有限节点所组成的集合，此集合可以是空集合，或由一个树根及其左右两个子树组成。
        二叉树最多有两个子节点，即度数小于或等于2
二叉树的特性：1、树不可以为空集，但二叉树可以
           2、树的度数为d>=0,但二叉树的节点度数为0<=d<=2
           3、树的子树间没有次序关系，二叉树则有
特殊二叉树：1、满二叉树：若二叉树的高度为h，树的节点数为2^h-1,h>0
         2、完全二叉树：若二叉树的高度为h,所含节点数小于2^h-1,h>0。其节点的编号方式从左到右、从上到下一一对应。
            对于完全二叉树而言，假设有N个节点，则二叉树的层数h=[log2(N+1)]
        3、斜二叉树：当一个二叉树完全没有右节点或者左节点时，就把它称为左斜二叉树或右斜二叉树
        4、严格二叉树：二叉树的每一个非叶节点均有非空的左右子树
"""


class BinaryTreeList(object):
    """
    用一维数组构建二叉树，也就是二叉查找树。
    二叉查找树遵循的规则：(1)左子树索引值是父节点索引值×2
                      (2)右子树索引值是父节点索引值×2+1
                      (3)可以是空集合，若不是空集合节点上一定有键值
                      (4)每一个树根的值大于其左子树,小于其右子树
                      (5)左右子树也是二叉查找树
                      (6)树的每个节点值都不相同
    """
    def __init__(self, data):
        self.binary_tree = [0] * 16  # 声明空的二叉树
        self.data = data  # 输入数据

    def create_binary_tree(self):
        """
        创建二叉树(二叉查找树)
        :return:
        """
        for i in range(0, len(self.data)):
            index = 1  # 索引值
            # 从根结点遍历不为0的结点，获得输入数据的放置索引
            while self.binary_tree[index] != 0:
                if self.data[i] > self.binary_tree[index]:  # 若输入数据大于根结点，则向右子树进行比较
                    index = index*2+1  # 更新索引值
                else:  # 输入数据小于根结点，则向左子树进行比较
                    index = index*2
            self.binary_tree[index] = self.data[i]  # 放置节点值


if __name__ == "__main__":
    data = [0, 6, 3, 5, 4, 7, 8, 9, 2]
    binary_tree = BinaryTreeList(data)
    binary_tree.create_binary_tree()
    print("原始数据：")
    for i in range(len(data)):
        print("{}".format(data[i]), end='')
    print('')
    print("二叉树：")
    for j in range(1, len(binary_tree.binary_tree)):
        print("{}".format(binary_tree.binary_tree[j]), end='')
    print('')