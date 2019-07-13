"""
图的数据表示方法有四种：
（1）邻接矩阵法：假设图A有n个顶点，n>=1,用n×n的二维矩阵来表示。矩阵中设A[i,j]=1,表示图中有一条边（Vi,Vj）存在
             反之，A[i,j]=0,则不存在边（Vi,Vj）
             特点：（1）对于无向图而言，邻接矩阵一定是对称，有向图不一定。而且对角线一定是0。
                  （2）在无向图中，任一节点i的度数是矩阵i行所有元素的和。在有向图中，节点i的入度数为第i行所有元素
                      的和，而出度数是第j列所有元素的和。
                  （3）用邻接矩阵法表示图共需要n×n个单元空间。由于无向图的邻接矩阵具有对称性，扣除对角线全部为零外
                      仅需存储上三角或下三角的数据即可，因此仅需要n(n-1)/2的单位空间。
(2)邻接表法：使用链表存储每个顶点出度的顶点。
(3)邻接复合链表法
(4)索引表格法
"""


def adjacent_matrix_graph(data, n):
    """
    无向图的邻接矩阵表示
    :param data: 无向图
    :param n: 顶点数
    :return: 邻接矩阵
    """
    adj_mat = [[0]*n for row in range(n)]  # 使用列表生成式生成邻接矩阵
    # for i in range(len(data)):
    #     for j in range(2):
    #          for k in range(n):
    #              origin = data[i][0]  # 起点
    #              end = data[i][1]  # 终点
    #              adj_mat[origin][end] = 1
    for i in range(len(data)):
        origin = data[i][0]  # 起点
        end = data[i][1]  # 终点
        adj_mat[origin][end] = 1
    print('无向图：')
    for i in range(n):
        for j in range(n):
            print(adj_mat[i][j], end=' ')
        print('')


def adjacent_matrix_digraph(data, n):
    """
    有向图的邻接表示法
    :param data: 有向图
    :param n: 顶点数
    :return: 邻接矩阵
    """
    adj_mat = [[0]*n for row in range(n)]  # 生成空的矩阵
    for i in range(len(data)):
        origin = data[i][0]
        end = data[i][1]
        adj_mat[origin][end] = 1  # 存在边
    print('有向图：')
    for i in range(n):
        for j in range(n):
            print(adj_mat[i][j], end=' ')
        print('')


class Node(object):
    """
    链表节点
    """
    def __init__(self):
        self.data = None
        self.next = None


def adjacency_list(data, n):
    """
    邻接表
    :param data:有向图，或者无向图
    :param n: 顶点数
    :return: 邻接表
    """
    # 生成顶点链表的头指针
    head = []
    for i in range(n):
        head.append(Node())
    for i in range(1, n+1):  # 读取每个顶点链表
        cur_node = head[i-1]  # 当前节点
        cur_node.data = 'V{}'.format(i)
        for j in range(len(data)):  # 读取图里的数据
            if data[j][0] == i:
                # 构建新节点
                new_node = Node()
                new_node.data = data[j][1]  # 保存出度的顶点
                cur_node.next = new_node  # 当前节点指向新节点
                cur_node = new_node  # 更新当前节点
    return head


if __name__ == '__main__':
    data = [[1, 2], [2, 1], [1, 5], [5, 1], [2, 3], [3, 2], [2, 4], [4, 2], [3, 4], [4, 3]]
    data2 = [[1, 2], [2, 1], [2, 3], [2, 4], [4, 3], [4, 1]]
    adjacent_matrix_graph(data, 6)
    adjacent_matrix_digraph(data2, 5)
    adj_list = adjacency_list(data2, 4)
    print("邻接表：")
    for i in range(len(adj_list)):
        head = adj_list[i]
        while head is not None:
            print('{}->'.format(head.data), end=' ')
            head = head.next
        print('')

