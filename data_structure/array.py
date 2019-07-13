
import math # 导入数学函数库

# 稀疏矩阵
# 将一个矩阵中0元素最多的矩阵，转化成三元组（3-tuple），三元组的形式是（行，列，值）。
def compress_sparse():
    # 变量
    temp = 1 # 临时变量
    count = 0 # 统计矩阵中元素个数
    # 稀疏矩阵
    sparse = [[25, 0, 0, 32], [0, -25, 0, 33], [77, 0, 0, 0], [0, 0, 0, 0], [101, 0, 0, 0]]
    # 统计元素个数,并输出原始数据
    for i in range(5):
        for j in range(4):
            # 输出
            print("%d" %(sparse[i][j]), end='\t')
            # 统计不为0的元素个数
            if sparse[i][j] != 0:
                count=count+1 # 自动加一
    print("不为零的个数有：%d" %(count), end='\n')

    # 压缩矩阵
    # 输出成count+1行3列的数组，从左向右结合，[None]*3生成3列，一个循环生成count+1行的数组
    compress = [[None] * 3 for row in range(count+1)] 
    print("输出初始压缩矩阵：", compress, end='\n')
    # 初始化三元组的第一行
    compress[0][0] = 5 # 表示此矩阵的行数
    compress[0][1] = 4 # 表示此矩阵的列数
    compress[0][2] = count # 表示此矩阵非零项的总数

    # 开始压缩矩阵
    for i in range(5):
        for j in range(4):
            if sparse[i][j] != 0:
                compress[temp][0] = i # 存储行
                compress[temp][1] = j # 存储列
                compress[temp][2] = sparse[i][j] #存储值
                temp = temp + 1
    # 输出压缩数据
    print("[稀疏矩阵压缩成三元组]")
    for i in range(count+1):
        for j in range(3):
            print(compress[i][j], end=' ')
        print(end='\n')
    return 0             

''' 
上三角形矩阵(Upper triangular matrix)：就是对角线以下都为0的N×N的矩阵
类别：分为左上三角形矩阵(left upper triangular matrix)和右上三角形(right upper triangular matrix)。
作用：因为矩阵中存在大量0元素，会造成内存空间的浪费，所以通过将二维矩阵中数据压缩到一维数组中
'''
# 三角形矩阵
# 对N×N的矩阵，当i>j时，A(i,j)=0;当i<=j时，A(i,j)=a(i,j)。所以需要一个一维数组B(1:n(1+n)/2)来存储。
# 将二维数组的非零项映射到一维数组中，有按行映射和按列映射的方式。
def triangular_matrix(matrix):
    n = len(matrix) # 求出二维矩阵的长度
    l = int((n*(1+n))/2) # 求出一维数组的长度
    k = 0 # 定义变量，用于更新一维数组的索引
    map_matrix = [None] * l # 为长度为l的一维数组中的每个元素设置成None
    # 显示三角形矩阵
    for i in range(n):
        for j in range(n):
            print("%d" %(matrix[i][j]), end='\t')
        print('\n')
    # 按行映射
    for i in range(n):
        for j in range(n):
            if matrix[i][j] != 0:
                map_matrix[k] = matrix[i][j]
    # 显示压缩后的一维数组
    for i in range(n):
        for j in range(n):
            index = get_index(i,j,n)
            print("%d" %(map_matrix[index]))
    return 

# 通过行、列获得一维数组的索引
def get_index(i,j,n):
    index = int(n*i - i*(i+1)/2 + j)
    return index

            

# 执行函数
# compress_sparse() # 稀疏矩阵压缩
triangular_matrix([[1, 2, 3, 4], [0, 5, 6, 7], [0, 0, 8, 9,], [0, 0, 0, 10]])

