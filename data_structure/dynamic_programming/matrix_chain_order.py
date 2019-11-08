import sys

"""
动态规划
实现矩阵链乘
时间复杂度：O(n^3)
空间复杂度：O(n^2)
"""

def matrix_multiply(a, b):
    a_len = len(a[0])
    b_len = len(b)
    c = [[0 for x in range(a_len)] for x in range(b_len)]

    for i in range(a_len):
        for j in range(b_len):
            c[i][j] = 2

def matrix_chain_order(array):
    n = len(array)
    matrix = [[0 for x in range(n)] for x in range(n)]
    sol = [[0 for x in range(n)] for x in range(n)]
    print(matrix)
    print(sol)

    for chain_length in range(2, n):
        for a in range(1, n-chain_length+1):
            b = a + chain_length-1
            print(b)

            print(sys.maxsize)
            matrix[a][b] = sys.maxsize
            for c in range(a, b):
                cost = (matrix[a][c] + matrix[c+1][b] + array[a-1]*array[c]*array[b])
                if cost < matrix[a][b]:
                    matrix[a][b]=cost
                    sol[a][b] = c
    return matrix, sol

def print_optimal_solution(optimal_value, i, j):
    if i == j:
        print('A'+str(i), end='')
    else:
        print('(', end='')
        print_optimal_solution(optimal_value, i, optimal_value[i][j])
        print_optimal_solution(optimal_value, optimal_value[i][j]+1, j)
        print(')', end='')




def main():
    a = [[1, 2, 3], [2, 3, 4]]
    b = [[0, 1, 2], [1, 1, 7], [4, 5, 6]]
    matrix_multiply(a, b)

    # array = [30, 35, 15, 5, 10, 20, 25]
    # n = len(array)
    # # 通过数组里数据将创建矩阵大小为：30*35，35*15,15*5,5*10,10*20,20*25
    # matrix, optimal_solution = matrix_chain_order(array)
    # # matrix_chain_order(array)
    #
    # print("No. of Operation required: " + str((matrix[1][n-1])))
    # print_optimal_solution(optimal_solution, 1, n-1)


if __name__ == '__main__':
    main()