

def max_sum(a):
    # 定义数组长度,最大子序列值，最大子序列值的索引值
    length = len(a)
    max_sum = 0
    best_i, best_j = 0, 0

    # 穷举法，i代表子序列的开始索引
    for i in range(length):
        for j in range(i, length):  # j代表子序列的结束索引
            # 在当前子序列的情况，得到子序列的值
            cur_sum = 0
            for k in range(i, j+1):
                cur_sum += a[k]
            # 当前子序列的值和最大值进行对比，并更新最大值，最大值的开始索引和结束索引
            if cur_sum > max_sum:
                max_sum = cur_sum
                best_i = i
                best_j = j
    return sum, best_i, best_j


def max_sum_improve(a):
    length = len(a)
    sum = 0
    best_i, best_j = 0, 0
    for i in range(length):
        cur_sum = 0
        for j in range(i, length):
            cur_sum += a[j]
            if cur_sum > sum:
                sum = cur_sum
                best_i = i
                best_j = j
    return sum, best_i, best_j


def max_sub_sum(a, left, right):
    sum = 0
    if left == right:
        if a[left] > 0:
            sum = a[left]
        else:
            sum = 0
    else:
        center = int((left+right)/2)
        left_sum = max_sub_sum(a, left, center)
        right_sum = max_sub_sum(a, center+1, right)

        left_max_sum, left_cur_sum = 0, 0
        for i in reversed(range(left, center+1)):
            left_cur_sum += a[i]
            if left_cur_sum > left_max_sum:
                left_max_sum = left_cur_sum

        right_max_sum, right_cur_sum = 0, 0
        for j in range(center+1, right+1):
            right_cur_sum += a[j]
            if right_cur_sum > right_max_sum:
                right_max_sum = right_cur_sum

        sum = left_max_sum + right_max_sum
        if sum < left_sum:
            sum = left_sum
        if sum < right_sum:
            sum = right_sum

    return sum


def max_sum_dc(a):
    length = len(a)
    sum = max_sub_sum(a, 0, length-1)
    return sum

def max_sum_dp(a):
    length = len(a)
    sum, b = 0, 0
    for i in range(length):
        if b>0:
            b += a[i]
        else:
            b = a[i]
        if b > sum:
            sum = b
    return sum

if __name__ == '__main__':
    # a = [-1, 2, 3, 5, -4, 2, 1]
    a = [-2, 11, -4, 13, -5, -2]
    max_sum_1 = max_sum(a)
    print(max_sum_1)
    max_sum_2 = max_sum_improve(a)
    print(max_sum_2)
    max_sum_3 = max_sum_dc(a)
    print(max_sum_3)
    max_sum_4 = max_sum_dp(a)
    print(max_sum_4)
