"""
跳转搜索(Jump Search),跳转搜索是搜索的一种算法，其查找的数组都是已经排好序的数组
基本思想：按固定的步长跳过一些数组中的元素从而检索更少的元素。
    例如，假设有n长度的数据arr，跳转步长为m，然后我们搜索索引arr[0],arr[m],arr[2m],..arr[km]等等。
    一旦我们发现间隔为（arr[km]<x<arr[(k+1)m]）,我们从索引km开始执行线性搜索去查找元素x。
问题：进行跳过的步长，设置成多少是最好？
    在最坏的情况下，我们不得不进行n/m(n:数组长度，m跳过步长)次跳过。如果搜索到的值正好是m索引的前一个，
    则需要进行m-1比较。因此，在最差情况下，总体的比较次数为（n/m）+(m-1)。当m=sqrt(n)时，那个函数的值
    将是最小。因此，最好的步长是m=sqrt(n)
重要点：
    （1）这个算法仅仅支持排好序的数组
    （2）跳转的最优步长为sqrt(n)[对n开根号]。这个使jump search的时间复杂度为O(sqrt(n))
    （3）跳转算法的时间复杂度是介于线性搜索（顺序，O(n)）和二分搜索（O(log(n))）之间
    （4）二分搜索比跳转搜索更好，但是在一个跳回很繁琐的情况下，使用跳转搜索更好。
"""

import math


def jump_search(data, key):
    """
    跳转搜索
    :param data: 查找数组
    :param key: 查找关键词
    :return: 查找位置
    """
    length = len(data)  # 数据长度
    # 计算跳转步长
    step = math.sqrt(length)
    pre_step_index = 0  # 保存前一个步长所在的索引
    # min()函数的使用，是选取最小值，防止步长越界，使索引保持在length内
    while data[int(min(step, length))-1] < key:  # 判断数据存在查找关键词的条件
        pre_step_index = step  # 保存当前步长索引
        step = step + math.sqrt(length)  # 更新下一个步长索引
        if pre_step_index >= length:  # 若前一个步长索引超过数组长度，则表明查找的值不存在
            return -1

    # 从pre_step_index索引开始查找关键词
    while data[int(pre_step_index)] < key:
        pre_step_index = pre_step_index + 1
        # 如果到达下一个步长索引或者最后的数组索引，则查找的值不存在
        if pre_step_index == min(length, step):
            return -1

    # 如果元素找到了
    if data[int(pre_step_index)] == key:
        return int(pre_step_index)
    return -1


if __name__ == '__main__':
    data = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    print(jump_search(data, 9))
