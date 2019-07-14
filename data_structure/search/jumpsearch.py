"""
跳转搜索(Jump Search),跳转搜索是搜索的一种算法，其查找的数组都是已经排好序的数组
基本思想：按固定的步长跳过一些数组中的元素从而检索更少的元素。
例如，假设有n长度的数据arr，跳转步长为m，然后我们搜索索引arr[0],arr[m],arr[2m],..arr[km]等等。
一旦我们发现间隔为（arr[km]<x<arr[(k+1)m]）,我们从索引km开始执行线性搜索去查找元素x。
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
    while data[int(min(step, length))] < key:  # 判断数据存在查找关键词的条件
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
    print(jump_search(data, 7))
