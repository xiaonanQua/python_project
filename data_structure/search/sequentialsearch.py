"""
查找表是由同一个类型的数据元素（或记录）构成的集合。
关键字是数据元素中某个数据项的值，又称键值。若此关键词可以唯一地标识一个记录，则称此关键字为主关键字。
对于识别多个数据元素的关键字，称为次关键字。
查找就是根据给定的一个值，在查找表中确定一个其关键字等于给定的数据元素或记录
查找表分为静态查找表(Static Search table)、动态查找表（Dynamic Search Table）。
静态查找表：只有查找操作的查找表。（1）查询某个特定数据元素是否在表中。（2）检索某个特定数据元素和各种属性
动态查找表：在查找过程中同时插入查找表中不存在的数据元素，或者从查找表中删除已经存在的某个数据元素。
"""


def sequential_search(data, key):
    """
    顺序查找
    :param data:查找表
    :param key: 关键字
    :return: 返回查找的位置
    """
    for i in range(len(data)):
        if data[i] is key:
            return i
    return 0


def binary_search(data, key):
    """
    二分查找，也就是折半查找。对列表数据的位置进行折半划分。
    :param data: 查找的数据
    :param key: 查找关键词
    :return: 查找结果
    """
    # 设置左、中坐标
    low = 0
    high = len(data) - 1
    # 当左坐标大于右坐标
    while low <= high:
        mid = int((low + high) / 2)  # 中间坐标
        if data[mid] < key:
            low = mid + 1  # 最低位置调整到中间下标大一位
        elif data[mid] > key:
            high = mid - 1  # 最高位置调整到中间下标低一位
        else:
            return mid  # 查找到关键词的位置
    return None


def interpolation_search(data, key):
    """
    插值查找，顾名思义就是通过数据值进行查找，是折半查找的一种改进。
    即对mid=int((low+high)/2)进行改进，
    修改成mid=int(low+(high-low)*(key-data[low])/(data[high]-data[low])
    :param data: 查找的列表
    :param key: 查找的关键词
    :return: 查找的位置
    """
    # 设置低、高位置
    low = 0
    high = len(data) - 1
    # 迭代查找满足的值
    while low <= high:  # 跳出循环的条件
        # 求出中间坐标
        mid = int(low + (high - low)*(key - data[low])/(data[high] - data[low]))
        if key < data[mid]:  # 向左遍历
            high = mid - 1
        elif key > data[mid]:
            low = mid + 1
        else:
            return mid
    return None


if __name__ == '__main__':
    data = [1, 3, 5, 6, 3, 0, 5]
    data_seq = [1, 2, 3, 4, 5, 6, 7]
    print(sequential_search(data, 3))
    # 二分查找
    print(binary_search(data_seq, 6))
    # 插值查找
    print(interpolation_search(data_seq, 3))
