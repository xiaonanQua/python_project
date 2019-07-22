"""
插值搜索
问题：给定n个均匀分布值的排序数组arr[],编写一个函数来搜索数组中的特定元素x。
对比：线性搜索在O(n)时间内找到元素，跳转搜索需要O(sqrt(n))的时间复杂度，二分法需要O(log(n))时间复杂度。
    插值搜索是对二分搜索的改进，其中排序数组中的值均匀分布。插值搜索根据需要搜索的关键值可能会去不同的位置。
    例如，如果搜索的关键值更接近最后一个元素，插值搜索是有可能从头搜索到尾。
探头位置公式：position=low+[(key-arr[low])*(high-low)/(arr[high]-arr[low])]
算法：step1：在一循环中，使用探头位置公式计算位置的值
    step2：如果这是匹配的值，返回索引的位置并结束
    step3：如果key小于arr[pos],计算左边子数组的探头位置。否则，计算右侧子数组的探头位置
    step4：重复操作直到匹配到值或者子数组减少为0
"""


def interpolation_search(arr, length, key):
    # 两个角落的索引
    low = 0
    high = (length - 1)
    # 因为数组是有序的，所以数组中存在的元素一定在角落定义的范围内
    while low <= high:
        # 若low=high则代表查找的范围为0
        if low == high:
            if key == arr[low]:
                return low
            return -1
        # 计算探头位置
        pos = low + int(((float(high-low)/(arr[high]-arr[low]))*(key - arr[low])))
        # 判断探头位置是否为查找的数据
        if key == arr[pos]:  # 找到数据
            return pos
        elif key < arr[pos]:  # 向左角落查找，更新high的索引位置
            high = pos - 1
        else:  # 向右角落查找，更新low的索引位置
            low = pos + 1
        print(pos)
    return -1


if __name__ == '__main__':
    # 使用列表生成式生成10到50内的数，步长为2
    arr = [x for x in range(10, 51) if x % 2 == 0]
    # 数组长度
    length = len(arr)
    key = 20
    # 查询关键字key的位置
    index = interpolation_search(arr, length, key)
    # 判断查询的结果
    if index == -1:
        print('查询不到结果')
    else:
        print('{}的位置为：'.format(key, index))
