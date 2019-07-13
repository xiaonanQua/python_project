from functools import reduce


def f1(x):
    return x * x


def f2(x, y):
    return x * y


def f3(x, y):
    return x*10 + y


def char2num(s):
    digits = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
    return digits[s]


def str2int(s):
    digits = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}

    def convert_int(x, y):
        return x*10 + y

    def str_to_int(x):
        return digits[x]
    return reduce(convert_int,map(str_to_int, s))


def is_odd(x):
    """
    判断是否是奇数
    :param x:
    :return:
    """
    return x % 2 == 1


def not_empty(s):
    return s and s.strip()


def log(func):
    def wrapper(*args, **kw):
        print('call %s():' % func.__name__)
        return func(*args, **kw)
    return wrapper


@log
def now():
    print('2019-7-11')


if __name__ == '__main__':
    list1 = [1, 2, 3, 4, 5]
    list2 = [3, 1, -2, -6, 4]
    list3 = ['nan', 'shuai', 'Pan']
    # map使用
    print(list(map(f1, list1)))
    print(list(map(str, list1)))  # 转化成字符串
    # reduce使用
    print(reduce(f2, list1))  # 累乘
    print(reduce(f3, list1))  # 转化成整数
    # 将字符串转化成整数
    print(reduce(f3,map(char2num, '12345')))
    print(str2int('123'))
    # filter过滤器的使用
    print(list(filter(is_odd, list1)))  # 过滤出奇数
    # 使用filter过滤掉空格
    print(list(filter(not_empty, ['s', 'x', ' ', None, ''])))
    # 使用sorted进行排序
    print(sorted([3, 2, 7, 6, 9]))  # 常规排序
    print(sorted(list2, key=abs))  # 使用key限制排序忽视负数
    print(sorted(list3, key=str.lower, reverse=True))  # 忽略大写逆向排序
    # 使用lambda匿名函数
    print(list(map(lambda x: x * x, list1)))
    f = now
    print(now.__name__)
    print(f.__name__)
    now()




