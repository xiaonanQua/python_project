"""
python表示函数私有在函数名面前加入'_'下划线。外部不需要引用的函数全部标成private，外部需要引用的函数标成public。
"""


def _private_1(name):
    return 'Hello, %s' % name


def _private_2(name):
    return 'Hi, %s' % name


def greeting(name):
    if len(name) >3:
        return _private_1(name)
    else:
        return _private_2(name)


if __name__ == '__main__':
    print(greeting('xiaonan'))