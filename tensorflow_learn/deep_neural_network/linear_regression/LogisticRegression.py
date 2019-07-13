'''
逻辑回归
'''
import tensorflow as tf

tf.enable_eager_execution()  # 使程序能够更快的运行

csv_path = 'data.csv'  # csv文件路径

def grade_map(grade1, grade2, label):
    '''
    csv文件数据映射成字典形式
    :param grade1: 分数1
    :param grade2: 分数2
    :param label: 标签
    :return: 字典数据
    '''
    return {grade1: grade1,}


