'''
逻辑回归
'''
import tensorflow as tf

tf.compat.v1.enable_eager_execution()  # 使程序能够更快的运行

csv_path = 'data.csv'  # csv文件路径


def grade_map(grade1, grade2, label):
    '''
    csv文件数据映射成字典形式
    :param grade1: 分数1
    :param grade2: 分数2
    :param label: 标签
    :return: 字典数据
    '''
    return {'grade1': grade1, 'grade2': grade2, 'label': label}


# 读取csv数据
dataset = tf.data.experimental.CsvDataset(csv_path,  # csv文件路径
                                          # 规定每列的类型
                                          record_defaults=[tf.float32, tf.float32, tf.int32],
                                          select_cols=[0, 1, 2],  # 选择的列
                                          field_delim=',',  # 分割符
                                          header=True)  # 当csv文件中有头标签，当进行解析时进行跳过
# 对数据进行进一步的处理
dataset = dataset.filter(lambda grade1, grade2, label: label < 1)
# dataset = dataset.map(map_func=grade_map)
# dataset = dataset.batch(1)
# 显示数据
for element in dataset:
    tf.print(element)




