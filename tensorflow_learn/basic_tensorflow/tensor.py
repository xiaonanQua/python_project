"""
简介：正如TensorFlow框架名字所示，Tensor是整个系统中重要的概念。若是将计算图称为计算模型，那tensor(张量)
    可以称为数据模型。
定义：张量（Tensor），可以理解为不同维度的数组。
    通常有五种形式：(1)标量(常数)：零阶张量，1
                 (2)向量(数组)：一阶张量，[1,]
                 (3)二维数组：二阶张量[[1],]
                 (4)n维数组：n阶张量[[[n]]]
张量的三个属性：(1)操作：可以看作张量的名称，也可以看作是唯一标识符。张量的命名具有规律性，这和计算图的节点有关。
                    由于计算图中每一个节点都代表一个运算，而节点的运算结果的属性是由张量保存下来。操作的
                    命名符是"node:src_output"的形式，其中node就是节点名称，src_output表示节点的第几个输出（编号从0开始）.
            (2)维度：描述张量的维度信息。这些维度都是可以进行修改的。
            (3)数据类型：每个张量的数据类型都是唯一的。进行某个运算的张量的数据类型必须一致，否则会报错。
                       TensorFlow中支持14种不同的数据类型，分为四类：整数型tf.(int8, int16, int32, int64, uint8)
                       实数型tf.(float32, float64),布尔型tf.bool,复数型tf.(complex64, complex128)。
注意：（1）张量只是引用程序中运算的结果，而不是真实的数组。张量保存的只是运算结果的属性，而不是真正的数字。
     （2）在声明变量或常量时，可以直接使用dtype指定数据类型。若不指定则使用默认的类型：如带小数点的数会被默认为float32,不带小数点是int32.
         为了防止默认类型潜在的类型不匹配问题，通常建议使用dtype指明变量或者常量的数据类型。
"""

import tensorflow as tf

# 定义两个常量的张量
a = tf.constant([1.0, 2.0], name='a')
b = tf.constant([3.0, 4.0], name='b')
result = a + b
print(result)

# 上面程序输出的结果为：Tensor("add:0", shape=(2,), dtype=float32)
# 由此可见，张量的运算结果，只是运算结果的属性。result属于一个张量，保存加法运算结果的3个属性：操作(op),
# 维度(shape)和数据类型(dtype)


# 定义计算图
plus_graph = tf.Graph()
with plus_graph.as_default():
    x = tf.constant([1, 2], name='x', dtype='float32')
    y = tf.constant([3, 4], name='y', dtype='float32')
    result = x + y
# 定义会话
with tf.Session(graph=plus_graph) as sess:
    tf.global_variables_initializer().run()
    # 使用会话显示真实的值
    print(sess.run(result))