---
layout: post
title: "详解CIFAR-10数据集转化成TFRecord格式和读取的过程"
date: 2019-8-02
tages: 
    -  DeepLearning
reward: false
---
> 这篇文章，我将详细讲解如何将CIFAR-10数据集转化成TFRecord格式，并对TFRecord格式的文件进行读取。开发环境：python3.5(python3+都可以)，Tensorflow r1.14(因为tensorflow新版本对一些函数重新进行归类，所以有些方法在你的版本会不在一个类中，这些可以根据自己版本查阅官方文档)
<!-- more-->

# 文章架构
1. 将CIFAR-10数据集转化成TFRecord格式
- 理解tf.train.Example()函数实现数据的序列化
2. 实现预处理函数
- 正则化、one-hot向量编码

# 将CIFAR-10数据集转化成TFRecord格式

将数据存储为TFRecord的格式，首先要对数据进行序列化处理。tf.train.Example()定义了将数据进行序列化时的格式。下面代码是Example 的定义：

```Python
message Example{
    Features features = 1;
};
message Features{
    map<string, Feature> feature = 1;
};
message Feature{
    oneof kind{
        BytesList byte_list = 1;
        FloatList float_list = 2;
        Int64List int64_list = 3;
    }
} 
```

由上述代码可以看出，tf.train.Example中包含一个根据属性名称获得属性值的字典映射map<string,  Feature>, 其中属性名称为一个字符串，属性值可以为字符串列表（BytesList）、实数列表（FloatList）、整数列表（Int64List）。因此，在本次实验中我定义了两个列表函数(函数都设置成私有)：字符串列表，整数列表，具体代码如下所示。
```Python
def _int64_feature(value):
    """
    将特征数据加入int列表中
    :param value: 加入的int类型特征
    :return:
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    """
    将特征数据加入字节列表中
    :param value: 特征数据（字符串）
    :return:
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
```

大概了解了序列化的使用后，接下来就可以进行数据集的转化。有个问题需要考虑，我们需要保存什么数据进TFRecord里呢？答案肯定是数据集中的图片数字数据、标签、或者再把图像数据的形状也保存下来。所以，我们有了三个属性先进行序列化，即先对图片数字数据、标签、形状数据进行序列化处理。下面函数展示如何转化成TFRecord的过程。
```Python
def convert_to_tfrecord(images, labels, save_file_dir, save_file_name):
    """
    将图片、标签数据打包成TFRecord格式
    :param images: 图片数字数据
    :param labels: 标签数据
    :param save_file_dir: 打包文件的存放目录
    :param save_file_name: 打包文件的名称
    :return:
    """
    # 文件存放路径
    file_path = os.path.join(save_file_dir, save_file_name+'.tfrecords')
    # 获得打包数据的样本数量
    num_examples = labels.shape[0]
    if images.shape[0] != num_examples:
        raise ValueError("图片数量：{}和标签数量{}不一致".format(images.shape[0], num_examples))

    # 图片的宽度、高度、深度
    image_width = images.shape[1]
    image_height = images.shape[2]
    image_depth = images.shape[3]

    print("进行{}文件的打包..".format(file_path))
    # 获得TFRecord的写入对象
    writer = tf.io.TFRecordWriter(path=file_path)

    # 循环对每一个数据进行转换
    for index in range(num_examples):
        # 将图片矩阵转化成string格式
        image_to_string = images[index].tostring()
        # 定义需要保存的属性字典，并添加到总的属性中
        feature = {'width': _int64_feature(image_width),
                   'height': _int64_feature(image_height),
                   'depth': _int64_feature(image_depth),
                   'label': _int64_feature(int(labels[index])),
                   'image_raw':_bytes_feature(image_to_string)}
        features = tf.train.Features(feature=feature)
        # 定义Example类，将前面定义的属性字典写入这个数据结构中
        example = tf.train.Example(features=features)
        # 使用SerializeToString()函数将协议内存块序列化为一个字符串，并写入到TFRecord文件中
        writer.write(example.SerializeToString())
    print("pack end...")
    # 关闭写入流
    writer.close()
```
## 理解原始图像数据集
原始训练数据中每个批次数据是用numpy数组中矩阵所表示的，即形状为(10000, 3072)。其中，10000表示样本数据的数量，行向量（3072）表示一个32×32像素的颜色图像。因为这个项目将使用CNN来进行分类任务，为了输送一个图像数据给CNN模型，那输入张量的维度应该是（width×height×num_channel）或者（num_channel×width×height）其中一个。选择那一种输入张量是由tensorflow中的conv2d函数决定的，tensorflow中默认使用第一个。

###  如何重塑成(width, height, channel)的格式？
对于一张图片的行向量是3072（1024个Red、1024个Green、1024个Blue），若是以32* 32 * 3计算则有相同数量的元素。为了重塑行向量为width×height×num_channel形式，需要两个步骤。第一个步骤是使用numpy中的reshape函数，第二个步骤是使用transpose函数。  

通过numpy官方网站的定义，reshape函数是转化一个数组成一个新的形状且没有改变数据。在这里，那句没有改变数据是很重要的，因为这样的转化是能损失数据的。reshape操作应该被分成二个更加详细的步骤。  

下面用逻辑概念描述的：

1. 划分行向量为三块，每一块意味着每个颜色通道（RGB）。行向量变成3*1024矩阵，整体的张量变成（10000， 3， 1024）。
2. 再把3块中的每一块划分出32,32是一个图像的宽度和高度，即1024=32*32。整体的张量变成（10000, 3, 32, 32）。  
![reshape 和 transpose](/assets/cifar_10/2.png)  

现在一个图像被表示成（num_channel, width, height）的格式。然而，这不是tensorflow和matplotlib所要求的格式。他们要求的格式是（width,height,num_channel）,你需要交换每一轴的顺序，也就是transponse（转置）的由来。  

转置可以取一个轴的列表并且每个值特定一个它想要移动的位置。在本实验中调用numpy 中transpose函数，设置参数为（0 ，2 , 3， 1），将形状为（10000， 3, 32， 32）转置成（10000，32， 32, 3）。

## 理解原始标签
每张图片对应一个类别，范围是0-9。通过打开文件batches.meta可以发现标签名词和对应的数字。
{'label_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], 'num_vis': 3072}。标签数组中顺序的索引代表对应的数字。

## 代码实现
```Python
"""
实现数据集的加载
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import os


class DataSetOp(object):
    """
    操作数据集类
    """
    def __init__(self, data_path):
        self.data_path = data_path

    def load_cifar_10_data(self):
        """
        装载CIFAR-10数据集
        :return: 训练特征、标签、标签名称
        """
        # 定义数据类型名称数组
        data_type = ['data_batch_1', 'data_batch_2', 'data_batch_3',
                       'data_batch_4', 'data_batch_5', 'test_batch', 'batches.meta']
        # 定义训练数据集、标签,测试集，标签名称
        train_data = []
        train_label = []
        test_data = []
        label_name = []
        # 打开数据集文件
        for i in range(len(data_type)):
            file_path = os.path.join(self.data_path, data_type[i])  # 数据文件路径
            # 使用上下文环境打开文件
            with open(file_path, mode='rb') as file:
                # 使用‘latin1’编码格式进行编码
                batch = pk.load(file, encoding='latin1')
            if i < 5:  # 处理训练数据批次
                # 将特征行向量重塑成（10000，3，32，32）再转置成（10000,32,32,3）
                train_batch = batch['data'].reshape(len(batch['data']), 3, 32, 32).transpose(0, 3, 2, 1)
                label_batch = batch['labels']
                # 保存所有批次的特征、标签
                train_data.append(train_batch)
                train_label.append(label_batch)
            elif i==5: # 处理测试数据批次
                test_data = batch['data'].reshape(len(batch['data']), 3, 32, 32).transpose(0, 3, 2, 1)
            else:
                label_name = batch['label_names']
        return train_data, train_label, test_data, label_name


if __name__ == '__main__':
    cfiar = DataSetOp(data_path='/home/xiaonan/Dataset/cifar-10/')
    train_data, train_label, test_data, label_name = cfiar.load_cifar_10_data()
    print(train_data[0].shape)
    print(train_label[0])
    print(test_data.shape)
    print(label_name)
```

# 实现预处理函数
你可能会发现像一些框架（如Tensorflow）提供了类似的预处理的函数，但是我们不能过度的依赖框架。我相信我可以构造出我自己的模型更好，或者去重新、实现最先进的模型（最新论文中提及的）。对于本次实验，我将实现正则化（normalize）和one-hot编码函数

## 正则化（Normalize）
正则化函数接受数据x，并且作为一个规范化Numpy数组返回。x可以是任何东西，且也可以为N-D数组。在本篇实验中，一个图像将变成3-D数组。使用了Min-Max 正则化（y=（x-min）/(max-min)）这个技巧，