---
layout: post
title: "详解CIFAR-10数据集转化成TFRecord格式和读取的过程"
date: 2019-8-02
tages: 
    -  DeepLearning
---
> 这篇文章，我将详细讲解如何将CIFAR-10数据集转化成TFRecord格式，并对TFRecord格式的文件进行读取。开发环境：python3.6(python3+都可以)，Tensorflow r1.14(因为tensorflow新版本对一些函数重新进行归类，所以有些方法在你的版本会不在一个类中，这些根据自己的版本查阅官方文档)[实验代码](https://github.com/xiaonanQua/python_project/blob/master/tensorflow_learn/basic_tensorflow/tfrecord.py)
<!-- more-->

# 文章架构
1. 将CIFAR-10数据集转化成TFRecord格式
- 理解tf.train.Example()函数实现数据的序列化
2. 读取TFRecord文件 

## 将CIFAR-10数据集转化成TFRecord格式

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

大概了解了序列化的使用后，接下来就可以进行数据集的转化。有个问题需要考虑，我们需要保存什么数据进TFRecord里呢？答案肯定是数据集中的图片数字数据、标签、或者再把图像数据的形状也保存下来。所以，我们有了三个属性先进行序列化，即先对图片数字数据、标签、形状数据进行序列化处理。下面函数展示如何转化成TFRecord的过程。在这个函数中，有四个参数：images(数据集中图片矩阵数组，传入的形状是[样本数量，宽度，高度，颜色通道]),labels（即标签数据，传入的形状是[样本数量，标签]），save_file_dir:文件保存的路径，save_file_name:文件保存名称。函数首先合成文件保存的路径、获得需要进行转化的样本总数量，若图片样本数量和标签样本数量不同则报错。然后，通过图片样本获得图片的宽度、高度、深度。接着，通过tf.io.TFRecordWriter()函数获取写入对象，注意此函数在以前版本是这样调用，即tf.train.TFRecordWriter()。  

接下来对数据进行序列化，并写入。通过循环迭代，对每个数据进行操作，即每一行数组数据。images(图像矩阵)需要转化成字符串，下面就是定义属性字典了，这是就用到之前定义的列表函数，对每个属性进行相应的转化。然后将获取的单个属性字典添加到总的属性字典中。下面通过Example类将总的属性字典添加到协议内存块中，再通过SerializeToString()序列化成字符串，最后写入TFRecord中。
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
## 读取TFRecord文件

对于TFRecord文件的读取，也和转化是一样的，不是直接就使用读取类那么简单的事。首先，你得创建一个队列对输入文件列表进行维护，这里的输入文件就是你需要读取的文件,在tensorflow中使用tf.train.string_input_producer([file_path])函数来创建文件队列。然后，你需要使用tf.TFRecordReader()定义读取类，去获取队列中的序列化样本。对于获得的序列化样本，我们需要使用解析工具进行解析，因为在进行TFRecord的转化时，我们定义了很多属性，这个工具就是解析出这些样本。也就是使用tf.io.parse_single_example()（若是以前版本是通过tf.parse_single_example()进行调用）函数解析读取的样本，使用tf.io.FixedLenFeature()（以前版本是tf.FixedLenFeature）函数对各个属性进行解析。最后，使用decode_raw()将原先转化成字符串的图像数据解码成图像数组，并将标签、形状属性由int64转化成int32。具体函数代码如下所示：
```Python
def read_tfrecords(file_dir, file_name):
    """
    读取tfrecords格式的文件
    :param file_dir: 文件目录
    :param file_name: 文件名称
    :return:images,labels,[width, shape, depth]
    """
    # 文件路径
    file_path = os.path.join(file_dir, file_name)

    # 创建TFRecordReader类的实例
    reader = tf.TFRecordReader()

    # 创建一个队列对输入文件列表进行维护
    file_queue = tf.train.string_input_producer([file_path])
    # 使用TFRecordReader.read()函数从文件中读取一个样本，使用.read_up_to()函数一次性读取多个样本
    _, serialized_example = reader.read(file_queue)
    # 使用sparse_single_example()函数解析读取的样本
    features = tf.io.parse_single_example(serialized_example,   features={ # 使用FixedLenFeature类对属性进行解析
                                           "image_raw": tf.io.FixedLenFeature([], tf.string),
                                           "label": tf.io.FixedLenFeature([], tf.int64),
                                           "width": tf.io.FixedLenFeature([], tf.int64),
                                           "height": tf.io.FixedLenFeature([], tf.int64),
                                           "depth": tf.io.FixedLenFeature([], tf.int64)
                                       }))

    # decode_raw()用于将字符串解码成图像对应的像素数组
    images = tf.decode_raw(features['image_raw'], tf.uint8)
    # 使用cast()函数进行类型转换
    labels = tf.cast(features['label'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    height = tf.cast(features['height'], tf.int32)
    depth = tf.cast(features['depth'], tf.int32)

    return images, labels, [width, height, depth]
```
另外需要说明一下，在主函数中进行读取操作时，需要用到协调器和开启多线程，因为使用了队列进行文件的读取，要不然读取的速度几乎为0。还有一点，对于文件的读取一定要放在启动多线程之前，若是放在之后线程无法捕捉这个操作，也就没有处理这个操作的线程。至于tensorflow中的队列、多线程知识，将在另外的篇章中进行说明。代码如下所示：
```Python
    with tf.Session() as sess:
        # 读取TFRecord格式的数据
        images, labels, shape = read_tfrecords(file_dir, file_name[0] + '.tfrecords')
        # 启动多线程处理输入数据
        coordinator = tf.train.Coordinator()  # 协调器
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)  # 开启所有队列线程
        for i in range(len(train_data)):
            image, label, shape = sess.run([images, labels, shape])
            print(label)
            print(shape)
            print(image.shape)
            break
```