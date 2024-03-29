"""
测试svhn数据集
"""

import data_load.config as cfg
import data_load.loaddata as load
import data_load.preprocessdata as preprocess
import data_load.queue_linked_list as queue
import matplotlib.pyplot as plt
cfg=cfg.Config()

if __name__ == '__main__':
    # 获取数据集、预处理、队列对象
    svhn = load.LoadDataSet()
    preprocess = preprocess.PreprocessData()
    queue = queue.QueueLink()
    # 加载svhn原始数据集
    images, labels = svhn.load_svhn(file_type='train')
    single_image = images[0]
    print("原始单张图片：{}".format(single_image))
    # plt.imshow(single_image)
    # plt.show()
    print("原始数据的图像形状：{}，标签形状：{}".format(images.shape, labels.shape))
    # 从训练集中划分出验证集
    train_data, train_labels, valid_data, valid_labels = preprocess.divide_valid_data(images, labels, valid_size=0.1)
    print("划分验证集...")
    print("train_data:{}, train_labels:{}\nvalid_data:{}, valid_labels:{}".format(train_data.shape,
                                                                                  train_labels.shape,
                                                                                  valid_data.shape,
                                                                                  valid_labels.shape))
    print("划分训练集成5批次...")
    # train_labels = preprocess.one_hot_encode(train_labels)
    print(train_data.shape, train_labels.shape)
    preprocess.divide_batch_data(images=train_data,
                                 labels=train_labels,
                                 num_batch=5,
                                 save_file_dir=cfg.save_svhn_batch_dir,
                                 save_file_name='svhn',
                                 save_type='tfrecord')
    print(train_labels[:30].tolist())
    train_data, train_labels = svhn.load_preprocess_data_tfrecord(cfg.save_svhn_batch_dir,
                                                                  'svhn_batch-1-of-5')
    # train_data, train_labels = svhn.load_preprocess_svhn_tfrecord(cfg.save_svhn_batch_dir,
    #                                                               'svhn_batch_*')
    print(train_data.shape, train_labels[0:30].tolist())
    # 进行数据归一化、标签onehot处理
    # train_data = preprocess.normalize(train_data[:90])
    # train_labels = preprocess.one_hot_encode(train_labels[:90])
    # print(train_data)
    # plt.imshow(train_data)
    # print(train_labels)
    # plt.show()

    # 获取批次并洗牌数据
    for i in range(4):
        batch_data, batch_labels = svhn.batch_and_shuffle_data(train_data[:90],
                                                               train_labels[:90],
                                                               30,
                                                               data_queue=queue,
                                                               shuffle=True)
        print(batch_data.shape, batch_labels.tolist())


