"""
实现猫狗识别
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import ConcatDataset, DataLoader
from torch.autograd import Variable
import os
from pytorch_learn.cat_vs_dog.cat_cog_dataset import CatDogDataset
from pytorch_learn.cat_vs_dog.lenet import LeNet

# 文件路径配置
root_dir = '/home/xiaonan/Dataset/cat_dog/'
train_dir = os.path.join(root_dir, 'train/')
test_dir = os.path.join(root_dir, 'test/')
train_files = os.listdir(train_dir)
test_files = os.listdir(test_dir)
cat_files = [cat for cat in train_files if 'cat' in cat]
dog_files = [dog for dog in train_files if 'dog' in dog]
print('输出数据列表\n', train_files[0], len(test_files), len(cat_files), dog_files)

# 使用transforms.Compose()函数拼凑数据处理的操作
data_transform = transforms.Compose([
    transforms.Resize(size=84),  # 将PIL格式的图像重塑成相应的大小size
    transforms.CenterCrop(84),  # 以给定的大小size，居中裁剪PIL格式的图像
    transforms.ToTensor(),  # 将PIL格式或者numpy.ndarray的图像重塑成tensor，将图片像素转化为0-1的数字
    # 用均值和标准差来归一化每个通道的张量图像，将数据由原来的0到1转化成-1到1
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# 通过数据集函数获得猫，狗，测试数据集对象
cats = CatDogDataset(file_list=cat_files,
                     file_dir=train_dir,
                     transform=data_transform)
dogs = CatDogDataset(file_list=dog_files,
                     file_dir=train_dir,
                     transform=data_transform)
testset = CatDogDataset(file_list=test_files,
                        file_dir=test_dir,
                        mode='test',
                        transform=data_transform)
# 组合猫、狗数据集
catdogs = ConcatDataset([cats, dogs])
# 训练、测试数据集加载器
train_loader = DataLoader(dataset=catdogs,
                          batch_size=4,
                          shuffle=True,
                          num_workers=4  # 开启多线程，用4个线程进行数据的加载
                          )
test_loader = DataLoader(dataset=testset,
                         batch_size=4,
                         shuffle=True,
                         num_workers=4)

# 实例化LeNet网络对象
le_net = LeNet()

# 定义交叉熵损失函数和随机梯度下降优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=le_net.parameters(),lr=0.0001, momentum=0.9)

# 开始训练
for epoch in range(3):

    for i, data in enumerate(train_loader, 0):
        train_data, train_labels = data
        train_data, train_labels = Variable(train_data), Variable(train_labels)
        optimizer.zero_grad()
        logit = le_net(train_data)
        loss = criterion(logit, train_labels)
        loss.backward()
        optimizer.step()

        if i % 2000 == 1999:
            print('[%d%5d]loss:%.3f'%(epoch+1, i+1, loss.item()))
            running_loss = 0.0
print('training end')







