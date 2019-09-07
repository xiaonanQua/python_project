# 加载所需的包
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import datasets, transforms
import os

# 定义超参数
input_size = 784  # 输入图像的大小
num_classes = 10  # 类别数量
num_epoch = 10  # 周期
batch_size = 50  # 批次大小
learning_rate = 0.001  # 学习率

# 判断当前是否能使用GPU运算
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# MNIST数据集路径
dataset_path = "/home/xiaonan/Dataset/mnist/"
# 加载训练、测试数据集
train_dataset = datasets.mnist.MNIST(root=dataset_path,   # 数据集目录，没有则下载
                                     train=True,  # 加载训练集
                                     transform=transforms.ToTensor(),  # 将PIL格式的图像和numpy.ndarray转化成张量
                                     download=True  # 若不存在数据集则进行下载
                                     )
test_dataset = datasets.mnist.MNIST(root=dataset_path,
                                    train=False,
                                    transform=transforms.ToTensor())
print(train_dataset, test_dataset)

# 训练、测试数据加载
train_loader = data.DataLoader(dataset=train_dataset,
                               batch_size=batch_size,
                               shuffle=True)
test_loader = data.DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=True)
print(train_loader, test_loader)


# 建立逻辑回归模型，继承Module
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        # 子类继承父类构造函数
        super(LogisticRegression, self).__init__()
        # 建立逻辑回归模型
        self.liner = nn.Linear(in_features=input_size,
                               out_features=num_classes)

    def forward(self, x):
        """
        重写模型的前向传播
        :param x: 输入特征
        :return: 前向传播结果
        """
        # 向逻辑回归模型输入特征
        out = self.liner(x)
        return out


# 获得逻辑回归模型
model = LogisticRegression(input_size, num_classes)
model.to(device)

# 定义损失和优化函数
# 二分类一般用交叉熵损失
criterion = nn.CrossEntropyLoss()
# 随机梯度下降
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epoch):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 28*28).to(device)
        labels = labels.to(device)

        # 所有梯度重置为零
        optimizer.zero_grad()
        # 将图片数据输入模型中,得到模型预测值
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播，计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()

        # 输出每批次的loss结果
        if (i+1) % 100 == 0:
            print('Epoch:[%d/%d], Step:[%d,%d], Loss:%.4f'
                  %(epoch+1, num_epoch, i+1, len(train_dataset)//batch_size, loss.item()))

# 测试模型的效果
correct = 0
total = 0
for images, labels in test_loader:
    images = images.view(-1, 28*28).to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()
print('10000张测试图片上的测试准确度:%d%%'%(100*correct/total))



