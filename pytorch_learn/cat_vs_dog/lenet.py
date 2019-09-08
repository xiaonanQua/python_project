"""
构建网络结构
"""
import torch
import torch.nn as nn
import torch.nn.functional as func


class LeNet(nn.Module):
    """
    继承父类模型来构建新模型
    """
    def __init__(self):
        # 继承父类构造函数
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=6,
                               kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16*18*18, out_features=800)
        self.fc2 = nn.Linear(in_features=800, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=2)

    def forward(self, x):
        """
        重写前向传播方法
        :param x: 输入特征
        :return:
        """
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        # 重塑数据的形状
        x = x.view(-1, 16*18*18)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        logit = self.fc3(x)
        return logit