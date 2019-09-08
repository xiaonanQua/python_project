import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable


class LeNet(nn.Module):
    def __init__(self):
        # 继承父类构造函数
        super(LeNet, self).__init__()
        # 卷积层1，’1‘表示输入图片为单通道，‘6’表示输出通道数， '5’表示卷积核为5*5
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=6,
                               kernel_size=5)
        # 卷积层2
        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=5)
        # 全连接层
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # 卷积->激活->池化
        x = func.max_pool2d(func.relu(input=self.conv1(x)), (2, 2))
        x = func.max_pool2d(func.relu(input=self.conv2(x)), 2)
        # 重塑x的形状，‘-1’表示自适应
        x = x.view(x.size(0), -1)
        # 全连接->激活
        x = func.relu(self.fc1(x))
        x = func.relu(input=self.fc2(x))
        # 输出，逻辑回归
        logit = self.fc3(x)
        return logit


if __name__ == '__main__':
    net=LeNet()
    print(net)
    # 返回网络中的学习参数
    params = list(net.parameters())
    print(params)
    # 输出网络中的学习参数及其名称
    for name, parameters in net.named_parameters():
        print(name, ':', parameters.size())
    print()

    # forward函数的输入和输出都是Variable，只有Variable才具有自动求导的功能，Tensor是没有的，所以在输入时需要把Tensor封装成Variable
    input = Variable(torch.randn(size=(1, 1, 32, 32)))
    out = net(input)
    print(out.size())
    # 所有参数的梯度清零
    net.zero_grad()
    out.backward(Variable(torch.ones(1, 10)))
    target = Variable(torch.arange(0, 10))
    criterion = nn.MSELoss()
    loss = criterion(input=input, target=target)
    print(loss)
