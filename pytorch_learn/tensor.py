from __future__ import print_function
import torch
import numpy as np

# 构建5*3的矩阵
x = torch.Tensor(5, 3)
print(x)
# 使用[0,1]均匀分布随机初始化二维数组
x = torch.rand((5, 3))
print(x)
# 查看x的形状,列的个数
print(x.size(), x.size()[0], x.size(1))
print()

# 加法的三种写法
y = torch.randn(size=(5, 3))
z = torch.Tensor(5, 3)
print(z)
print(x + y)
print(torch.add(x, y))
torch.add(x, y, out=z)
print(z)
print()

# Tensor和numpy之间互相转化，若Tensor不支持的操作，可先转化成numpy数组处理，之后再转化成Tensor
a = torch.ones(size=[2, 3])
print(a)
b = a.numpy()  # 将Tensor转化成Numpy
print(b)
a = np.ones(shape=[2, 3])
print(a)
b = torch.from_numpy(a)  # 将numpy数据格式转化成Tensor
print(b)
print()

# Tensor对象可通过.cuda方法转化成GPU的Tensor，从而享受GPU的加速运算
if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    print(x + y)

