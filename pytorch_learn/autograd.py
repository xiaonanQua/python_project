import torch
from torch.autograd import Variable

# 使用Tensor新建一个Variable
x = Variable(torch.ones([2,3]))
print(x)
print(torch.sum(x))