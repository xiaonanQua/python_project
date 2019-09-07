"""
实现猫狗识别
"""
import torch
import torch.nn as nn
from torchvision import transforms, datasets

# 使用transforms.Compose()函数拼凑数据处理的操作
data_transform = transforms.Compose([
    transforms.Resize(size=84),  # 将PIL格式的图像重塑成相应的大小size
    transforms.CenterCrop(84),  # 以给定的大小size，居中裁剪PIL格式的图像
])