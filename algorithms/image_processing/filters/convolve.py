"""
卷积运算
"""
import cv2 as cv2
import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 读取图片
    image_path = '../cat.jpg'
    image = cv2.imread(image_path)
    print('image shape:', image.shape)
    cv2.imshow('RGB',image)

    # 将彩色图片转化成灰度图片(将三维图像转化成二维图像)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print('image RGB to gray', image_gray.shape)
    cv2.imshow('GRAY',image_gray)

    # 进行卷积运算
    # 定义卷积核
    convolve_kernel =

    cv2.waitKey(0)



