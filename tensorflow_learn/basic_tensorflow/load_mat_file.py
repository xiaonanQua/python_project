"""
加载matlab文件
"""
import scipy.io as sio
import os
import matplotlib.pyplot as plt
import h5py
import hdf5storage

dataset_dir = '/home/xiaonan/Dataset/svhn/'
file_name = ['train_32.mat', 'test_32×32.mat', 'digitStruct.mat', 'see_bboxes.m']

dataset_2_dir = '/home/xiaonan/Dataset/svhn/train/'

file_path = os.path.join(dataset_dir, file_name[0])
file2_path = os.path.join(dataset_2_dir, file_name[2])
file3_path = os.path.join(dataset_2_dir, file_name[3])

mat_contents = sio.loadmat(file_path)
with h5py.File(file2_path, 'r') as file:
    print(list(file))
    data = file['digitStruct']
    print(data[data['name'][0,0]].value)
# file2 = hdf5storage.loadmat(file2_path)
# print(file2.pop())
# file2 = sio.loadmat(file2_path)
# file3 = sio.loadmat(file3_path)

# print(file2)
# print(file3)

# x = mat_contents['X']
# y = mat_contents['y']
# print(mat_contents)
# print(x[:, :, :, 1])
# print(y[1])
# plt.imshow(x[:, :, :, 1])
# plt.show()