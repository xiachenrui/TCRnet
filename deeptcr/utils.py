# -*- coding: UTF-8 -*-

import torch
import os
import pickle
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.autograd import Variable
# from load_file import one_hot


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataList=None, labelList=None):
        self.data = dataList
        self.dataLen = len(dataList)
        self.label = labelList

    def __getitem__(self, index):
        vector = self.data[index]
        label = self.label[index]
        return vector, label

    def __len__(self):
        return self.dataLen

# dataset_TCR = Dataset(vector, label)
# train_loader = torch.utils.data.DataLoader(dataset=dataset_TCR, batch_size=64, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=dataset_TCR, batch_size=64, shuffle=True)
## 打印一次输入的数据
# for a, b in train_loader:
#     print("a=")
#     print(a.shape)
#     print("b=")
#     print(b.shape)
#     break
# for a, b in test_loader:
#     print("a=")
#     print(a.shape)
#     print("b=")
#     print(b.shape)
#     break
#


# def read_data_train():
#     train_dataset = datasets.MNIST(root=r'C:\Users\14121\Desktop\mnist',
#                                 train=True,
#                                 transform=transforms.ToTensor(),
#                                 download=True
#                                )
#     return train_dataset
# def read_data_test():
#     test_dataset = datasets.MNIST(root=r'C:\Users\14121\Desktop\mnist',
#                                 train=False,
#                                 transform=transforms.ToTensor())
#     return test_dataset
#
# #read_data_train()