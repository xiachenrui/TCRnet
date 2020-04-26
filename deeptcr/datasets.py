# -*- coding: UTF-8 -*-

import torch
import os
import pickle
import numpy as np
# from utils import read_data_train,read_data_test
from utils import Dataset

cur_dir = os.getcwd()
if os.path.exists("all_sample_labels.pickle") and os.path.exists("all_sample_vectors.pickle"):
    with open("all_sample_labels.pickle", "rb") as f1:
        label = pickle.load(f1)
    with open("all_sample_vectors.pickle", "rb") as f2:
        vector = pickle.load(f2)
else:
    # 但是这条命令还是强烈依赖python环境，如果要写入exe必须进行修改
    str = "python ./load_file.py"
    p = os.system(str)
    with open("all_sample_labels.pickle", "rb") as f1:
        label = pickle.load(f1)
    with open("all_sample_vectors.pickle", "rb") as f2:
        vector = pickle.load(f2)

# 进行训练集和测试集的划分
length = vector.shape[0]
valid_length = label.shape[0]
if length != valid_length:
    print("ERROR, please check your input data")
row_indices = np.random.permutation(length)
# Make any necessary calculations.
# You can save your calculations into variables to use later.
line = int((vector.shape[0])*0.9)
# Create a Training Set
vector_train = vector[row_indices[0:line], :]
label_train = label[row_indices[0:line]]
# # Create a Cross Validation Set
# X_crossVal = X_norm[row_indices[600:800],:]
# Create a Test Set
vector_test = vector[row_indices[line:length], :]
label_test = label[row_indices[line:length]]

# print(vector_train.shape)
# print(vector_test.shape)
# print(label_train.shape)
# print(label_test.shape)

dataset_TCR_train = Dataset(vector_train, label_train)
dataset_TCR_test = Dataset(vector_test, label_test)


def train_loader_function(batch_size=64):
    train_loader = torch.utils.data.DataLoader(dataset=dataset_TCR_train,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True)
    return train_loader


def test_loader_function(batch_size=64):
    test_loader = torch.utils.data.DataLoader(dataset=dataset_TCR_test,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              drop_last=True)
    return test_loader

# train_dataset = read_data_train()
# test_dataset = read_data_test()
# batch_size = 64
# def train_loader_function():
#     train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=True,
#                                            drop_last=True)
#     return train_loader
# def test_loader_function():
#     test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False,
#                                           drop_last=True)
#     return test_loader

# for a,b in train_loader:
#     print("a=")
#     # print(a)
#     print(a.size())
#     print("b=")
#     print(b.size())
#     break

