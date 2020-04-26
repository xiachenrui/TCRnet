# -*- coding: UTF-8 -*-

import xlrd
import sys
import pickle
import os
import pandas as pd
from numpy import argmax
import torch
from torch.utils.data import Dataset
import numpy as  np


def one_hot(cdr3, max_length, head_cut=3, end_cut=2):
    max_length = int(max_length)
    head_cut = int(head_cut)
    end_cut = int(end_cut)
    max_length = max_length -head_cut - end_cut

    all_data = []
    for index, row in cdr3.iterrows():
        un_cut_seq = row["cdr3"]
        seq = un_cut_seq[head_cut:-end_cut]

        # define universe of possible input values
        alphabet = 'ACDEFGHIKLMNPQRSTVWY'
        # define a mapping of chars to integers
        char_to_int = dict((c, i) for i, c in enumerate(alphabet))
        int_to_char = dict((i, c) for i, c in enumerate(alphabet))
        # integer encode input data
        integer_encoded = [char_to_int[char] for char in seq]
        # one hot encode
        onehot_encoded = list()

        for value in integer_encoded:
            letter = [0 for _ in range(len(alphabet))]
            letter[value] = 1
            # 在这设置报错机制，防止稀有氨基酸干扰运算
            onehot_encoded.append(letter)
            # inverted = int_to_char[argmax(onehot_encoded[0])]
        one_hot_vector = onehot_encoded
        # print(len(one_hot_vector))
        null_aa_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        while len(one_hot_vector) < max_length:
            one_hot_vector.append(null_aa_vector)
        # print(one_hot_vector)
        all_data.append(one_hot_vector)
    return all_data
    # print(len(all_data))


# def read_file():
cur_dir = os.path.abspath('..')
data_dir = os.path.join(cur_dir, "data")
all_sample_labels = []
all_sample_vectors = np.empty((0,23,20))
# all_sample_max_length = 0
# 在这里应该考虑所有样本最大长度不一样的情况
for file_name in os.listdir(data_dir):
    if "csv" in file_name:
        full_file_name = os.path.join(data_dir, file_name)
        csv_data = pd.read_csv(full_file_name)
        cdr3 = csv_data["cdr3"]
        cdr3 = cdr3[~cdr3.isin(["None"])]
        label = pd.DataFrame(columns=['label'])
        cdr3 = pd.concat([cdr3, label], axis=1)
        length = cdr3['cdr3'].str.len()
        max_length = length.max()
        sample_vector = one_hot(cdr3, max_length)
        sample_length = len(sample_vector)
        if "N" in file_name:
            sample_label = np.zeros(sample_length)
        elif "P" in file_name:
            sample_label = np.ones(sample_length)
        else:
            print("Please change your filename into required format")
        sample_vector = np.array(sample_vector)
        all_sample_vectors = np.array(all_sample_vectors)
        # print(all_sample_vectors.shape)
        # print(sample_vector.shape)
        all_sample_labels = np.append(all_sample_labels, sample_label)
        all_sample_vectors = np.concatenate((all_sample_vectors, sample_vector), axis=0)
        # print(all_sample_labels.shape)
        # print(all_sample_vectors.shape)

with open('all_sample_labels.pickle', 'wb') as f:
    pickle.dump(all_sample_labels, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('all_sample_vectors.pickle', 'wb') as f:
    pickle.dump(all_sample_vectors, f, protocol=pickle.HIGHEST_PROTOCOL)






    # 循环工作簿的所有行
    # for row in rsheet.get_rows():
    #     product_column = row[1]  # 品名所在的列
    #     product_value = product_column.value  # 项目名
    #     if product_value != '品名':  # 排除第一行
    #         price_column = row[4]  # 价格所在的列
    #         price_value = price_column.value
    #         # 打印
    #         print("品名", product_value, "价格", price_value)
