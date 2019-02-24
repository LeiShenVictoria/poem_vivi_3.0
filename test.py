# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
# import cPickle as pickle
import json
import torch

import torch
import torch.utils.data as Data
from torch.utils.data.dataset import Dataset
import numpy as np

t = torch.Tensor((4))
print(t)
print(t.size())
t = t.unsqueeze(0)
print(t)
print(t.size())
t = t.unsqueeze(0)
print(t)
print(t.size())

# li = torch.Tensor([[1],[2],[3]])
# a = [].append(li.size(0))
# print(a)
# print(li.size(0))

# a = [1]*5
# print(a)
# b = torch.Tensor([[[1, 2, 3], [4, 5, 6]],[[7, 8, 9], [10, 11, 12]],[[13, 14, 15], [16, 17, 18]],[[19, 20, 21], [22, 23, 24]]])
# print(b)
# print(b.size())
# print(b[:, 0])
# print(b[:, 0].size())

# a = torch.Tensor([[[1, 2, 3], [4, 5, 6]],[[7, 8, 9], [10, 11, 12]],[[13, 14, 15], [16, 17, 18]],[[19, 20, 21], [22, 23, 24]]])
# b = torch.Tensor([[5],[6],[7],[8]])
# print(a)
# print(a.size())
# c = torch.transpose(a,0,1)
# print(c)
# print(c.size())

# class TxtDataset(Dataset):  # 这是一个Dataset子类
#     def __init__(self):
#         self.Data = np.asarray([[1, 2], [3, 4], [2, 1], [6, 4], [4, 5]])  # 特征向量集合,特征是2维表示一段文本
#         self.Label = np.asarray([1, 2, 0, 1, 2])  # 标签是1维,表示文本类别
# 
#     def __getitem__(self, index):
#         txt = torch.Tensor(self.Data[index])
#         label = torch.Tensor(self.Label[index])
#         return txt, label  # 返回标签
# 
#     def __len__(self):
#         return len(self.Data)
# 
# Txt = TxtDataset()
# print(Txt.Data[1])
# print(type(Txt.Data[1]))
# print(Txt.Label[1])
# print(type(Txt.Label[1]))
# print(Txt.__len__())


# x=np.array([1,4,3,-1,6,9])
# print(x)
# y = np.argsort(-x)
# print(y)
# z = np.argsort(y)
# print(z)
# x = x[y]
# print(x)
# x_p = torch.nn.utils.rnn.pack_padded_sequence(x, len(x), batch_first=True)
# print(x_p)

# 
# t0 = torch.Tensor([])
# # t0 = t0.float()
# t1 = torch.Tensor([[1], [2], [3])
# t2 = torch.Tensor([[4.4], [5.5], [6.6]])
# t1 = torch.reshape(t1, (1,-1,1))
# t2 = torch.reshape(t2, (1,-1,1))
# print(t0)
# print(t0.size())
# print(t1)
# print(t1.size())
# print(t2)
# print(t2.size())
# t0 = torch.cat((t0, t1, t2), 0)
# print(t0)
# print(t0.size())


# # 虚构要训练的数据
# x = torch.linspace(11, 20, 10)  # 在[11, 20]里取出10个间隔相等的数 (torch tensor)
# y = torch.linspace(20, 11, 10)
# 
# BATCH_SIZE = 5  # 每批需要训练的数据个数
# 
# # 把tensor转换成torch能识别的数据集
# torch_dataset = Data.TensorDataset(x, y)
# 
# # 把数据集放进数据装载机里
#  loader = Data.DataLoader(
#     dataset=torch_dataset,  # 数据集
#     batch_size=BATCH_SIZE,  # 每批需要训练的数据个数
#     shuffle=True,  # 是否打乱取数据的顺序(打乱的训练效果更好)
#     num_workers=2,  # 多线程读取数据
# )
# 
# # 批量取出数据来训练
# for epoch in range(3):  # 把整套数据重复训练3遍
#     for step, (batch_x, batch_y) in enumerate(loader):  # 每次从数据装载机里取出批量数据来训练
#         # 以下为训练的地方
#         # …………
#         # 把每遍里每次取出的数据打印出来
#         print('Epoch:', epoch, '|Step:', step,  # Epoch表示哪一遍, Step表示哪一次
#               'batch x:', batch_x.numpy(),
#               'batch y:', batch_y.numpy(),
#               )

# t = torch.Tensor([1.1,2.2,3.3,0.0])
# sort, id = torch.sort(t, descending=True)
# print(t)
# print(sort)
# print(id)
# print(id.size())
# li = id.tolist()[0]
# print(li)

# f = open('model/encoder_params.pkl', encoding='utf-8')
# f1 = pickle.load(f)
# print(f1)
# f.close()

# with open('resource/word2vec_poem_58k.txt', 'rb') as f:
#     wordMisc = pickle.load(f) 
#     # print(type(wordMisc))
#     word2id = wordMisc['word2id']
#     id2word = wordMisc['id2word']
#     P_Emb = wordMisc['P_Emb']
# dic = {}
# dic['word2id'] = word2id
# dic['id2word'] = id2word
# print(word2id['UNK'])
# print(len(id2word))
# print(len(P_Emb))

# with open('word2id.json', 'w') as f1:
#     json.dump(word2id, f1)
# with open('id2word.json', 'w') as f2:
#     json.dump(id2word, f2)
# with open('emb.txt', 'w') as f3:
#     for vec in P_Emb:
#         for dim in vec:
#             f3.write(str(dim))
#             f3.write(' ')
#         f3.write('\n')

# with open('resource/id2word.json', 'r') as f:
#     id2word = json.load(f)
#     print(type(id2word))
#     print(len(id2word))
#     print(id2word['4'])
# 
# with open('resource/emb.txt', 'r') as f3:
#     emb = []
#     for line in f3.readlines():
#         vec = []
#         for dim in line.split(' ')[:-1]:
#             vec.append(float(dim))
#         emb.append(vec)
#     print(len(emb[0]))

# with open('resource/emb.txt', 'r') as f3:
#     emb = []
#     for line in f3.readlines():
#         vec = []
#         for dim in line.split(' ')[:-1]:
#             vec.append(float(dim))
#         emb.append(vec)
# print(len(emb))
# print(len(emb[937]))
# print(emb[937][:10])
# x = torch.Tensor([937])
# print(type(x))
# print(int(x[0]))
# embedded = emb[int(x[0])]
# print(type(embedded))
# print(len(embedded))
# print(embedded[:10])
# print(emb[937] == embedded)
# embedded2 = torch.tensor(embedded, dtype=torch.long).view(1, 1, -1)
# print(type(embedded2))
# print(embedded2.shape)
# print(embedded2)
# print(embedded2[0][0][:10])

# list = [1,2,3,4]
# list = [0.046891, -0.102716, 0.126854, -0.040811, 0.018327, 0.04187, -0.033878, -0.023025, -0.053526, -0.153362]
# tensor = torch.tensor(list, dtype=torch.double).view(1, 1, -1)
# print(tensor)

# def read_test_data():
#     print('read test set')
#     input_set = []
#     lines = []
#     for line in open('resource/testset.txt', 'r', encoding='utf-8').readlines():
#         lines.append(line)
#         input_words = line.split(' ')
#         input_ids = [word2id.get(word, vocab_size - 1) for word in input_words]  # default = 4776 '-' ?
#         input_ids.append(EOS_token)
#         input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).view(-1, 1)
#         input_set.append(input_tensor)
#     print('read test set done')
#     return input_set, lines