# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import torch
import json
from torch.utils.data.dataset import Dataset
import torch.utils.data as Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = True if torch.cuda.is_available() else False

# word2vector

# id2word = {'0':'START' '1':'END' '2':'/' '3':'START1' '4':'END1' '4776':'-' '4777':'UNK'} len=4778
# word2id len=4777
# emb len=4777
def Word2vec():
    with open('resource/id2word.json', 'r') as f1:
        id2word = json.load(f1)
    with open('resource/word2id.json', 'r') as f2:
        word2id = json.load(f2)
    with open('resource/emb.txt', 'r') as f3:
        emb = []
        for line in f3.readlines():
            vec = []
            for dim in line.split(' ')[:-1]:
                vec.append(float(dim))
            emb.append(vec)
    vocab_size = len(emb)
    word2count = {} 
    for p in open('resource/poem_58k_theme.txt', 'r', encoding='utf-8').readlines():
        target = p.strip().split('==')[1]
        lines = target.split('\t')
        for l in lines:
            words = l.split(' ')
            for w in words:
                temp_w = w
                if temp_w in word2count:
                    word2count[temp_w] += 1
                else:
                    word2count[temp_w] = 1
    return word2id, id2word, emb, word2count, vocab_size

word2id, id2word, emb, word2count, vocab_size = Word2vec()
SOS_token = 3
EOS_token = 4
PAD_token = 4776 #
UNK_token = 4777

# data processing

INPUT_MAX_LENGTH = 32
MAX_LENGTH = 41 

class PoetryRNNData(Dataset):
    def __init__(self, data, max_topic_counts=INPUT_MAX_LENGTH, max_poetry_length=MAX_LENGTH):
        # when chunk size = 120, evenly divide; = 259, leave one out
        # most poetries have length around 40 - 80
        # data is nested list of word idx 嵌套列表 
        assert any(isinstance(i, list) for i in data)
        # assert是一种插入调试断点到程序的一种便捷的方式。如果assert的内容不成立，则抛出AssertionError异常，后面程序不执行
        # isinstance(a, b)判断a变量是否是b类型
        # 本句想确定data中所有元素均为list类型

        # topic words
        topics = [i[0] for i in data]
        self.topic_lens = torch.LongTensor([min(len(x), max_topic_counts) for x in topics])

        # poetry text
        data = [i[1] for i in data]
        self.lengths = torch.LongTensor([min(len(x), max_poetry_length) for x in data])
        # self.lengths = torch.LongTensor([min(len(x), max_poetry_length) - 1 for x in data]) # -1?
        self.lens = len(self.lengths)

        # pad data
        max_len = min(max(self.lengths), max_poetry_length)

        self.topics = torch.zeros((self.lens, max_topic_counts)).long()
        self.data = torch.zeros((self.lens, max_len)).long()
        # self.target = torch.zeros((self.lens, max_len)).long()
        for i in range(self.lens):
            TL = min(self.topic_lens[i], max_topic_counts)
            self.topics[i, :TL] = torch.LongTensor(topics[i][:TL])

            L = min(self.lengths[i], max_poetry_length)
            self.data[i, :L] = torch.LongTensor(data[i][:L])  # 0:L
            # self.target[i, :L] = torch.LongTensor(data[i][1:(L + 1)]) # 1:L+1 target?
        if use_cuda:
            self.topics = self.topics.cuda()
            self.topic_lens = self.topic_lens.cuda()
            self.data = self.data.cuda()
            # self.target = self.target.cuda()
            self.lengths = self.lengths.cuda()

    def __len__(self):
        return self.lens

    def __getitem__(self, index):
        out = (
            self.topics[index, :], self.topic_lens[index], self.data[index, :],
            # self.topics[index, :], self.topic_lens[index], self.data[index, :], self.target[index, :],
            self.lengths[index])
        return out

def indexesFromSentence(sentence):
    return [word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(sentence):
    indexes = indexesFromSentence(sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(pair[0])
    target_tensor = tensorFromSentence(pair[1])
    return (input_tensor, target_tensor)

def pad_seq(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq

def read_train_data(file):
    print('read training set')
    pairs_tensor = []
    pairs_li = []
    lines = 0

    for line in open(file, 'r', encoding='utf-8').readlines():
        lines += 1
        source, target = line.split('==')
        # ?
        source_words = ('START1 ' + source + ' END1').split(' ') # 用了START1和END1
        # source = source.replace(' - ', ' ') # 去掉-
        # source_words += source.split(' ')
        target = target[:-2] if target.find('\r\n') > -1 else target[:-1]
        # target_words += target.replace('\t', ' ').split(' ') + target.split('\t')[0].split(' ') # 去掉/
        target_words = target.replace('\t', ' / ').split(' ') + ['/'] + target.split('\t')[0].split(' ') # 用5个句子训练
        
        source_ids = [word2id.get(word, vocab_size - 1) for word in source_words] # default = 4776 '-' ?
        source_tensor = torch.tensor(source_ids, dtype=torch.long, device=device).view(-1, 1)
        
        target_ids = [word2id.get(word, vocab_size - 1) for word in target_words]
        target_ids.append(EOS_token)
        target_tensor = torch.tensor(target_ids, dtype=torch.long, device=device).view(-1, 1)
        
        pairs_tensor.append((source_tensor, target_tensor))
        pairs_li.append([source_ids, target_ids])
        
    print('read traning set done')
    return pairs_li, lines # tmp

# 与上一个数据读取命名统一 input或source取一个 
def read_test_data():
    print('read test set')
    input_set = []
    lines = []
    for line in open('resource/testset.txt', 'r', encoding='utf-8').readlines():
        line = line[:-1] # 保留-，和训练一致
        # line = line[:-1].replace(' - ', ' ') # 去掉末尾的\n和-
        lines.append(line)
        input_words = line.split(' ')
        input_ids = [word2id.get(word, vocab_size - 1) for word in input_words]  # default = 4776 '-' ?
        input_ids.append(EOS_token)
        # input_set.append(input_ids)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        input_set.append(input_tensor)
    print('read test set done')
    return input_set, lines

def read_val_data():
    return read_train_data(file='poem_50_val.txt')