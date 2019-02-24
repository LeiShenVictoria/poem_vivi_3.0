# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import rnn
import train
import predict
import data_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('cuda:'+str(torch.cuda.is_available()))
print(device)

######################################################################
# Training and Evaluating
# =======================

train_mode = False # False: predict
# train params
training_set = 'resource/poem_58k_theme.txt'
learning_rate = 1
batch_size = 80
epochs = 10
# predict params
enc_model = 'model/enc_epoch=9.pkl'
dec_model = 'model/dec_epoch=9.pkl'

word2id, id2word, emb, word2count, vocab_size = data_utils.Word2vec()

hidden_size = 500
embed_size = 200
encoder1 = rnn.EncoderRNN_batch(embed_size, hidden_size).to(device)
attn_decoder1 = rnn.AttnDecoderRNN(hidden_size,  vocab_size, dropout_p=0.1, embed_size = embed_size).to(device)

if train_mode:
    train.trainIters_batch(training_set, encoder1, attn_decoder1, learning_rate, batch_size, epochs) # add lr
    # train.trainIters(encoder1, attn_decoder1, 500, print_every=100, learning_rate=0.1, save_every=100)  # add lr
else: # predict, load existing model
    encoder = encoder1
    decoder = attn_decoder1
    if device == 'cpu': # gpu下训的模型cpu不能使用 （改了也没用？）
        encoder.load_state_dict(torch.load(enc_model), map_location = 'cpu')
        decoder.load_state_dict(torch.load(dec_model), map_location = 'cpu')
    else:
        encoder.load_state_dict(torch.load(enc_model))
        decoder.load_state_dict(torch.load(dec_model))
    predict.evaluateTestset(encoder, decoder) 
