# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import data_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
word2id, id2word, emb, word2count, vocab_size = data_utils.Word2vec()

######################################################################
# The Seq2Seq Model
# =================
######################################################################
# The Encoder
# -----------

class EncoderRNN_batch(nn.Module):
    def __init__(self, embed_size, hidden_size): # input size 没用 就是embed size
        super(EncoderRNN_batch, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        pretrained_weight = np.array(emb)  # 已有词向量的numpy
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.embedding.weight.requires_grad = False
        
        self.gru = nn.GRU(embed_size, hidden_size)

    def forward(self, input, input_lengths, hidden):
        embedded = self.embedding(input)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True) 
        output, hidden = self.gru(packed, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


######################################################################
# Attention Decoder
# ^^^^^^^^^^^^^^^^^

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=data_utils.INPUT_MAX_LENGTH,
                 embed_size=200):  # 
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        # embedding
        self.embedding = nn.Embedding(vocab_size, embed_size)
        pretrained_weight = np.array(emb)  # 已有词向量的numpy
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.embedding.weight.requires_grad = False

        # attn gru
        self.attn = nn.Linear(self.hidden_size + embed_size, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size + embed_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[:, 0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs)  # unsqueeze插入一个维度

        output = torch.cat((embedded[:, 0], attn_applied[:, 0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# class EncoderRNN(nn.Module):
#     def __init__(self, input_size, hidden_size): # input size 没用 就是embed size
#         super(EncoderRNN, self).__init__()
#         self.hidden_size = hidden_size
# 
#         # self.embedding = nn.Embedding(input_size, hidden_size)
#         self.gru = nn.GRU(input_size, hidden_size)
#         # self.gru = nn.GRU(hidden_size, hidden_size)
# 
#     def forward(self, input, hidden):
#         embedded = emb[int(input[0])]
#         embedded = torch.tensor(embedded, dtype=torch.float, device=device).view(1, 1, -1)
#         # embedded = self.embedding(input).view(1, 1, -1)
#         output = embedded
#         output, hidden = self.gru(output, hidden)
#         return output, hidden
# 
#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)
    
######################################################################
# The Decoder
# -----------

######################################################################
# Simple Decoder
# ^^^^^^^^^^^^^^
# The initial input token is the start-of-string ``<SOS>``
# token, and the first hidden state is the context vector (the encoder's
# last hidden state).
#

# class DecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size):
#         super(DecoderRNN, self).__init__()
#         self.hidden_size = hidden_size
# 
#         # self.embedding = nn.Embedding(output_size, hidden_size)
#         self.gru = nn.GRU(hidden_size, hidden_size)
#         self.out = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=1)
# 
#     def forward(self, input, hidden):
#         embedded = emb[input.item()]
#         embedded = torch.tensor(embedded, dtype=torch.float, device=device).view(1, 1, -1)
#         output = embedded
#         # output = self.embedding(input).view(1, 1, -1)
#         output = F.relu(output)
#         output, hidden = self.gru(output, hidden)
#         output = self.softmax(self.out(output[0]))
#         return output, hidden
# 
#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)


