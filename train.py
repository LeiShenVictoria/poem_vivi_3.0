# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import torch.utils.data as Data
from torch.utils.data.dataset import Dataset

import data_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################################################
# Training the Model
# ------------------
#
# Then the decoder is given the ``<SOS>`` token as its first input, 
# and the last hidden state of the encoder as its first hidden state.
#
teacher_forcing_ratio = 0.5

def train_batch(batch_size, input_batch, input_lengths, target_batch, target_lengths, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion): # 
    encoder_hidden = encoder.initHidden(batch_size)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0

    encoder_outputs, encoder_hidden = encoder(input_batch, input_lengths, encoder_hidden)

    # 将encoder_outputs padding至INPUT_MAX_LENGTH 因为attention中已经固定此维度大小为INPUT_MAX_LENGTH
    encoder_outputs_padded = torch.zeros(batch_size, data_utils.INPUT_MAX_LENGTH, encoder.hidden_size, device=device)
    for b in range(batch_size):
        for ei in range(input_lengths[b]):
            encoder_outputs_padded[b, ei] = encoder_outputs[b, ei]
    
    decoder_input = torch.tensor([[data_utils.SOS_token]* batch_size], device=device).transpose(0, 1) #
    
    decoder_hidden = encoder_hidden # Use last hidden state from encoder to start decoder

    target_max_length = max(target_lengths)

    # use_teacher_forcing = False #
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs_padded)
            loss += criterion(decoder_output, target_batch[:, di])
            decoder_input = target_batch[:, di].unsqueeze(1)  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs_padded)
            target = target_batch[:, di] # size正确
            loss += criterion(decoder_output, target) # input: batch*class, target: batch （结果取了batch平均）
            topv, topi = decoder_output.topk(1) # value 和 id
            decoder_input = topi.detach()  # detach from history as input
            # if decoder_input.item() == data_utils.EOS_token: # 
            #     break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / int(sum(target_lengths))

######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

######################################################################
# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#

def sort_batch_data(batch_x, x_len, batch_y):
    sorted_length, sorted_id = x_len.sort(dim=0, descending=True)
    sorted_x = batch_x[sorted_id]
    sorted_y = batch_y[sorted_id]
    return sorted_x, sorted_length, sorted_y

def trainIters_batch(training_set, encoder, decoder, learning_rate, batch_size, epochs):
    start = time.time() 
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    # 优化器排除embedding层梯度（embedding层是固定的）
    encoder_optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, encoder.parameters()), lr=learning_rate, rho=0.95, eps=1e-06, weight_decay=0) #
    decoder_optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, decoder.parameters()), lr=learning_rate, rho=0.95, eps=1e-06, weight_decay=0) #
    # encoder_optimizer = optim.SGD(filter(lambda p: p.requires_grad, encoder.parameters()), lr=learning_rate) 
    # decoder_optimizer = optim.SGD(filter(lambda p: p.requires_grad, decoder.parameters()), lr=learning_rate)
    # encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    # decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    pairs, lines = data_utils.read_train_data(training_set)

    criterion = nn.CrossEntropyLoss() # 对batch取平均
    # criterion = nn.NLLLoss()

    # --------------batch--------------
    BATCH_SIZE = batch_size  # 批训练的数据个数，大数量中每次抽取五个

    dataset = data_utils.PoetryRNNData(pairs)
    # torch_dataset = Data.Dataset(data_tensor=input_set, target_tensor=output_set)

    # loader使训练变成小批。 把 dataset 放入 DataLoader+66688
    loader = Data.DataLoader(
        dataset=dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        # num_workers=2,  # 多线程来读数据，提取xy的时候几个数据一起提取
    )

    steps = lines // batch_size
    print_every = steps // 5 # 每个epoch打印5个loss
    plot_every = print_every
    
    for epoch in range(epochs):  # 把整个数据进行训练的次数，通过loader来确定是否打乱数据，打乱比较好
        print('epoch: %d' % epoch)
        for step, (batch_x, x_len, batch_y, y_len) in enumerate(loader):  
            # 每一步 loader 释放一小批数据用来学习，step=总数据量/batch_size，enumerate把每次提取编写索引。
            # batch_x: B*T tensor
            # x_len: list of actual length for each sentence in a batch

            # 打出来一些数据
            # print('Epoch: ', epoch, '| Step: ', step, '\n batch x: ',
            #       batch_x, '\n', x_len, '\n batch y: ', batch_y, '\n', y_len)

            batch_x, x_len, batch_y = sort_batch_data(batch_x, x_len, batch_y)
            loss = train_batch(BATCH_SIZE, batch_x, x_len, batch_y, y_len, encoder, decoder, 
                         encoder_optimizer, decoder_optimizer, criterion)
            
            print_loss_total += loss
            plot_loss_total += loss
     
            if step % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, (step+1) / steps), 
                                             step+1, (step+1) / steps * 100, print_loss_avg))
                # print('%s (%d %d%%) %.4f' % (timeSince(start, step / n_iters),
                                             # step, step / n_iters * 100, print_loss_avg))

            if step % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        print('save model')  # save for every epoch
        torch.save(encoder.state_dict(), 'model/enc_epoch=' + str(epoch) + '.pkl')
        torch.save(decoder.state_dict(), 'model/dec_epoch=' + str(epoch) + '.pkl')
    showPlot(plot_losses)
    # ---------------------------------

######################################################################
# Plotting results
# ----------------
#
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()
