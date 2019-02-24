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

import data_utils
import constrains

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
word2id, id2word, emb, word2count, vocab_size = data_utils.Word2vec()

######################################################################
# Evaluation
# ==========

def evaluate(encoder, decoder, sentence): #
    with torch.no_grad(): # for inference, no bp, save memory
        encoder_hidden = encoder.initHidden(1)

        input_length = sentence.size(1)
        encoder_outputs, encoder_hidden = encoder(sentence, [input_length], encoder_hidden)

        # 将encoder_outputs padding至INPUT_MAX_LENGTH 因为attention中已经固定此维度大小为INPUT_MAX_LENGTH
        encoder_outputs_padded = torch.zeros(1, data_utils.INPUT_MAX_LENGTH, encoder.hidden_size,
                                             device=device)
        for b in range(1):
            for ei in range(input_length):
                encoder_outputs_padded[b, ei] = encoder_outputs[b, ei]

        decoder_input = torch.tensor([[data_utils.SOS_token]], device=device)  #
        decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder

        target_max_length = 28 # 暂定
        decoded_words = []
        
        for di in range(target_max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs_padded)
            topi, word = constrains.get_next_word(decoder_output.data, decoded_words)
            if word == 'N':
                print('cannot meet requirements')
                break
            decoder_input = topi.reshape((1, 1)).detach()  # detach from history as input
            decoded_words.append(word)
            # if (di + 1) % 7 == 0:
            #     decoded_words.append('/')
            # if decoder_input.item() == data_utils.EOS_token: # 
            #     break
            
        return decoded_words
        
        
        # 
        # input_tensor = sentence # 
        # # input_tensor = data_utils.tensorFromSentence(input_lang, sentence)  # 
        # input_length = input_tensor.size()[0]
        # encoder_hidden = encoder.initHidden()
        # 
        # encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        # 
        # for ei in range(input_length):
        #     encoder_output, encoder_hidden = encoder(input_tensor[ei],
        #                                              encoder_hidden)
        #     encoder_outputs[ei] += encoder_output[0, 0]
        # 
        # decoder_input = torch.tensor([[data_utils.SOS_token]], device=device)  # SOS
        # 
        # decoder_hidden = encoder_hidden
        # 
        # decoded_words = []
        # decoder_attentions = torch.zeros(max_length, max_length)
        # decoder_outputs = torch.zeros(max_length, vocab_size) #
        # 
        # for di in range(28):
        #     decoder_output, decoder_hidden, decoder_attention = decoder(
        #         decoder_input, decoder_hidden, encoder_outputs)
        #     decoder_attentions[di] = decoder_attention.data
        #     #
        #     
        #     # decoder_outputs[di] = decoder_output.data #
        #     print(decoder_output.data.size()) #
        #     topi, word = constrains.get_next_word(decoder_output.data, decoded_words)
        #     decoded_words.append(word)
        #     #
        #     # topv, topi = decoder_output.data.topk(1) # topk returns a tuple(values, indices)
        #     # if topi.item() == data_utils.EOS_token:
        #     #     decoded_words.append('<EOS>')
        #     #     break
        #     # else:
        #     #     decoded_words.append(id2word[str(topi.item())])
        # 
        #     decoder_input = topi.squeeze().detach()
        # 
        # print(decoder_attentions.size()) #
        # 
        # return decoded_words, decoder_attentions[:di + 1]


######################################################################
# We can evaluate random sentences from the training set and print out the
# input, target, and output to make some subjective quality judgements:
#

def evaluateTestset(encoder, decoder):
    input_set, lines = data_utils.read_test_data()
    for i in range(len(input_set)):
        input = input_set[i]
        line = lines[i]
        output_words = evaluate(encoder, decoder, input)
        # output_words, attentions = evaluate(encoder, decoder, input)
        output_words.insert(7, '/')
        output_words.insert(15, '/')
        output_words.insert(23, '/')
        output_sentence = ' '.join(output_words)
        print(line, ' ==== ', output_sentence)
        # print('>', line)
        # print('<', output_sentence)