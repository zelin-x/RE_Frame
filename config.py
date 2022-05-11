# -*- coding: utf-8 -*-

class DefaultConfig(object):
    root_path = r''
    rel_path = r'relation.txt'
    bert_vocab_path = r'./bert_chinese/vocab.txt'
    result_dir = './out'
    load_model_path = r''
    bert_path = r'./bert_chinese'

    batch_size = 1  # batch size
    num_workers = 0  # how many workers for loading data

    vocab_size = 21128  # vocab + UNK + BLANK
    rel_num = 0

    limit = 50  # the position range <-limit, limit>
    max_len = 150 + 2  # max_len for each sentence + two padding
    entity_max_len = 20
    word_dim = 768
    pos_dim = 50
    pos_size = 202

    lambda_pcnn = 0.05
    lambda_san = 1.0
    drop_out = 0.5

    # Conv
    filters = [3]
    hidden_size = 230
