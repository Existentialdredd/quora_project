import os
import sys
from importlib import reload

import numpy as np
import pandas as pd
from random import shuffle

sys.path.append('../embeddings')
sys.path.append('../eda')
import svd_embeddings as svd_em
import utilities as ut
import neural_bow_classifier as nbow

DATA_PATH = '~/google_drive/data/quora/'
DATA_FILE = '{}{}'.format(DATA_PATH,'train.csv')
data_full = pd.read_csv(DATA_FILE)
data = data_full
data_pos_tokenized = [[vec[0],ut.canon_token_sentence(vec[1]),vec[2]]
                      for vec in data.to_numpy() if vec[2] == 1]
data_neg_tokenized = [[vec[0],ut.canon_token_sentence(vec[1]),vec[2]]
                      for vec in data.to_numpy() if vec[2] == 0]

NVALID = 1000
NTRAIN = 100000

train_data = data_pos_tokenized[NVALID:NTRAIN+NVALID] + data_neg_tokenized[NVALID:NTRAIN+NVALID]
shuffle(train_data)
valid_data = data_pos_tokenized[:NVALID] + data_neg_tokenized[:NVALID]
shuffle(valid_data)

train_comments = [tpl[1] for tpl in train_data]
svd_embd = svd_em.SVD_Embeddings(train_comments,max_sequence_length=100,tokenized=True)

svd_embd.SVD_train()
nbow_classifier = nbow.Neural_BOW({'n_tokens':svd_embd.vocab_size,'max_sequence_length':100})
nbow_classifier.build_graph()

sequences_train = svd_embd.train_sequences_w_pad
labels_train = [val[2] for val in train_data]
sequences_valid = svd_embd.sequences_to_ids_w_pad([val[1] for val in valid_data],tokenized=True)
labels_valid = [val[2] for val in valid_data]
train_dict = {'embeddings':svd_embd.trained_embeddings,
              'sequences_train':svd_embd.train_sequences_w_pad,'labels_train': labels_train,
              'sequences_valid':sequences_valid,'labels_valid': labels_valid}

nbow_classifier.train_graph(train_dict)
nbow_classifier.predict_and_report(sequences_valid,labels_valid,svd_embd.trained_embeddings)
