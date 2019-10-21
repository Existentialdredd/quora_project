import os
import sys
from importlib import reload

import numpy as np
import pandas as pd
from random import shuffle

sys.path.append('../embeddings')
sys.path.append('../eda')
import svd_embeddings as svd_em
import embedding_utilities as embd_util
import utilities as ut
import neural_cnn_classifier as ncnn

DATA_PATH = '~/google_drive/data/quora/'
DATA_FILE = '{}{}'.format(DATA_PATH,'train.csv')
data = pd.read_csv(DATA_FILE)

data_pos_tokenized = [[vec[0],ut.canon_token_sentence(vec[1]),vec[2]]
                      for vec in data.to_numpy() if vec[2] == 1]
data_neg_tokenized = [[vec[0],ut.canon_token_sentence(vec[1]),vec[2]]
                      for vec in data.to_numpy() if vec[2] == 0]

NVALID = 5000
NTRAIN = 10000
MAX_SEQ_LEN = 100
EMBD_DIM = 300

train_data = data_pos_tokenized[NVALID:NTRAIN+NVALID] + data_neg_tokenized[NVALID:NTRAIN+NVALID]
shuffle(train_data)
valid_data = data_pos_tokenized[:NVALID] + data_neg_tokenized[:NVALID]
shuffle(valid_data)

train_comments = [tpl[1] for tpl in train_data]
valid_comments = [tpl[1] for tpl in valid_data]
labels_train = [val[2] for val in train_data]
labels_valid = [val[2] for val in valid_data]

#svd_embd = svd_em.SVD_Embeddings(train_comments,max_sequence_length=100,tokenized=True)
#svd_embd.SVD_train()
#sequences_train = svd_embd.train_sequences_w_pad
#sequences_valid = svd_embd.sequences_to_ids_w_pad([val[1] for val in valid_data],tokenized=True)
#embeddings = svd_embd.trained_embeddings

filename = '../embeddings/quora_full_embeds.txt'
embeddings, token_to_row_number = embd_util.embedding_extraction(filename,embedding_length=EMBD_DIM)
embeddings.append([0]*EMBD_DIM)
train_id_sequences = embd_util.token_sequence_to_id_sequence(train_comments,
                                                             token_to_row_number,
                                                             unknown_token='UNK',
                                                             max_sequence_length=100)
valid_id_sequences = embd_util.token_sequence_to_id_sequence(valid_comments,
                                                             token_to_row_number,
                                                             unknown_token='UNK',
                                                             max_sequence_length=100)

ncnn_classifier = ncnn.Neural_CNN({'n_tokens':len(embeddings)-1,
                                   'embedding_dimension':EMBD_DIM,
                                   'max_sequence_length':100})
ncnn_classifier.build_graph()

sequences_train = train_id_sequences
sequences_valid = valid_id_sequences

train_dict = {'embeddings':embeddings,
              'sequences_train':sequences_train,'labels_train': labels_train,
              'sequences_valid':sequences_valid,'labels_valid': labels_valid,'batch_size':200}

ncnn_classifier.train_graph(train_dict)
ncnn_classifier.predict_and_report(sequences_valid,labels_valid,embeddings)