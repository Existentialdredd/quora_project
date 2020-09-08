import numpy as np
import random
import pandas as pd
from random import shuffle
import embeddings.embedding_utilities as embd_util
import eda.utilities as ut
from neural_bow.neural_bow_classifier_torch import NeuralBOW
import torch
import torch.nn as nn
import torch.optim as optim

tokenize = False
write_to_csv = False

# Data Loading and Tokenization
DATA_PATH = '~/Desktop/resources/quora_project/'
DATA_FILE = '{}{}'.format(DATA_PATH, 'train.csv')

if tokenize:
    data = pd.read_csv(DATA_FILE)
    data_pos_tokenized = [[vec[0], ut.canon_token_sentence(vec[1]), vec[2]]
                          for vec in data.to_numpy() if vec[2] == 1]
    data_neg_tokenized = [[vec[0], ut.canon_token_sentence(vec[1]), vec[2]]
                          for vec in data.to_numpy() if vec[2] == 0]
    if write_to_csv:
        pos_array = np.array(data_pos_tokenized)
        pd.DataFrame(pos_array).to_csv(''.join([DATA_PATH, "data/train_pos_tokenized.csv"]))
        neg_array = np.array(data_neg_tokenized)
        pd.DataFrame(neg_array).to_csv(''.join([DATA_PATH, "data/train_neg_tokenized.csv"]))
else:
    data_pos_tokenized = pd.read_csv(''.join([DATA_PATH, "data/train_pos_tokenized.csv"])).to_numpy().tolist()
    data_neg_tokenized = pd.read_csv(''.join([DATA_PATH, "data/train_neg_tokenized.csv"])).to_numpy().tolist()


# Variables
NVALID = 5000
NTRAIN = 100000
MAX_SEQ_LEN = 100
EMBD_DIM = 100

# Data Extraction and Splittings
train_data = (data_pos_tokenized[NVALID:NTRAIN+NVALID] +
              data_neg_tokenized[NVALID:NTRAIN+NVALID])
shuffle(train_data)
valid_data = data_pos_tokenized[:NVALID] + data_neg_tokenized[:NVALID]
shuffle(valid_data)

train_comments = [tpl[2] for tpl in train_data]
valid_comments = [tpl[2] for tpl in valid_data]
labels_train = [val[3] for val in train_data]
labels_valid = [val[3] for val in valid_data]

embedding_path = '/Users/upenner/Desktop/resources/quora_project/embeddings/quora_full_embeds.txt'
# Word Embeddings Extraction
embeddings, token_to_row_number = embd_util.embedding_extraction(embedding_path,
                                                                 embedding_length=EMBD_DIM)
embeddings.append([0]*EMBD_DIM)
embeddings = np.array(embeddings).astype(np.float32)

# Conversion of Training and Validation Data to Embedding id sequences
train_id_sequences = embd_util.token_sequence_to_id_sequence(train_comments,
                                                             token_to_row_number,
                                                             unknown_token='UNK',
                                                             max_sequence_length=100)
train_ids = torch.tensor(train_id_sequences, dtype=torch.long)
train_labels = torch.tensor(labels_train, dtype=torch.long)
valid_id_sequences = embd_util.token_sequence_to_id_sequence(valid_comments,
                                                             token_to_row_number,
                                                             unknown_token='UNK',
                                                             max_sequence_length=100)
valid_ids = torch.tensor(valid_id_sequences, dtype=torch.long)
valid_labels = torch.tensor(labels_valid, dtype=torch.long)

# Class instantiation and graph building
model = NeuralBOW({'n_tokens': len(embeddings)-1,
                             'max_sequence_length': 100,
                             'embedding_dimension': EMBD_DIM,
                             'input_embedding': embeddings})

def batches(N_elems, N_batches):
    """
    PURPOSE: Generate indexs for minibatchs used in minibatch gradient descent
    """
    elems = list(range(N_elems))
    random.shuffle(elems)
    return torch.tensor([elems[i::N_batches] for i in range(N_batches)]).view(N_batches, -1)


N_epochs  = 20
N_batches = 100

train_state = {
     'epoch_index':  0,
     'train_loss':  [],
     'train_acc':  [],
     'val_loss': [],
     'val_acc':  [],
     'test_loss': -1,
     'test_acc': -1
     }

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(),
                       lr=0.0005)

# if torch.cuda.is_available():
    # device = torch.device("cuda")
# else:
    # device = torch.device("cpu")

# model.to(device)

for epoch in range(N_epochs):
    model.train()
    for batch in batches(NTRAIN, N_batches):
        optimizer.zero_grad()
        train_features_batch = train_ids[batch]
        train_labels_batch = train_labels[batch]
        train_label_pred = model(train_features_batch)
        loss = loss_fn(train_label_pred, train_labels_batch)
        loss.backward()
        optimizer.step()
        train_accuracy = torch.eq(torch.argmax(train_label_pred, dim=1),
                                  train_labels_batch).float().mean()
        train_loss = loss.data.item()
        train_state['train_loss'].append(train_loss)
        train_state['train_acc'].append(train_accuracy)

    model.eval()
    valid_features = valid_ids
    valid_labels_pred = model(valid_features)
    loss = loss_fn(valid_labels_pred, valid_labels)
    valid_accuracy = torch.eq(torch.argmax(valid_labels_pred, dim=1),
                              valid_labels).float().mean()
    valid_loss = loss.data.item()
    train_state['val_loss'].append(valid_loss)
    train_state['val_acc'].append(valid_accuracy)

    print(f"Epoch: {epoch}")
    print(f"Train Loss: {train_state['train_loss'][-1]}")
    print(f"Train Acc: {train_state['train_acc'][-1]}")
    print(f"Valid Loss: {train_state['val_loss'][-1]}")
    print(f"Valid Acc: {train_state['val_acc'][-1]}")
