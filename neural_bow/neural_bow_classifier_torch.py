from __future__ import print_function
from __future__ import division
import torch.nn as nn
import torch
import torch.nn.functional as F


class NeuralBOW(nn.Module):
    """
    PURPOSE: To build, and train a deep neural network corresponding to the
             "neural bag of words" model first developted in "A Neural
             Probabilistic Language Model" by Bengio et al. in the Journal of
             Machine Learning Research.

    ARGS:
    setup                       (dict) containing all some or none of the following
        learning_rate               (float) ml gradient descent learning rate
        max_sequence_length         (int) maximum length of token sequence
        embedding_dimension         (int) truncated length of embedding vectors
        batch_normalization         (bool) indicator for use of batch normalization
        n_hidden                    (list) list of neurons per hidden layer
        hidden_activation           (str) hidden layer activation function
        dropout_rate                (float) probability of dropout
        n_tokens                    (int) total number of rows in embed matrix
        n_outputs                   (int) number of unique labels
        root_log_dir                (str) directory where tf logs will be kept
        checkpoint_dir              (str) directory where checkpoints are kept
    """

    def __init__(self, setup):

        super().__init__()

        self.LRN_RATE = setup.get('learning_rate', 0.001)
        self.MAX_SEQ_LEN = setup.get('max_sequence_length', 100)
        self.input_embeddings = setup.get('input_embedding', None)
        self.EMBD_DIM = setup.get('embedding_dimension', 100)
        self.BATCH_NORM = setup.get('batch_normalization', True)
        self.n_hidden = setup.get('n_hidden', [100, 100, 100])
        self.hidden_activation = setup.get('hidden_activation', 'elu')
        self.DROP_RATE = setup.get('dropout_rate', 0.5)
        self.N_TOKN = setup.get('n_tokens', 100)
        self.N_OUTPUTS = setup.get('n_outputs', 2)

        if self.input_embeddings is None:
            self.embeddings = nn.Embedding(self.N_TOKN+1, self.EMBD_DIM)
        else:
            self.embeddings = nn.Embedding.from_pretrained(torch.tensor(self.input_embeddings))

        self.fc1 = nn.Linear(self.EMBD_DIM, self.n_hidden[0])
        self.bn1 = nn.BatchNorm1d(self.n_hidden[0])
        self.fc1_activation = nn.ReLU()
        self.do1 = nn.Dropout(p=self.DROP_RATE)

        self.fc2 = nn.Linear(self.n_hidden[0], self.n_hidden[1])
        self.bn2 = nn.BatchNorm1d(self.n_hidden[1])
        self.fc2_activation = nn.ReLU()
        self.do2 = nn.Dropout(p=self.DROP_RATE)

        self.fc3= nn.Linear(self.n_hidden[1], self.n_hidden[2])
        self.bn3 = nn.BatchNorm1d(self.n_hidden[2])
        self.fc3_activation = nn.ReLU()
        self.do3 = nn.Dropout(p=self.DROP_RATE)

        self.fc3 = nn.Linear(self.n_hidden[2], self.N_OUTPUTS)


    def forward(self, inputs):
        embeds = self.embeddings(inputs).sum(dim=1)

        raw_out1 = self.fc1(embeds)
        norm_out1 = self.bn1(raw_out1)
        act_out1 = F.relu(norm_out1)
        drop_out1 = self.do1(act_out1)

        raw_out2 = self.fc2(drop_out1)
        norm_out2 = self.bn2(raw_out2)
        act_out2 = F.relu(norm_out2)
        drop_out2 = self.do2(act_out2)

        raw_out3 = self.fc2(drop_out2)
        norm_out3 = self.bn2(raw_out3)
        act_out3 = F.relu(norm_out3)
        drop_out3 = self.do2(act_out3)

        logits = self.fc3(drop_out3)
        return logits
