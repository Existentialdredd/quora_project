import torch
import random
import numpy as np
import csv
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn import functional
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from data.pytorch_dataset import QuoraBertDataSet, bert_tensor_colate
from pprint import pformat

train_data = QuoraBertDataSet('train_neg_50k.csv', 'train_pos_50k.csv', data_slice=[0, 10])
valid_data = QuoraBertDataSet('train_neg_50k.csv', 'train_pos_50k.csv', data_slice=[11, 21])
train_dataloader = DataLoader(train_data, batch_size=4 ,collate_fn=bert_tensor_colate)
valid_dataloader = DataLoader(valid_data, batch_size=len(valid_data), collate_fn=bert_tensor_colate)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True)

optimizer = AdamW(model.parameters(), lr=1e-5)

N_epochs = 2

train_state = {
     'epoch_index':  0,
     'train_loss':  [],
     'train_acc':  [],
     'val_loss': [],
     'val_acc':  [],
     'test_loss': -1,
     'test_acc': -1
     }

for epoch in range(N_epochs):
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        model_output = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        loss = functional.binary_cross_entropy_with_logits(model_output.logits, batch['label'])
        loss.backward()
        optimizer.step()
        train_accuracy = torch.eq(torch.argmax(model_output.logits, dim=1),
                                  torch.argmax(batch['label'], dim=1)).float().mean()
        train_loss = loss.data.item()
        train_state['train_loss'].append(train_loss)
        train_state['train_acc'].append(train_accuracy.item())

    model.eval()
    for valid_batch in valid_dataloader:
        model_output = model(input_ids=valid_batch['input_ids'], attention_mask=valid_batch['attention_mask'])
        loss = functional.binary_cross_entropy_with_logits(model_output.logits, valid_batch['label'])
        valid_accuracy = torch.eq(torch.argmax(model_output.logits, dim=1),
                                  torch.argmax(valid_batch['label'], dim=1)).float().mean()
        valid_loss = loss.data.item()

        train_state['val_loss'].append(valid_loss)
        train_state['val_acc'].append(valid_accuracy.item())

    print(f"Epoch: {epoch}")
    print(f"Train Loss: {train_state['train_loss'][-1]}")
    print(f"Train Acc: {train_state['train_acc'][-1]}")
    print(f"Valid Loss: {train_state['val_loss'][-1]}")
    print(f"Valid Acc: {train_state['val_acc'][-1]}")

print(f" {pformat(train_state)}")
