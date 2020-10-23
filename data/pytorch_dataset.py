import torch
import os
from torch.utils.data import Dataset
import random
import numpy as np
import csv
from transformers import BertTokenizer, BertForSequenceClassification


class QuoraCommentsDataset(Dataset):
    """
    Quora Comments Dataset
    """

    def __init__(self,
                 pos_csv_file,
                 neg_csv_file,
                 root_dir='~/Desktop/resources/quora_project/data/',
                 data_slice=[0, 100]):

        NEG_DATA_FILE = os.path.join(root_dir, pos_csv_file)
        POS_DATA_FILE = os.path.join(root_dir, neg_csv_file)

        self.combined_data, self.combined_labels = [], []
        with open(NEG_DATA_FILE, newline='') as f:
            csv_reader = csv.reader(f, quotechar='|')
            for idx, row in enumerate(csv_reader):
                if idx < data_slice[0]:
                    pass
                elif idx >= data_slice[1]:
                    break
                else:
                    self.combined_data.append(row[0])
                    self.combined_labels.append([float(1), float(0)])

        with open(POS_DATA_FILE, newline='') as f:
            csv_reader = csv.reader(f, quotechar='|')
            for idx, row in enumerate(csv_reader):
                if idx < data_slice[0]:
                    pass
                elif idx > data_slice[1]:
                    break
                else:
                    self.combined_data.append(row[0])
                    self.combined_labels.append([float(0), float(1)])

        shuffle = list(range(len(self.combined_data)))
        self.features = [self.combined_data[idx] for idx in shuffle]
        self.labels = [self.combined_labels[idx] for idx in shuffle]

    def __len__(self):
        return len(self.features)


class QuoraBertDataSet(QuoraCommentsDataset):

    def __init__(self,
                 pos_csv_file,
                 neg_csv_file,
                 root_dir='/Users/upenner/Desktop/resources/quora_project/data/',
                 data_slice=[0, 100]):
        super(QuoraBertDataSet, self).__init__(pos_csv_file, neg_csv_file, root_dir, data_slice)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenized_features = self.tokenizer(self.features, padding=True, truncation=True, return_tensors='pt')

    def __getitem__(self, idx):
        """
        """
        output = dict()
        output['input_ids'] = self.tokenized_features['input_ids'][idx]
        output['token_type_ids'] = self.tokenized_features['token_type_ids'][idx]
        output['attention_mask'] = self.tokenized_features['attention_mask'][idx]
        output['label'] = torch.tensor(self.labels[idx])

        return output


def bert_tensor_colate(output_dict_list):
    output = dict()
    output['input_ids'] = torch.stack([out['input_ids'] for out in output_dict_list])
    output['token_type_ids'] = torch.stack([out['token_type_ids'] for out in output_dict_list])
    output['attention_mask'] = torch.stack([out['attention_mask'] for out in output_dict_list])
    output['label'] = torch.stack([out['label'] for out in output_dict_list])

    return output


