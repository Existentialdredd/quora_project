import torch
from torch.utils.data import DataLoader
import logging
from pprint import pformat
from bert_model_runs import train_model

train_model(train_config_json= 'bert_train_config_001.json',
            valid_config_json= 'bert_valid_config_001.json',
            n_epochs=5)

# from transformers import BertTokenizer, BertModel, BertForMaskedLM
# from data.pytorch_dataset import QuoraCommentsDataset, QuoraBertDataSet, bert_tensor_colate

# logging.basicConfig(level=logging.INFO)

# train_data = QuoraCommentsDataset('train_neg_50k.csv', 'train_pos_50k.csv', data_slice=[1, 100])

# train_data = QuoraBertDataSet('train_neg_50k.csv', 'train_pos_50k.csv', data_slice=[0, ])

# train_dataloader = DataLoader(train_data, batch_size=3, collate_fn=bert_tensor_colate)

# for batch in train_dataloader:
    # print(f"\n {pformat(batch)}")
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# text = "[CLS] who was Jim Henson? [SEP] Jim Henson was a puppeteer [SEP]"
# tokenized_text = tokenizer.tokenize(text)

# print(f"Text: \n  {tokenized_text}")

# masked_index = 8
# tokenized_text[masked_index] = '[MASK]'
# print(f"Masked Text: \n {tokenized_text}")

# index_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# tokens_tensor = torch.tensor([index_tokens])
# segments_tensors = torch.tensor([segments_ids])

# print(f"Token Indexes: \n {tokens_tensor}")
# print(f"Segments: \n {segments_tensors}")

# # ----------
# model = BertModel.from_pretrained('bert-base-uncased')
# model.eval()

# with torch.no_grad():
    # outputs = model(tokens_tensor, token_type_ids=segments_tensors)

# model = BertForMaskedLM.from_pretrained('bert-base-uncased')
# model.eval()

# with torch.no_grad():
    # outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    # predictions = outputs[0]

# predicted_index = torch.argmax(predictions[0, masked_index]).item()
# predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

# print(f"Predicted Index: \n {predicted_index}")
# print(f"Predicted Token: \n {predicted_token}")
