import torch
from transformers import BertTokenizer
from data.pytorch_dataset import QuoraCommentsDataset


class QuoraBertDataSet(QuoraCommentsDataset):

    def __init__(self, data_json: str = None, config_json: str = None):
        super().__init__(data_json, config_json)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        for observation in self.data.observations:
            tokenizer_output = self.tokenizer(observation.raw_input,
                                              padding='max_length',
                                              truncation=True,
                                              return_tensors='pt',
                                              max_length=100)
            observation.attention_mask = tokenizer_output['attention_mask']
            observation.token_ids = tokenizer_output['input_ids']
            observation.token_types = tokenizer_output['token_type_ids']


def bert_tensor_colate(output_dict_list):
    output = dict()
    output['token_ids'] = torch.squeeze(torch.stack([out['token_ids'] for out in output_dict_list]),1)
    output['token_type_ids'] = torch.squeeze(torch.stack([out['token_type_ids'] for out in output_dict_list]),1)
    output['attention_mask'] = torch.squeeze(torch.stack([out['attention_mask'] for out in output_dict_list]),1)
    output['label'] = torch.stack([out['label'] for out in output_dict_list])

    return output
