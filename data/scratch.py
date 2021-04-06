import os
from pytorch_dataset import QuoraCommentsDataset, QuoraBertDataSet
from schemas import QuoraDataConfig, QuoraObservation,  QuoraObservationSet

root_dir = os.path.dirname(os.path.abspath(__file__))
json_file = os.path.join(root_dir, 'json_configs', 'test_config.json')

with open(json_file, 'r') as f:
    text = ''.join([text for text in f])
    data_config = QuoraDataConfig.from_json(text)

obs_set = QuoraObservationSet(data_config.name, data_config.description)

config = QuoraDataConfig('train_pos.csv', 'train_neg.csv')
# base_ds = QuoraCommentsDataset.from_config_json('test_config.json')
bert_ds = QuoraBertDataSet(config_json='test_config.json')
