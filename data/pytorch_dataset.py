import torch
import os
from torch.utils.data import Dataset
import csv
from data.schemas import QuoraDataConfig, QuoraObservation, QuoraObservationSet


class QuoraCommentsDataset(Dataset):
    """
    Quora Comments Dataset
    """
    def __init__(self, data_json: str = None, config_json: str = None):
        if data_json is not None:
            self.from_data_json(data_json)
        elif config_json is not None:
            self.from_config_json(config_json)

    def from_config_json(self, config_json_file: str, root_dir: str = None):
        """
        """
        if root_dir is None:
            root_dir = os.path.dirname(os.path.abspath(__file__))
            DATA_DIR = os.path.join(root_dir, 'json_configs')
        else:
            DATA_DIR = root_dir

        DATA_FILE = os.path.join(DATA_DIR, config_json_file)
        with open(DATA_FILE, 'r') as f:
            text = ''.join([text for text in f])
            data_config = QuoraDataConfig.from_json(text)

        NEG_DATA_FILE = os.path.join(data_config.root_dir, data_config.pos_csv_file)
        POS_DATA_FILE = os.path.join(data_config.root_dir, data_config.neg_csv_file)

        self.data = QuoraObservationSet(data_config.name, data_config.description)
        with open(NEG_DATA_FILE, newline='') as f:
            csv_reader = csv.reader(f, quotechar='|')
            for idx, row in enumerate(csv_reader):
                if idx < data_config.data_slice[0]:
                    pass
                elif idx >= data_config.data_slice[1]:
                    break
                else:
                    uid, input_raw, label = row
                    if label == "0":
                        label_flt = [float(1), float(0)]
                    elif label == "1":
                        label_flt = [float(0), float(1)]
                    self.data.observations.append(QuoraObservation(uid, label_flt, input_raw))

        with open(POS_DATA_FILE, newline='') as f:
            csv_reader = csv.reader(f, quotechar='|')
            for idx, row in enumerate(csv_reader):
                if idx < data_config.data_slice[0]:
                    pass
                elif idx > data_config.data_slice[1]:
                    break
                else:
                    uid, input_raw, label = row
                    if label == "0":
                        label_flt = [float(1), float(0)]
                    elif label == "1":
                        label_flt = [float(0), float(1)]
                    self.data.observations.append(QuoraObservation(uid, label_flt, input_raw))

    def from_data_json(self, data_json_file: str, root_dir: str = None):
        """
        PURPOSE: Load data from a json file.
        """
        if root_dir is None:
            root_dir = os.path.dirname(os.path.abspath(__file__))
            DATA_DIR = os.path.join(root_dir, 'json_data')
        else:
            DATA_DIR = root_dir

        DATA_FILE = os.path.join(DATA_DIR, data_json_file)
        with open(DATA_FILE, 'r') as f:
            text = ''.join([text for text in f])
            self.data = QuoraObservationSet.from_json(text)

    def __len__(self):
        return len(self.data.observations)

    def __getitem__(self, idx):
        """
        """
        output = dict()
        output['token_ids'] = self.data.observations[idx].token_ids
        output['token_type_ids'] = self.data.observations[idx].token_types
        output['attention_mask'] = self.data.observations[idx].attention_mask
        output['label'] = torch.tensor(self.data.observations[idx].label)

        return output


