import os
import csv
from typing import List, Optional
from dataclasses_json import dataclass_json, config
from dataclasses import dataclass, field


@dataclass_json
@dataclass
class QuoraDataConfig:
    name: str
    description: str
    pos_csv_file: str = field(metadata=config(field_name="posDataFile"), default=None)
    neg_csv_file: str = field(metadata=config(field_name="negDataFile"), default=None)
    root_dir: Optional[str] = field(metadata=config(field_name="rootDir"),
                                    default='/Users/upenner/Desktop/resources/quora_project/data/')
    data_slice: Optional[List[int]] = field(metadata=config(field_name="dataSlice"), default=None)


@dataclass_json
@dataclass
class QuoraObservation:
    uid: str = None
    label: int = None
    rawinput: List[str] = field(default_factory=list)
    tokenized_input: Optional[List[str]] = field(default_factory=list)
    token_ids: Optional[List[int]] = field(default_factory=list)
    token_types: Optional[List[int]] = field(default_factory=list)
    attention_mask: Optional[List[int]] = field(default_factory=list)

    @property
    def feature_dict(self):
        """
        """
        return {
            "token_ids": self.token_ids,
            "token_types": self.token_types,
            "attention_mask": self.attention_mask,
            "label": self.label
        }


@dataclass_json
@dataclass
class QuoraObservationSet:
    name: str
    description: str
    observations: List[QuoraObservation] = field(default_factory=list)

    @classmethod
    def from_config_json(cls, config_json_file: str, root_dir: str = None):
        """
        """
        def add_data(data_path: str, data_config: QuoraDataConfig):
            """
            """
            observations = []
            with open(data_path, newline='') as f:
                csv_reader = csv.reader(f, quotechar='|')
                for idx, row in enumerate(csv_reader):
                    if idx > data_config.data_slice[0] and idx <= data_config.data_slice[1]:
                        uid, input_raw, label = row
                        if str(label) == "0":
                            label_flt = [float(1), float(0)]
                        elif str(label) == "1":
                            label_flt = [float(0), float(1)]
                        observations.append(QuoraObservation(uid, label_flt, input_raw))
                    elif idx > data_config.data_slice[1]:
                        return observations

        data_set = []
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
        data_set += add_data(NEG_DATA_FILE, data_config)
        data_set += add_data(POS_DATA_FILE, data_config)

        return cls(name=data_config.name, description=data_config.description, observations=data_set)

    @classmethod
    def from_data_json(cls, data_json_file: str, root_dir: str = None):
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
            data_set = cls.from_json(text)

        return cls(data_set)

    def feature_dict_generator(self):
        """
        """
        for obs in self.observations:
            yield obs.feature_dict


@dataclass_json
@dataclass
class ModelState:
    epoch: int = 0
    train_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    test_loss: float = -1.0
    test_acc: float = -1.0
