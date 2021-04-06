from typing import List, Optional
from dataclasses_json import dataclass_json, Undefined, LetterCase, config
from dataclasses import dataclass, field


@dataclass_json
@dataclass
class QuoraDataConfig:
    name: str
    description: str
    pos_csv_file: str = field(metadata=config(field_name="posDataFile"),
                              default=None)
    neg_csv_file: str = field(metadata=config(field_name="negDataFile"),
                              default=None)
    root_dir: Optional[str] = field(metadata=config(field_name="rootDir"),
                                    default='/Users/upenner/Desktop/resources/quora_project/data/')
    data_slice: Optional[List[int]] = field(metadata=config(field_name="dataSlice"),
                                            default=None)


@dataclass_json
@dataclass
class QuoraObservation:
    uid: str = None
    label: int = None
    raw_input: List[str] = None
    tokenized_input: Optional[List[str]] = None
    token_ids: Optional[List[int]] = None
    token_types: Optional[List[int]] = None
    attention_mask: Optional[List[int]] = None


@dataclass_json
@dataclass
class QuoraObservationSet:
    name: str
    description: str
    observations: List[QuoraObservation] = field(default_factory=list)


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

