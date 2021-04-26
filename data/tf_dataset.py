import tensorflow.compat.v1 as tf
import numpy as np
from random import shuffle
from typing import List
from data.schemas import QuoraObservationSet


class QuoraTFDataset():

    def __init__(self, data_set: QuoraObservationSet,
                 batch_size: int = 10,
                 valid_split: float = 0.1):

        self.data = data_set
        self.batch_size = batch_size
        self.valid_split = valid_split
        self.train_valid_split()

    def train_valid_split(self):
        """
        """
        shuffle(self.data.observations)
        if (self.valid_split > 0 and self.valid_split < 1):
            valid_size = round(len(self.data.observations)*self.valid_split)
            self.train_data = QuoraObservationSet(self.data.name, self.data.description,
                                                  self.data.observations[valid_size:])
            self.valid_data = QuoraObservationSet(self.data.name, self.data.description,
                                                  self.data.observations[:valid_size])
        else:
            self.train_data, self.valid_data = self.data, None

    def get_tensor_datasets(self) -> tf.data.Dataset:
        """
        """
        train_ds = tf.data.Dataset.from_generator(
            self.train_data.feature_dict_generator,
            output_types={
                "token_ids": tf.int64,
                "token_types": tf.int64,
                "attention_mask": tf.int64,
                "label": tf.int64
            }).batch(self.batch_size)

        if self.valid_data is not None:
            valid_ds = tf.data.Dataset.from_generator(
                self.valid_data.feature_dict_generator,
                output_types={
                    "token_ids": tf.int64,
                    "token_types": tf.int64,
                    "attention_mask": tf.int64,
                    "label": tf.int64
                }).batch(len(self.valid_data.observations))
        else:
            self.valid_ds = None

        return train_ds, valid_ds

    def get_list_datasets(self) -> List[List[int]]:
        """
        """
        def train_ds():
            for idx in range(self.batch_size):
                yield {
                    'token_ids': [obs.token_ids for obs in self.train_data.observations[idx::self.batch_size]],
                    'label': [obs.label for obs in self.train_data.observations[idx::self.batch_size]]
                }

        valid_ds = {'token_ids': [obs.token_ids for obs in self.train_data.observations],
                    'label': [int(obs.label) for obs in self.train_data.observations]}

        return train_ds(), valid_ds
