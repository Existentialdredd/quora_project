import tensorflow.compat.v1 as tf
from data.schemas import QuoraObservationSet


class QuoraTFDataset():

    def __init__(self, data_set: QuoraObservationSet, batch_size: int = 10):
        self.data = data_set
        self.batch_size = batch_size

    def get_dataset(self):
        """
        """
        return tf.data.Dataset.from_generator(self.data.feature_dict_generator,
                                              output_types={
                                                  "token_ids": tf.int64,
                                                  "token_types": tf.int64,
                                                  "attention_mask": tf.int64,
                                                  "label": tf.int64
                                              }).batch(self.batch_size)
