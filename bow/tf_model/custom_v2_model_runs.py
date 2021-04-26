import tensorflow as tf
from data.schemas import QuoraDataConfig, QuoraObservation,  QuoraObservationSet
from data.tf_v2_dataset import QuoraTFDataset
from pprint import pprint
from embeddings.embedding_utilities import ParagramEmbeddings, EmbeddingConfig
from bow.tf_model.bow_custom_v2_model import (BOWModelConfig, EmbeddingLookup,
                                              HiddenActivation, HiddenLayer, OutputLayer,
                                              LossLayer, EvalLayer, BOWModel)


embd_config = EmbeddingConfig(pad_to_length=True, max_sequence_length=100, hidden_dimension=50)
embeddings_extractor = ParagramEmbeddings(embd_config)

dataroot_dir = "/Users/ericpenner/Desktop/projects/quora/data/"
datafile_name = "partial_long.json"
raw_data = QuoraObservationSet.from_data_json(datafile_name, dataroot_dir)
converted_data = embeddings_extractor.get_embedding_ids(raw_data)
data_carrier = QuoraTFDataset(converted_data)
train_ds, valid_ds = data_carrier.get_tensor_datasets()
model_config = BOWModelConfig(n_hidden=[100, 100])
model = BOWModel(model_config=model_config, embedding_list=embeddings_extractor.embedding_list)
model.compile(model.optimizer, run_eagerly=True)
model.fit(train_ds, epochs=10)
