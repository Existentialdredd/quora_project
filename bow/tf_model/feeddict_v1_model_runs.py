from data.schemas import QuoraDataConfig, QuoraObservation,  QuoraObservationSet
from data.tf_dataset import QuoraTFDataset
from embeddings.embedding_utilities import ParagramEmbeddings, EmbeddingConfig
from bow.tf_model.bow_feeddict_v1_model import BOWModelConfig, BOWModel
import tensorflow.compat.v1 as tf

tf.logging.set_verbosity(tf.logging.INFO)

embd_config = EmbeddingConfig(pad_to_length=True, max_sequence_length=100, hidden_dimension=50)
embeddings_extractor = ParagramEmbeddings(embd_config)

dataroot_dir = "/Users/ericpenner/Desktop/projects/quora/data/"
datafile_name = "partial_long.json"
raw_data = QuoraObservationSet.from_data_json(datafile_name, dataroot_dir)
converted_data = embeddings_extractor.get_embedding_ids(raw_data)
data_carrier = QuoraTFDataset(converted_data)

model_config = BOWModelConfig(n_hidden=[100, 100])
model = BOWModel(dataset=data_carrier,
                 embedding_list=embeddings_extractor.embedding_list,
                 model_config=model_config)
model.build_graph()
model.train()
model.evaluate()
