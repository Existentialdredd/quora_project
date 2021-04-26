from data.schemas import QuoraDataConfig, QuoraObservation,  QuoraObservationSet
from data.tf_dataset import QuoraTFDataset
from embeddings.embedding_utilities import ParagramEmbeddings, EmbeddingConfig
from bow.tf_model.bow_estimator_v1_model import BOWModelConfig, BOWModel, BOWModelRunner
import tensorflow.compat.v1 as tf

tf.logging.set_verbosity(tf.logging.INFO)

embd_config = EmbeddingConfig(pad_to_length=True, max_sequence_length=100, hidden_dimension=50)
embeddings_extractor = ParagramEmbeddings(embd_config)

dataroot_dir = "/Users/ericpenner/Desktop/projects/quora/data/"
datafile_name = "partial_small.json"
raw_data = QuoraObservationSet.from_data_json(datafile_name, dataroot_dir)
converted_data = embeddings_extractor.get_embedding_ids(raw_data)
data_carrier = QuoraTFDataset(converted_data)

model_config = BOWModelConfig(n_hidden=[100, 100])
model = BOWModel(model_config=model_config,
                 embedding_matrix=embeddings_extractor.embedding_list,
                 embedding_config=embd_config)

run_config = tf.estimator.RunConfig(model_dir='/Users/ericpenner/Desktop/projects/quora/bow/tf_model/runs/test3',
                                    save_summary_steps=2, save_checkpoints_secs=20)
params = {'num_epochs': 5, 'learning_rate': 0.001}

model_runner = BOWModelRunner(dataset=data_carrier, run_config=run_config, model=model, params=params)

model_runner.train_model()
