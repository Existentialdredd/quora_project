from __future__ import print_function
from __future__ import division
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import List
from data.tf_dataset import QuoraTFDataset
from embeddings.embedding_utilities import EmbeddingConfig
import os
import sys
import numpy as np
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)


@dataclass_json
@dataclass
class BOWModelConfig:
    """
    PURPOSE:

    ARGS:
    :max_sequence_length: maximum length of token sequence
    :n_tokens: total number of rows in embed matrix
    :embedding_dimension: truncated length of embedding vectors
    :n_hidden: list of neurons per hidden layer
    :hidden_activation: hidden layer activation function
    :batch_normalization: indicator for use of batch normalization
    :dropout_rate: probability of dropout
    :n_outputs: number of unique labels
    :root_log_dir: directory where tf logs will be kept
    :checkpoint_dir: directory where checkpoints are kept
    """
    max_sequence_length: int = 100
    num_tokens: int = 100
    embedding_dimension: int = 100
    n_hidden: List[int] = field(default_factory=list)
    hidden_activation: str = 'elu'
    batch_normalization: bool = True
    drop_rate: float = 0.5
    num_outputs: int = 2
    model_dir: str = os.path.dirname(os.path.abspath(__file__)) + '/models'

    @classmethod
    def from_json(cls, filename: str, root_dir: str = None):
        """
        """
        if root_dir is None:
            root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
            filepath = os.path.join(root_dir, filename)
        else:
            filepath = os.path.join(root_dir, filename)

        with open(filepath, 'r') as file:
            text = ''.join([line for line in file])

        return cls.from_json(text)

    def write_to_file(self, filename: str, root_dir: str = None):
        """
        """
        if root_dir is None:
            root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
            filepath = os.path.join(root_dir, filename)
        else:
            filepath = os.path.join(root_dir, filename)

        with open(filepath, 'r') as file:
            file.write(self.__dict__)


class BOWModel(object):
    """
    PURPOSE: To build, and train a deep neural network corresponding to the
             "neural bag of words" model first developted in "A Neural
             Probabilistic Language Model" by Bengio et al. in the Journal of
             Machine Learning Research.

    ARGS:
    :config: model configuration
    """
    def __init__(self,
                 model_config: BOWModelConfig,
                 run_config: tf.estimator.RunConfig,
                 embedding_matrix: List[List[int]],
                 embedding_config: EmbeddingConfig,
                 learning_rate: float = 0.01):
        self.model_config = model_config
        self.run_config = run_config
        self.embedding_config = embedding_config
        self.embedding_matrix = embedding_matrix
        self.params = {'learning_rate': learning_rate}
        self.model_fn = self.model_function_builder()
        self.generate_estimator()

    def _embedding_lookup_layer_(self, embedding_matrix, token_ids_, reduce='sum'):
        """
        PURPOSE: Constructing the Embedding Look up Layer

        ARGS:
        embedding_mat_       (tf.tensor) product embedding matrix
        token_ids_     (tf.tensor) product numbers of included product in each sequences
        reduce               (str) reduce method 'mean' or 'sum'

        RETURNS:
        embedding_sum_       (tf.tensor) vector representation of an order.
        """
        with tf.variable_scope("EmbeddingLookup"):
            embedding_sequence_ = tf.gather(embedding_matrix, indices=token_ids_)
            if reduce == 'sum':
                embedding_sum_ = tf.reduce_sum(embedding_sequence_, axis=1, name='BagOfWords')
            elif reduce == 'mean':
                embedding_sum_ = tf.reduce_mean(embedding_sequence_, axis=1, name='BagOfWords')

        return embedding_sum_

    def __activation_lookup__(self, layer, layer_name):
        """
        PURPOSE: Applying an activation function to a layer

        ARGS:
        layer:      (tf.tensor) layer activation function will be applied
        layer_name: (str) name of layer

        RETURNS:
        activation_ (tf.tensor) layer with activation function applied
        """
        if self.model_config.hidden_activation == 'elu':
            activation_ = tf.nn.elu(layer, name=layer_name)
        if self.model_config.hidden_activation == 'relu':
            activation_ = tf.nn.relu(layer, name=layer_name)
        if self.model_config.hidden_activation == 'leaky_relu':
            activation_ = tf.nn.leaky_relu(layer, name=layer_name)
        return activation_

    def _fully_connected_layer_(self, embedding_sum_, training_):
        """
        PURPOSE: Constructing the sequence of hidden layers

        ARGS:
        embedding_sum_         (tf.tensor) vector representation of tokens
        training_              (tf.tensor) indicator for training or testing task

        RETURNS:
        h_          (tf.tensor) output of last hidden layer
        """
        he_init_ = tf.initializers.he_uniform()
        h_ = embedding_sum_
        for i, dim in enumerate(self.model_config.n_hidden):
            with tf.variable_scope(("HiddenLayer_%d" % i)):
                if self.model_config.batch_normalization:
                    h_ = tf.layers.dense(h_, dim, kernel_initializer=he_init_, name=("Hidden_%d_b4_bn" % i))
                    h_ = tf.layers.batch_normalization(h_, training=training_, name=("Hidden_%d_bn" % i))
                    h_ = self.__activation_lookup__(h_, ("Hidden_%d_act" % i))
                else:
                    h_ = tf.layers.dense(h_, dim, activation=self.model_config.hidden_activation,
                                         kernel_initializer=he_init_, name=("Hidden_%d" % i))
                if self.model_config.drop_rate > 0:
                    h_ = tf.layers.dropout(h_, rate=self.model_config.drop_rate, training=training_)
        return h_

    def _output_layer_(self, h_):
        """
        PURPOSE: Constructing the logits output layer

        ARGS:
        h_      (tf.tensor) output of last hidden layer

        RETURNS:
        logits_ (tf.tensor) logits output layer
        """
        with tf.variable_scope("OutputLayer"):
            logits_ = tf.layers.dense(h_, self.model_config.num_outputs, name='Logits_lyr')
            soft_max_ = tf.nn.softmax(logits_, name="SoftMax")
        return logits_, soft_max_

    def _loss_function_(self, logits_, labels_):
        """
        PURPOSE:Constructing the cross entropy loss function

        ARGS:
        logits_     (tf.tensor) logits output layer
        labels_          (tf.tensor) order class label

        RETURN:
        xentropy    (tf.tensor) raw cross entropy values
        loss_       (tf.tensor) mean cross cross entropy
        """
        with tf.variable_scope("Loss"):
            xentropy_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_, logits=logits_, name="Xentropy")
            loss_ = tf.reduce_mean(xentropy_, name="Loss")

        return xentropy_, loss_

    def _optimizer_(self, loss_, learning_rate: float):
        """
        PURPOSE: Constructing the optimizer and training method

        ARGS:
        loss_       (tf.tensor) mean cross entropy values

        RETURNS:
        optimizer_      (tf.tensor) optimizer
        training_op_    (tf.tensor) training method
        """
        with tf.variable_scope("Optimizer"):
            optimizer_ = tf.train.AdagradOptimizer(learning_rate=learning_rate)
            training_op_ = optimizer_.minimize(loss_, global_step=tf.train.get_global_step())

        return optimizer_, training_op_

    def _evaluation_(self, logits_, labels_):
        """
        PURPOSE: Constructing the Evaluation Piece

        ARGS:
        logits_     (tf.tensor) output layer
        labels_     (tf.tensor) order class labels

        RETURNS:
        correct_    (tf.tensor) number of correct classifications
        accuracy_    (tf.tensor) accuracy on entire dataset
        """
        correct_ = tf.nn.in_top_k(targets=labels_, predictions=logits_, k=1)
        accuracy_ = tf.reduce_mean(tf.cast(correct_, tf.float32))
        return correct_, accuracy_

    def _initializer_(self):
        """
        PURPOSE: Initializing all Variables

        RETURNS:
        init_       (tf.tensor) initializer for all graph variables
        saver_      (tf.tensor) saver method
        """
        init_ = tf.global_variables_initializer()
        saver_ = tf.train.Saver()
        return init_, saver_

    def model_function_builder(self):
        """
        """

        def model_fn(features, labels, mode, params):
            """
            """
            with tf.variable_scope("InputLayer"):
                input_layer = tf.reshape(features['input_ids'], [-1, self.embedding_config.max_sequence_length])

            training_ = (mode == tf.estimator.ModeKeys.TRAIN)
            embedding_tensor = tf.convert_to_tensor(self.embedding_matrix)
            input_embedding = self._embedding_lookup_layer_(embedding_tensor, input_layer)
            hidden_output = self._fully_connected_layer_(input_embedding, training_)
            logits, soft_max = self._output_layer_(hidden_output)
            predictions = tf.math.argmax(soft_max, axis=1)

            if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
                xentropy, loss = self._loss_function_(logits, labels['labels'])

                with tf.variable_scope("EvalMetrics"):
                    accuracy = tf.metrics.accuracy(labels['labels'], predictions)
                    tf.summary.scalar('Accuracy', accuracy[0])
                    recall = tf.metrics.recall(labels['labels'], predictions)
                    tf.summary.scalar('Recall', recall[0])
                    precision = tf.metrics.precision(labels['labels'], predictions)
                    tf.summary.scalar('Precision', precision[0])

                eval_ops = {"recall": recall, "accuracy": accuracy, "precision": precision}
            else:
                loss, eval_ops = None, None

            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer, train_op = self._optimizer_(loss, learning_rate=params['learning_rate'])
            else:
                train_op = None

            return tf.estimator.EstimatorSpec(mode, predictions=predictions, loss=loss,
                                              train_op=train_op, eval_metric_ops=eval_ops)

        return model_fn

    def input_fn_builder(self, dataset: QuoraTFDataset, mode=tf.estimator.ModeKeys.TRAIN, num_epochs: int = 1):
        """
        """
        if mode == tf.estimator.ModeKeys.TRAIN:
            data = dataset.train_data.feature_dict_generator
            batch_size = dataset.batch_size
        else:
            data = dataset.valid_data.feature_dict_generator
            batch_size = len(dataset.valid_data.observations)

        def input_fn():
            """
            """
            dataset = tf.data.Dataset.from_generator(
                data,
                output_types={
                    "token_ids": tf.int64,
                    "token_types": tf.int64,
                    "attention_mask": tf.int64,
                    "label": tf.int64
                }).batch(batch_size)
            dataset.repeat(num_epochs)

            iterator = dataset.make_one_shot_iterator()
            feature_dict = iterator.get_next()
            return {"input_ids": feature_dict["token_ids"]}, {"labels": feature_dict['label']}

        return input_fn

    def generate_estimator(self):
        """
        """
        self.estimator = tf.estimator.Estimator(
            model_fn=self.model_fn, model_dir=self.model_config.model_dir,
            config=self.run_config, params=self.params)

    def train_model(self, dataset: QuoraTFDataset, num_epochs):
        """
        """
        input_fn = self.input_fn_builder(dataset, mode=tf.estimator.ModeKeys.TRAIN, num_epochs=1)
        self.estimator.train(input_fn)
        input_fn = self.input_fn_builder(dataset, mode=tf.estimator.ModeKeys.EVAL, num_epochs=1)
        self.estimator.evaluate(input_fn)

