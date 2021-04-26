from __future__ import print_function
from __future__ import division
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import List, Dict
from data.tf_dataset import QuoraTFDataset
from embeddings.embedding_utilities import EmbeddingConfig
import os
import sys
import numpy as np
import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)

@dataclass_json
@dataclass
class BOWModelConfig:
    """
    PURPOSE:

    ARGS:
    :learning_rate: ml gradient descent learning rate
    :max_sequence_length: maximum length of token sequence
    :embedding_dimension: truncated length of embedding vectors
    :batch_normalization: indicator for use of batch normalization
    :n_hidden: list of neurons per hidden layer
    :hidden_activation: hidden layer activation function
    :dropout_rate: probability of dropout
    :n_tokens: total number of rows in embed matrix
    :n_outputs: number of unique labels
    :root_log_dir: directory where tf logs will be kept
    :checkpoint_dir: directory where checkpoints are kept
    """
    learning_rate: float = 0.01
    max_sequence_length: int = 100
    embedding_dimension: int = 100
    batch_normalization: bool = True
    n_hidden: List[int] = field(default_factory=list)
    hidden_activation: str = 'elu'
    drop_rate: float = 0.5
    num_tokens: int = 100
    num_outputs: int = 2
    rootdir: str = os.path.dirname(os.path.abspath(__file__)) + '/runs'
    reduce: str = 'sum'

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


class EmbeddingLookup(tf.keras.layers.Layer):

    def __init__(self, model_config: BOWModelConfig, embedding_list: List[List[int]]):
        self.layer_name = "EmbeddingLookup"
        super(EmbeddingLookup, self).__init__(name=self.layer_name)
        self.model_config = model_config
        self.embedding_tensor = tf.convert_to_tensor(embedding_list)

    def call(self, inputs):
        with tf.name_scope(self.layer_name):
            embedding_sequence = tf.gather(self.embedding_tensor, indices=inputs)
            if self.model_config.reduce == 'sum':
                reduced_embedding = tf.reduce_sum(embedding_sequence, axis=1, name='BagOfWords')
            elif self.model_config.reduce == 'mean':
                reduced_embedding = tf.reduce_mean(embedding_sequence, axis=1, name='BagOfWords')

        return reduced_embedding


class HiddenActivation(tf.keras.layers.Layer):

    def __init__(self, model_config: BOWModelConfig):
        super(HiddenActivation, self).__init__()
        self.model_config = model_config

    def call(self, inputs):
        if self.model_config.hidden_activation == 'elu':
            activation = tf.nn.elu(inputs, name="ELUActivation")
        if self.model_config.hidden_activation == 'relu':
            activation = tf.nn.relu(inputs, name="ReluActivation")
        if self.model_config.hidden_activation == 'leaky_relu':
            activation = tf.nn.leaky_relu(inputs, name="LeakyReluActivation")

        return activation


class HiddenLayer(tf.keras.layers.Layer):

    def __init__(self, model_config: BOWModelConfig, index: int):
        self.model_config = model_config
        self.layer_name = f"HiddenLayer-{index}-"
        super(HiddenLayer, self).__init__(name=self.layer_name)
        self.units = self.model_config.n_hidden[index]
        self.activation_fn = HiddenActivation(model_config=self.model_config)

        if self.model_config.batch_normalization:
            self.batch_normalizer = tf.keras.layers.BatchNormalization(name=self.layer_name + "-BatchNormalization")

        if self.model_config.drop_rate > 0:
            self.dropout = tf.keras.layers.Dropout(rate=self.model_config.drop_rate)

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer="random_normal", trainable=True,
                                 name=self.layer_name+"-Projection")
        self.b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True,
                                 name=self.layer_name+"-Bias")

    def call(self, inputs, training: bool = False):
        with tf.name_scope(self.layer_name):
            linear_proj = tf.matmul(inputs, self.w) + self.b

            if training:
                if self.model_config.batch_normalization:
                    linear_proj = self.batch_normalizer(linear_proj, training=training)

            activated_proj = self.activation_fn(linear_proj)

            if training:
                if self.model_config.drop_rate > 0:
                    activated_proj = self.dropout(activated_proj)

        return activated_proj


class OutputLayer(tf.keras.layers.Layer):

    def __init__(self, model_config: BOWModelConfig):
        self.model_config = model_config
        self.layer_name = "OutputLayer"
        super(OutputLayer, self).__init__(name=self.layer_name)
        self.units = self.model_config.num_outputs

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer="random_normal", trainable=True,
                                 name=self.layer_name+"-Projection")
        self.b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True,
                                 name=self.layer_name+"-Bias")

    def call(self, inputs):
        with tf.name_scope(self.layer_name):
            logits = tf.matmul(inputs, self.w) + self.b
            soft_max = tf.nn.softmax(logits, name=self.layer_name + "-SoftMax")

        return logits, soft_max


class LossLayer(tf.keras.layers.Layer):

    def __init__(self, model_config: BOWModelConfig):
        super(LossLayer, self).__init__(name="LossLayer")

    def call(self, logits, labels=None):
        if labels is not None:
            with tf.name_scope("Loss"):
                xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name="Xentropy")
                loss = tf.reduce_mean(xentropy, name="Loss")
                self.add_loss(loss)


class EvalLayer(tf.keras.layers.Layer):

    def __init__(self, model_config: BOWModelConfig):
        self.model_config = model_config
        super(EvalLayer, self).__init__(name="EvalLayer")

    def call(self, logits, labels=None):
        correct = tf.nn.in_top_k(targets=labels, predictions=logits, k=1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        self.add_metric(accuracy, name="accuracy")


class BOWModel(tf.keras.Model):

    def __init__(self, model_config: BOWModelConfig, embedding_list: List[List[float]]):
        self.model_config = model_config
        super(BOWModel, self).__init__(name='BOWModel')
        self.embedding_layer = EmbeddingLookup(self.model_config, embedding_list)
        self.hidden_layers = [
            HiddenLayer(self.model_config, index=idx)
            for idx, _ in enumerate(self.model_config.n_hidden)
        ]
        self.output_layer = OutputLayer(self.model_config)
        self.loss_layer = LossLayer(self.model_config)
        self.eval_layer = EvalLayer(self.model_config)
        self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=self.model_config.learning_rate)

    def call(self, inputs, training=False):
        embedding_val = self.embedding_layer(inputs['token_ids'])
        hidden_layer_vals = [self.hidden_layers[0](embedding_val, training)]
        for idx, hidden_layer in enumerate(self.hidden_layers[1:]):
            hidden_layer_vals.append(hidden_layer(hidden_layer_vals[idx], training))

        logits, soft_max = self.output_layer(hidden_layer_vals[-1])
        self.loss_layer(logits, inputs['label'])
        self.eval_layer(logits, inputs['label'])

        return logits

    def train_step(self, inputs: Dict):
        """
        """
        with tf.GradientTape() as tape:
            logits = self(inputs, training=True)
            loss = self.losses

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        output_dict = {m.name: m.result() for m in self.metrics}
        output_dict.update({"Loss": loss[0]})
        # print(output_dict)

        return output_dict
