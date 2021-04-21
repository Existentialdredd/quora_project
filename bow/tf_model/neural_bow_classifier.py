from __future__ import print_function
from __future__ import division
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from tqdm import tqdm as ProgressBar
from typing import List
from data.tf_dataset import QuoraTFDataset
import datetime as dt
import random
import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)


@dataclass_json
@dataclass
class BOWConfig:
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
    root_log_dir: str = os.path.dirname(os.path.abspath(__file__)) + '/logs'
    check_pt_dir: str = os.path.dirname(os.path.abspath(__file__)) + '/ckpts'
    summary_dir: str = os.path.dirname(os.path.abspath(__file__)) + '/summary'

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
    def __init__(self, config):
        self.config = config
        self.graph = None

    def _placeholders_(self):
        """
        PURPOSE: Construct graph placeholder variables.

        RETURNS:
        token_ids_  (tf.tensor) input token sequences
        embedding_mat_    (tf.tensor) matrix of embeddings
        labels_                (tf.tensor) class labels
        training_         (tf.tensor) training indicator
        """
        token_ids_ = tf.placeholder(tf.int32,
                                    shape=[None, self.config.max_sequence_length],
                                    name='TokenIDs')
        embedding_mat_ = tf.placeholder(tf.float32,
                                        shape=[self.config.num_tokens, self.config.embedding_dimension],
                                        name='W_embed')
        labels_ = tf.placeholder(tf.int64, shape=[None], name='Labels')
        training_ = tf.placeholder_with_default(False, shape=(), name='training')

        return token_ids_, embedding_mat_, labels_, training_

    def _embedding_lookup_layer_(self,
                                 embedding_mat_,
                                 token_ids_,
                                 reduce='sum'):
        """
        PURPOSE: Constructing the Embedding Look up Layer

        ARGS:
        embedding_mat_       (tf.tensor) product embedding matrix
        token_ids_     (tf.tensor) product numbers of included product in each sequences
        reduce               (str) reduce method 'mean' or 'sum'

        RETURNS:
        embedding_sum_       (tf.tensor) vector representation of an order.
        """
        embedding_sequence_ = tf.nn.embedding_lookup(embedding_mat_,
                                                     token_ids_,
                                                     name='Embeddings')
        if reduce == 'sum':
            embedding_sum_ = tf.reduce_sum(embedding_sequence_,
                                           axis=1,
                                           name='BagOfWords')
        elif reduce == 'mean':
            embedding_sum_ = tf.reduce_mean(embedding_sequence_,
                                            axis=1,
                                            name='BagOfWords')
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
        if self.config.hidden_activation == 'elu':
            activation_ = tf.nn.elu(layer, name=layer_name)
        if self.config.hidden_activation == 'relu':
            activation_ = tf.nn.relu(layer, name=layer_name)
        if self.config.hidden_activation == 'leaky_relu':
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
        for i, dim in enumerate(self.config.n_hidden):
            with tf.variable_scope(("HiddenLayer_%d" % i)):
                if self.config.batch_normalization:
                    h_ = tf.layers.dense(h_,
                                         dim,
                                         kernel_initializer=he_init_,
                                         name=("Hidden_%d_b4_bn" % i))
                    h_ = tf.layers.batch_normalization(h_,
                                                       training=training_,
                                                       name=("Hidden_%d_bn" %
                                                             i))
                    h_ = self.__activation_lookup__(h_, ("Hidden_%d_act" % i))
                else:
                    h_ = tf.layers.dense(h_,
                                         dim,
                                         activation=self.config.hidden_activation,
                                         kernel_initializer=he_init_,
                                         name=("Hidden_%d" % i))
                if self.config.drop_rate > 0:
                    h_ = tf.layers.dropout(h_,
                                           rate=self.config.drop_rate,
                                           training=training_)
        return h_

    def _output_layer_(self, h_):
        """
        PURPOSE: Constructing the logits output layer

        ARGS:
        h_      (tf.tensor) output of last hidden layer

        RETURNS:
        logits_ (tf.tensor) logits output layer
        """
        logits_ = tf.layers.dense(h_, self.config.num_outputs, name='Logits_lyr')
        return logits_

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
        xentropy_ = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels_, logits=logits_, name="Xentropy")
        loss_ = tf.reduce_mean(xentropy_, name="Loss")

        return xentropy_, loss_

    def _optimizer_(self, loss_):
        """
        PURPOSE: Constructing the optimizer and training method

        ARGS:
        loss_       (tf.tensor) mean cross entropy values

        RETURNS:
        optimizer_      (tf.tensor) optimizer
        training_op_    (tf.tensor) training method
        """
        optimizer_ = tf.train.AdagradOptimizer(learning_rate=self.config.learning_rate)
        training_op_ = optimizer_.minimize(loss_)
        return optimizer_, training_op_

    def _evaluation_(self, logits_, labels_):
        """
        PURPOSE: Constructing the Evaluation Piece

        ARGS:
        logits_     (tf.tensor) output layer
        labels_          (tf.tensor) order class labels

        RETURNS:
        correct_    (tf.tensor) number of correct classifications
        accuracy_    (tf.tensor) accuracy on entire dataset
        """
        correct_ = tf.nn.in_top_k(logits_, labels_, 1)
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

    def build_graph(self):
        """
        PURPOSE: Building the lazily executed tensor flow graph
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope("PlaceHolders"):
                token_ids_, embedding_mat_, labels_, training_ = self._placeholders_(
                )
            for var in (token_ids_, embedding_mat_, labels_, training_):
                tf.add_to_collection("Input_var", var)

            with tf.name_scope("EmbeddingLyr"):
                embedding_sum_ = self._embedding_lookup_layer_(
                    embedding_mat_, token_ids_)

            with tf.name_scope("HiddenLyr"):
                h_ = self._fully_connected_layer_(embedding_sum_, training_)

            with tf.name_scope("OutputLyr"):
                logits_ = self._output_layer_(h_)
            tf.add_to_collection("LogitsLyr", logits_)

            with tf.name_scope("Loss"):
                xentropy_, loss_ = self._loss_function_(logits_, labels_)
            for op in (xentropy_, loss_):
                tf.add_to_collection("Loss_ops", op)
            with tf.name_scope("Optimizer"):
                optimizer_, training_op_ = self._optimizer_(loss_)
            for op in (optimizer_, training_op_):
                tf.add_to_collection("Optimizer_ops", op)

            with tf.name_scope("Evaluation"):
                correct_, accuracy_ = self._evaluation_(logits_, labels_)
            for op in (correct_, accuracy_):
                tf.add_to_collection("Eval_ops", op)

            with tf.name_scope("VariableInit"):
                init_, saver_ = self._initializer_()
            for op in (init_, saver_):
                tf.add_to_collection("Init_Save_ops", op)
        # Generating file names
        self._file_names_()

    def _file_names_(self):
        """
        PURPOSE: Constructing appropriate file names for logging, and checkpointing
        """
        now_time = dt.datetime.now().second
        file_ext = ''.join([
            'FF', '_',
            str(self.config.embedding_dimension), '_',
            str(len(self.config.n_hidden)), 'x',
            str(self.config.n_hidden[0]), '-t',
            str(now_time)
        ])
        self.log_dir = ''.join([self.config.root_log_dir, '/run-', file_ext, '/'])
        self.temp_ckpt = ''.join(
            [self.config.check_pt_dir, '/run-', file_ext, '/', 'temp.ckpt'])
        self.final_ckpt = ''.join(
            [self.config.check_pt_dir, '/run-', file_ext, '/', 'final.ckpt'])
        self.summary_file = ''.join(
            [self.config.summary_dir, '/run-', file_ext, '.txt'])

    def write_graph(self):
        """
        PURPOSE: Writing the graph to a log file so the it can be reused and or
                 viewed in tensorboard
        """
        if self.graph is not None:
            with tf.Session(graph=self.graph) as sess:
                init_, _ = tf.get_collection('Init_Save_ops')
                init_.run()
                file_writer = tf.summary.FileWriter(self.log_dir, self.graph)
        else:
            print('No Current Graph')

    def _partition_(self, list_in, n):
        """
        PURPOSE: Generate indexs for minibatchs used in minibatch gradient descent
        """
        random.shuffle(list_in)
        return [list_in[i::n] for i in range(n)]

    def train_graph(self, dataset: QuoraTFDataset, n_stop: int = 2):
        """
        PURPOSE: Train a deep neural net classifier for baskets of products

        ARGS:
        train_dict                 (dict) dictionary with ALL the following key values
            embeddings             (list(list)) trained product embedding layers
            train_batch['token_ids']  (list(list)) training order product numbers
            labels_train           (list) training order class labels
            valid_data['token_ids']  (list(list)) test order product numbers
            valid_data['label']           (list) test order class
            batch_size             (int) number of training example per mini batch
            n_stop                 (int) early stopping criteria
        """
        embeddings = dataset.embedding_tensor
        train_data, valid_data = dataset.get_list_datasets()
        train_data = list(train_data)
        # train_data_full = list(train_data.make_one_shot_iterator())
        # valid_data = valid_data.make_one_shot_iterator().next()
        done, epoch, acc_reg = 0, 0 [0, 1]

        with self.graph.as_default():
            correct_, accuracy_ = tf.get_collection('Eval_ops')
            acc_summary = tf.summary.scalar('Accuracy', accuracy_)
            file_writer = tf.summary.FileWriter(self.log_dir, self.graph)

        with tf.Session(graph=self.graph) as sess:
            init_, saver_ = tf.get_collection('Init_Save_ops')
            correct_, accuracy_ = tf.get_collection('Eval_ops')
            optimizer_, training_op_ = tf.get_collection("Optimizer_ops")
            token_ids_, embedding_mat_, labels_, training_ = tf.get_collection("Input_var")

            sess.run(init_)
            while done != 1:
                epoch += 1
                # Mini-Batch Training step
                iteration = 0
                for train_batch in ProgressBar(train_data, f"Epoch {epoch} Iterations"):
                    iteration += 1
                    sess.run([training_op_], feed_dict={training_: True, embedding_mat_: embeddings,
                                                        token_ids_: train_batch['token_ids'],
                                                        labels_: train_batch['label']})
                    # Intermediate Summary Writing
                    if iteration % 10 == 0:
                        summary_str = acc_summary.eval(
                            feed_dict={training_: True, embedding_mat_: embeddings,
                                       token_ids_: valid_data['token_ids'], labels_: valid_data['label']})
                        step = epoch * dataset.batch_size + iteration
                        file_writer.add_summary(summary_str, step)
                # Early Stopping Regularization
                if epoch % 2 == 0:
                    # Evaluating the Accuracy of Current Model
                    acc_ckpt = accuracy_.eval(
                        feed_dict={training_: False, embedding_mat_: embeddings,
                                   token_ids_: valid_data['token_ids'], labels_: valid_data['label']})
                    if acc_ckpt > acc_reg[0]:
                        # Saving the new "best" model
                        save_path = saver_.save(sess, self.temp_ckpt)
                        acc_reg = [acc_ckpt, 1]
                    elif acc_ckpt <= acc_reg[0] and acc_reg[1] < n_stop:
                        acc_reg[1] += 1
                    elif acc_ckpt <= acc_reg[0] and acc_reg[1] >= n_stop:
                        # Restoring previous "best" model
                        saver_.restore(sess, self.temp_ckpt)
                        done = 1
                # Calculating Accuracy for Output
                acc_train = accuracy_.eval(
                    feed_dict={training_: False, embedding_mat_: embeddings,
                               token_ids_: train_batch['token_ids'], labels_: train_batch['label']})
                acc_test = accuracy_.eval(
                    feed_dict={training_: False, embedding_mat_: embeddings,
                               token_ids_: valid_data['token_ids'], labels_: valid_data['label']})
                print(f"Register:{acc_reg} Epoch:{epoch} Train Accuracy:{acc_train} Validation Accuracy: {acc_test}")

            # Final Model Save
            save_path = saver_.save(sess, self.final_ckpt)

    def predict_and_report(self,
                           sequences,
                           labels,
                           W_embed,
                           report=True,
                           file=False):
        """
        PURPOSE: Prediction using best model on provided examples and generating
                 report if indicated and labels are provided.

        ARGS:
        sequences      (list(list)) order of product numbers
        labels      (list) order class labels
        W_embed     (list(list)) trained word embedding Matrix
        report      (bool) indicator for whether a report is generated
        file        (bool) indicator whether a summary file is generated
        """
        from sklearn.metrics import confusion_matrix, classification_report
        import json

        with tf.Session(graph=self.graph) as sess:
            _, saver_ = tf.get_collection('Init_Save_ops')
            saver_.restore(sess, self.final_ckpt)
            logits_ = self.graph.get_tensor_by_name(
                'OutputLyr/Logits_lyr/BiasAdd:0')
            token_ids_, embedding_mat_, labels_, training_ = tf.get_collection(
                "Input_var")
            self.logits_prediction = logits_.eval(
                feed_dict={
                    embedding_mat_: W_embed,
                    token_ids_: sequences,
                    training_: False
                })
            self.class_prediction = np.argmax(self.logits_prediction, axis=1)

            if report:
                confusion_mat = confusion_matrix(labels, self.class_prediction)
                print('-----------{}-----------'.format('Confusion Matrix'))
                print(confusion_mat, '\n')
                print(
                    '-----------{}-----------'.format('Classification Report'))
                print(classification_report(labels, self.class_prediction))
            if file:
                summary_dict = self.__dict__.copy()
                class_report_dict = classification_report(
                    labels, self.class_prediction, output_dict=True)
                summary_dict.update(class_report_dict)
                summary_dict.pop('graph', None)
                summary_dict.pop('logits_prediction', None)
                summary_dict.pop('class_prediction', None)
                with open(self.summary_file, 'w') as file:
                    json.dump(summary_dict, file, indent=2)
