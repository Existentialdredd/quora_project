from collections import defaultdict
from dataclasses_json import dataclass_json
from dataclasses import dataclass
from data.schemas import QuoraObservationSet
import os


@dataclass_json
@dataclass
class EmbeddingConfig:
    root_dir: str = None
    embedding_file: str = 'quora_full_embeds.txt'
    hidden_dimension:  int = 100
    max_sequence_length:  int = 64
    pad_to_length: bool = False


class ParagramEmbeddings():

    UNKNOWN = '[UNK]'
    UNKNOWN_ID = 2
    PAD = '[PAD]'
    PAD_ID = 3

    def __init__(self, config: EmbeddingConfig = None):
        """
        """
        self.config = config
        self.token_ids = dict()
        self.embedding_tensor = []
        self.embedding_extraction()

    def embedding_extraction(self):
        """
        PURPOSE: Extracting embeddings and a token to id coorespondance dictionary

        ARGS:
        filename        (str) file where the space seperated embeddings are found

        RETURNS:
        embeddings              (list(list(int))) embedding vectors
        token_to_row_number     (dict) of token id number key value pairs
        """
        embeddings = []
        running_count = 0

        if self.config.root_dir is None:
            root_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(root_dir, self.config.embedding_file)
        else:
            file_path = os.path.join(self.config.root_dir, self.config.embedding_file)

        with open(file_path, 'r') as file:
            for line in file:
                token, *vector = line.split(' ')
                vector_float = list(map(float, vector))[:self.config.hidden_dimension]
                self.embedding_tensor.append(vector_float)
                self.token_ids[token.lower()] = running_count
                running_count += 1

        self.token_ids[self.UNKNOWN] = running_count
        self.token_ids[self.PAD] = running_count + 1
        self.embedding_tensor.append([float(0) for i in range(self.config.hidden_dimension)])

    def get_embedding_ids(self, observation_set: QuoraObservationSet = None):
        """
        PURPOSE: Convert sequences of token sequences to a sequence of seqeunces of
                 corresponding id numbers in token_to_row_number dictionary.

        ARGS:
        sequences               (list(list(str)))
        token_to_row_number     (dict) of token id number key value pairs
        unknown_token           (str) the token in token_to_row_number for the unknown token

        RETURNS:
        id_sequences            (list(list(int)) list of list of id numbers
        """
        for observation in observation_set.observations:
            for token in observation.tokenized_input[:self.config.max_sequence_length]:
                observation.token_ids.append(self.token_ids.get(token, self.UNKNOWN))
                observation.token_types.append(self.UNKNOWN_ID if observation.token_ids[-1] is self.UNKNOWN else 1)
                observation.attention_mask.append(1)

            if self.config.pad_to_length:
                observation.token_ids += [self.token_ids.get(self.PAD, -1)]*(self.config.max_sequence_length - len(observation.token_ids))
                observation.token_types += [self.PAD_ID]*(self.config.max_sequence_length - len(observation.token_types))
                observation.attention_mask += [0]*(self.config.max_sequence_length - len(observation.attention_mask))

        return observation_set
