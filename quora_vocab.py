from __future__ import print_function
from __future__ import division
from tqdm import tqdm as ProgressBar

from collections import defaultdict, Counter

class CommentVocab(object):

    def __init__(self,comments):
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(lambda: Counter())
        for _,comment,_ in ProgressBar(comments,desc='Processing Comments'):
            prev_word = None
            for word in comment:
                self.unigram_counts[word] += 1
                self.bigram_counts[prev_word][word] += 1
                prev_word = word
        self.bigram_counts.default_factory = None  # make into a normal dict

        # Leave space for "<s>", "</s>", and "<unk>"
        top_counts = self.unigram_counts.most_common(None if size is None else (size - 3))
        vocab = ([self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN] +
                 [w for w,c in top_counts])

        # Assign an id to each word, by frequency
        self.id_to_word = dict(enumerate(vocab))
        self.word_to_id = {v:k for k,v in self.id_to_word.items()}
        self.size = len(self.id_to_word)
        if size is not None:
            assert(self.size <= size)

        # For convenience
        self.wordset = set(self.word_to_id.keys())
