from __future__ import print_function
from __future__ import division
from tqdm import tqdm as ProgressBar
import numpy as np
import utilities as ut
from collections import defaultdict, Counter


class NgramModel(object):

    def __init__(self,comments):
        pass


    def _vocab_counts_(self,comments):
        self.unigram_counts = [Counter(),Counter()]
        self.bigram_counts = [defaultdict(lambda: Counter()),
                              defaultdict(lambda: Counter())]
        for _,comment,label in ProgressBar(comments,desc='Processing Comments'):
            comment = ['<s>'] + comment + ['</s>']
            prev_word = None
            for word in comment:
                self.unigram_counts[label][word] += 1
                if prev_word is not None:
                    self.bigram_counts[label][prev_word][word] += 1
                prev_word = word
        self.bigram_counts[0].default_factory = None  # make into a normal dict
        self.bigram_counts[1].default_factory = None  # make into a normal dict
