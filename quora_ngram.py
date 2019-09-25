from __future__ import print_function
from __future__ import division
from tqdm import tqdm as ProgressBar
import numpy as np
import utilities as ut
from collections import defaultdict, Counter


class NgramModel(object):

    def __init__(self,comments):
        self._vocab_counts_(comments)


    def _vocab_counts_(self,comments):
        """
        PURPOSE: Perform initial counting of uni/bi/trigrams present in all comments
                 grouped by classification

        ARGS: 
        comments        (tuple) of the following 
            comments[0]     (str)       unique comment id
            comments[1]     (list(str)) comments as list of tokens
            comments[2]     (int)       class    
        """
        # Attribute Setup
        self.unigram_counts = [Counter(),Counter()]
        self.bigram_counts = [defaultdict(lambda: Counter()),
                              defaultdict(lambda: Counter())]
        self.trigram_counts = [defaultdict(lambda:  Counter()),
                               defaultdict(lambda:  Counter())]

        # Token Processing
        for _,comment,label in ProgressBar(comments, desc='Processing Comments'):
            comment = comment
            prev_unigram = None
            prev_bigram = None  
            for word in comment:
                self.unigram_counts[label][word] += 1
                if prev_unigram is not None:
                    self.bigram_counts[label][prev_unigram][word] += 1
                    if prev_bigram is not None: 
                        self.trigram_counts[label][prev_bigram][word] += 1
                if prev_unigram is not None: 
                    prev_bigram = '_'.join([prev_unigram,word])
                prev_unigram = word

            # Uni/Bi/Trigram Totals
            self.unigram_totals = [sum(self.unigram_counts[i].values()) for i in [0,1]]
            self.bigram_totals = [sum([sum(endgram_counts.values()) for endgram_counts in self.bigram_counts[i].values()])
                                    for i in [0,1]]
            self.trigram_totals = [sum([sum(endgram_counts.values()) for endgram_counts in self.trigram_counts[i].values()]) 
                                    for i in [0,1]]

            # Converting counter to regular dictionaries 
#            self.bigram_counts[0].default_factory = None  
#            self.bigram_counts[1].default_factory = None 
#            self.trigram_counts[0].default_factory = None  
#            self.trigram_counts[1].default_factory = None 
        
    def _additive_smoothing_(self,gram_length=1,param=1):
        """
        PURPOSE: Calculate the smoothed frequencies of the requested uni/bi/trigrams 

        ARGS: 
        gram_length         (int) gram frequencies requested (1) unigram, (2) bigram, (3) trigram
        param               (float) smoothing parameter
        """
        def smooth_freq(gram_count,gram_total_count,vocab_count,param=param): 
            return np.log((param + gram_count)/(param*vocab_count + gram_total_count))

        # Unigram Frequencies
        if gram_length == 1: 
            vocab_size = [len(self.unigram_counts[i].keys()) for i in [0,1]]
            self.gram_frequency = [ { gram:smooth_freq(gram_count,self.unigram_totals[i],vocab_size[i],param) 
                                        for gram, gram_count in self.unigram_counts[i].items()}
                                     for i in [0,1]]
            # Adding smoothed values for words not in training vocab
            for i in [0,1]:
                self.gram_frequency[i].update({'<unk>': smooth_freq(1,self.unigram_totals[i],vocab_size[i],param)}) 

        # Bigram Frequencies
        if gram_length == 2: 
            vocab_size = [sum([len(endgram_counts.keys()) for endgram_counts in self.bigram_counts[i].values()])
                            for i in [0,1]]
            self.gram_frequency = [ [ {'_'.join([gram1,gram2]):smooth_freq(gram_count,self.bigram_totals[i],vocab_size[i],param) 
                                        for gram2,gram_count in self.bigram_counts[i].get(gram1).items()} 
                                      for gram1 in self.bigram_counts[i].keys() ]
                                    for i in [0,1] ]
            self.gram_frequency = [ { k:v for d in self.gram_frequency[i] for k,v in d.items()} for i in [0,1]] 
            # Adding smoothed values for words not in training vocab
            for i in [0,1]:
                self.gram_frequency[i].update({'<unk>': smooth_freq(1,self.bigram_totals[i],vocab_size[i],param)}) 

        # Trigram Frequencies
        if gram_length == 3: 
            vocab_size = [sum([len(endgram_counts.keys()) for endgram_counts in self.trigram_counts[i].values()])
                            for i in [0,1]]
            self.gram_frequency = [ [ {'_'.join([gram1,gram2]):smooth_freq(gram_count,self.trigram_totals[i],vocab_size[i],param) 
                                        for gram2,gram_count in self.trigram_counts[i].get(gram1).items()} 
                                      for gram1 in self.trigram_counts[i].keys() ]
                                    for i in [0,1] ]
            #Flattening a list of dict into a single dict
            self.gram_frequency = [ { k:v for d in self.gram_frequency[i] for k,v in d.items()} for i in [0,1]]
            # Adding smoothed values for words not in training vocab
            for i in [0,1]:
                self.gram_frequency[i].update({'<unk>': smooth_freq(1,self.trigram_totals[i],vocab_size[i],param)}) 


    def predict(self,comment,gram_length=1,smoothing='additive',**kwargs): 
        """
        PURPOSE: Predict the classification of a tokenized comment with a ngram model of the specified length
                 with the specified smoothing technique

        ARGS: 
        comment             (list(str)) list of tokens representing the comment
        gram_length         (int) gram frequencies requested (1) unigram, (2) bigram, (3) trigram
        smoothing           (str) smoothing technique used in ['additive'] 
        **kwargs            parameters needed for the indicated smoothing method

        RETURNS: 
        predicted           (int) predicted class
        log_probabilities   (list(float)) calculated log likelihood
        """
        if smoothing == 'additive': 
            param = kwargs.get('param',1)
            self._additive_smoothing_(gram_length,param)

        if gram_length == 1: 
            unk_prob = [self.gram_frequency[i].get('<unk>') for i in [0,1]]
            log_probs = [ sum([ self.gram_frequency[i].get(gram,unk_prob[i]) for gram in comment]) for i in [0,1]]

        predicted = int(log_probs[0] < log_probs[1])    
        return predicted, log_probs
