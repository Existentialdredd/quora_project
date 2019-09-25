import unittest
import numpy as np
from collections import defaultdict
from quora_ngram import NgramModel 

class TestQuoraNgram(unittest.TestCase): 

    def setUp(self):
        comments = [['xxxxx',['where','are','you'],1],['yyyyy',['who','are','you'],0]]
        self.vocab = NgramModel(comments)

    def tearDown(self):
        self.vocab = None

    def test_unigram_counts(self): 
        unigram_counts_target = [{'where': 1, 'are': 1, 'you':1},
                                 {'who': 1, 'are': 1, 'you':1}]
        self.assertDictEqual(self.vocab.unigram_counts[0],unigram_counts_target[1])
        self.assertDictEqual(self.vocab.unigram_counts[1],unigram_counts_target[0])
        self.assertEqual(self.vocab.unigram_totals,[3,3])

    def test_bigram_counts(self): 
        bigram_counts_target = [{'where':{'are':1},
                                'are':{'you':1}},
                                {'who':{'are':1},
                                'are':{'you':1}}]
        self.assertDictEqual(dict(self.vocab.bigram_counts[0]),bigram_counts_target[1])
        self.assertDictEqual(dict(self.vocab.bigram_counts[1]),bigram_counts_target[0])
        self.assertEqual(self.vocab.bigram_totals,[2,2])


    def test_trigram_counts(self): 
        trigram_counts_target = [{'where_are':{'you':1}},
                                 {'who_are':{'you':1}}]
        self.assertDictEqual(dict(self.vocab.trigram_counts[0]),trigram_counts_target[1])
        self.assertDictEqual(dict(self.vocab.trigram_counts[1]),trigram_counts_target[0])
        self.assertEqual(self.vocab.trigram_totals,[1,1])


class TestQuoraNgram2(unittest.TestCase): 

    def setUp(self):
        comments = [['xxxxx',['where','are','you'],1],['yyyyy',['who','are','you'],0]]
        self.vocab = NgramModel(comments)
        

    def tearDown(self):
        self.vocab = None

    def test_unigram_freq(self): 
        self.vocab._additive_smoothing_(1,1)
        unigram_freqs_target = [{'where':np.log((1+1)/(1*3+3)),
                                  'are': np.log((1+1)/(1*3+3)),
                                  'you':np.log((1+1)/(1*3+3)),
                                  '<unk>':np.log((1+1)/(1*3+3))},
                                 {'who':np.log((1+1)/(1*3+3)),
                                  'are': np.log((1+1)/(1*3+3)),
                                  'you':np.log((1+1)/(1*3+3)),
                                  '<unk>':np.log((1+1)/(1*3+3))}]
        self.assertDictEqual(self.vocab.gram_frequency[0],unigram_freqs_target[1])
        self.assertDictEqual(self.vocab.gram_frequency[1],unigram_freqs_target[0])

    def test_bigram_freq(self): 
        self.vocab._additive_smoothing_(2,1)
        bigram_freqs_target = [{'where_are':np.log((1+1)/(1*2+2)),
                                  'are_you': np.log((1+1)/(1*2+2)),
                                  '<unk>':np.log((1+1)/(1*2+2))},
                                 {'who_are':np.log((1+1)/(1*2+2)),
                                  'are_you': np.log((1+1)/(1*2+2)),
                                  '<unk>':np.log((1+1)/(1*2+2))}]
        self.assertDictEqual(self.vocab.gram_frequency[0],bigram_freqs_target[1])
        self.assertDictEqual(self.vocab.gram_frequency[1],bigram_freqs_target[0])


    def test_trigram_freq(self): 
        self.vocab._additive_smoothing_(3,1)
        trigram_freqs_target = [{'where_are_you':np.log((1+1)/(1*1+1)),
                                  '<unk>':np.log((1+1)/(1*1+1))},
                                 {'who_are_you':np.log((1+1)/(1*1+1)),
                                  '<unk>':np.log((1+1)/(1*1+1))}]
        self.assertDictEqual(self.vocab.gram_frequency[0],trigram_freqs_target[1])
        self.assertDictEqual(self.vocab.gram_frequency[1],trigram_freqs_target[0])

if __name__ == '__main__': 
    unittest.main()

