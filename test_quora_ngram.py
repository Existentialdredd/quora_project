import unittest
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

if __name__ == '__main__': 
    unittest.main()

