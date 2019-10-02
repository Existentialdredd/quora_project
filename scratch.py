import unittest
from collections import defaultdict
from quora_ngram import NgramModel
import pprint
pp=pprint.PrettyPrinter(indent=4)

comments = [['xxxxx',['where','where','where','are','are','you','you','now'],1],
            ['yyyyy',['who','who','who','are','are','you','you','you'],0]]
ngrams = NgramModel(comments)
ngrams.train_classifier()
# ngrams._additive_smoothing_(1,1)
pp.pprint(ngrams.gram_frequency)
gt_test = ngrams.good_turing()
pp.pprint(gt_test)
#
test_comment = [ 'where', 'the' , 'heck'] 
prediction, log_probabilities = ngrams.predict(test_comment)
print('Predicted Class is {} with Log Probabilities {:5.4f} and {:5.4f}'.format(prediction,
                                               log_probabilities[0],log_probabilities[1]))
