import unittest
from collections import defaultdict
from quora_ngram import NgramModel 

comments = [['xxxxx',['where','where','where','are','are','you','you','now'],1],
            ['yyyyy',['who','who','who','are','are','you','you','you'],0]]
ngrams = NgramModel(comments)
ngrams._additive_smoothing_(1,1)
print(ngrams.gram_frequency)
#
test_comment = [ 'where', 'the' , 'heck'] 
prediction, log_probabilities = ngrams.predict(test_comment,param=0.5)
print('Predicted Class is {} with Log Probabilities {:5.4f} and {:5.4f}'.format(prediction,log_probabilities[0],log_probabilities[1]))

