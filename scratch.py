import unittest
from collections import defaultdict
from quora_ngram import NgramModel
import pprint

pp = pprint.PrettyPrinter(indent=4)

comments = [['xxxxx',['where','where','where','are','are','you','you','now'],1],
            ['yyyyy',['who','who','who','are','are','you','you','you'],0]]
ngrams = NgramModel(comments)
#
test_comment = [[ 'where', 'the' , 'heck'],['who','are','those','people']]
test_comment_labels = [0,1]
test_level_jm_1 = ngrams.jelinek_mercer(2,param=0.5,scnd_smoother='good_turing')
pp.pprint(ngrams.jelinek_mercer(2,param=0.5,scnd_smoother='good_turing'))
pp.pprint(ngrams.good_turing(gram_length=2,param=0.5,))
#ngrams.train_classifier(gram_length=1,smoothing='additive')
#pp.pprint(ngrams.gram_frequency)
#prediction, log_probabilities = ngrams.predict(test_comment[0])
#print('Predicted Class is {} with Log Probabilities {:5.4f} and {:5.4f}'.format(prediction,
#                                               log_probabilities[0],log_probabilities[1]))
#
#ngrams.evaluate_classifier(test_comment,test_comment_labels)


#test_gram_models = [ [{'123':1,'456':2,'789':3},{'123':10,'456':20,'789':30}],
#                [{'123_456':4,'456_789':5},{'123_456':40,'456_789':50}],
#                [{'123_456_789':6},{'123_456_789':60}]] 
#test_gram = '123_456_789'
#pp.pprint(ngrams._gram_freq_(test_gram,test_gram_models))
#ngrams._smoothed_gram_freqs_(test_gram,test_gram_models)
