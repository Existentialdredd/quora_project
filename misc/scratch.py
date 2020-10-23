import unittest
from collections import defaultdict
from quora_ngram import NgramModel
import pprint

pp = pprint.PrettyPrinter(indent=4)

comments = [['xxxxx',['where','where','where','are','are','you','you','now'],1],
            ['yyyyy',['who','who','who','are','are','you','you','you'],0]]
ngrams = NgramModel(comments)
#gram_models = [ngrams._additive_smoothing_(gram_length=1,param=0,counts=False,logs=False),
#               ngrams._additive_smoothing_(gram_length=2,param=0,counts=False,logs=False),
#               ngrams._additive_smoothing_(gram_length=3,param=0,counts=True,logs=False)]
#pp.pprint(gram_models)
#pp.pprint(ngrams._smoothed_gram_freqs_('who_are_are',gram_models,0.5,logs=False))
#
#test_level_jm_1 = ngrams.jelinek_mercer(1,cnvx_param=0.5,scnd_smoother='additive',param=0)
#pp.pprint(test_level_jm_1)
#test_level_jm_2 = ngrams.jelinek_mercer(2,cnvx_param=0.5,scnd_smoother='additive',param=0)
#pp.pprint(test_level_jm_2)
#test_level_jm_3 = ngrams.jelinek_mercer(3,cnvx_param=0.5,scnd_smoother='additive',param=0)
#pp.pprint(test_level_jm_3)

ngrams.train_classifier(gram_length=2,smoothing='jelinek-mercer',
                                       scnd_smoother='additive',
                                       cnvx_param=0.5,
                                       smth_param=1)
pp.pprint(ngrams.gram_frequency)

test_comment = [[ 'where', 'the' , 'heck'],['who','are','those','people']]
test_comment_labels = [0,1]
prediction, log_probabilities = ngrams.predict(test_comment[1])
print('Predicted Class is {} with Log Probabilities {:5.4f} and {:5.4f}'.format(prediction,
                                               log_probabilities[0],log_probabilities[1]))
#
#ngrams.evaluate_classifier(test_comment,test_comment_labels)


#test_gram_models = [ [{'123':1,'456':2,'789':3},{'123':10,'456':20,'789':30}],
#                [{'123_456':4,'456_789':5},{'123_456':40,'456_789':50}],
#                [{'123_456_789':6},{'123_456_789':60}]] 
#test_gram = '123_456_789'
#pp.pprint(ngrams._gram_freq_(test_gram,test_gram_models))
#ngrams._smoothed_gram_freqs_(test_gram,test_gram_models)
