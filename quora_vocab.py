from __future__ import print_function
from __future__ import division
from tqdm import tqdm as ProgressBar
import numpy as np
import utilities as ut
from collections import defaultdict, Counter

class CommentVocab(object):

    def __init__(self,comments,size=None):
        self.UNK_TOKEN = '<unk>'
        self.VOCAB_SIZE = size
        self._comment_lengths_(comments)
        self._vocab_counts_(comments)


    def _comment_lengths_(self,comments):
        """
        PURPOSE:
        """
        comment_lengths = [Counter([len(comment[1]) for comment in comments if comment[2] == i]) 
                                for i in range(2)]
        comment_lengths = [zip(comment_lengths[0].keys(),comment_lengths[0].values()),
                           zip(comment_lengths[1].keys(),comment_lengths[1].values())] 
        self.sorted_comment_lengths = [sorted(comment_lengths[0],key = lambda x: x[0]),
                                       sorted(comment_lengths[1],key = lambda x: x[0])]


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

        # Leave space for "<s>", "</s>", and "<unk>"
        self.top_counts = [counts.most_common(None if self.VOCAB_SIZE is None else (self.VOCAB_SIZE - 1))
                        for counts in self.unigram_counts ]
        vocab = [self.UNK_TOKEN] + list(set([w for w,c in self.top_counts[0]]
                                           +[w for w,c in self.top_counts[1]]))

        # Assign an id to each word, by frequency
        self.id_to_word = dict(enumerate(vocab))
        self.word_to_id = {v:k for k,v in self.id_to_word.items()}
        # For convenience
        self.wordset = set(self.word_to_id.keys())

    def _unigram_frequencies_(self):
        """
        PURPOSE:
        """
        #Unigram frequencies and differences
        unigram_totals = [ sum(list(counts.values())) for counts in self.unigram_counts]
        unigram_freq = [sorted([ (unigram,count/unigram_totals[i]) 
                        for unigram,count in self.unigram_counts[i].items()]
                               ,key=lambda x: -x[1]) 
                        for i in [0,1]]
        unigram_freq_alt = [ dict(lst) for lst in unigram_freq]
        unigram_diff = [ (word,unigram_freq_alt[0].get(word,int(0))-unigram_freq_alt[1].get(word,int(0)))
                                for word in set(list(unigram_freq_alt[0].keys())
                                              + list(unigram_freq_alt[1].keys()))]
         
        unigram_diff = sorted(unigram_diff,key=lambda x: -x[1])                                

        return unigram_freq, unigram_diff                                

    def _bigram_frequencies_(self):                                
        """
        PURPOSE: 
        """
        #Bigram frequencies and differences
           
        neg_bigram_counts = [[('{}_{}'.format(word1,word2),count) for word2,count in words.items()]
                               for word1,words in list(self.bigram_counts[0].items())]
        neg_bigram_counts = dict(ut.flatten_sort_listx2_tuple(neg_bigram_counts,1))

        pos_bigram_counts = [[('{}_{}'.format(word1,word2),count) for word2,count in words.items()]
                               for word1,words in list(self.bigram_counts[1].items())]
        pos_bigram_counts = dict(ut.flatten_sort_listx2_tuple(pos_bigram_counts,1))
        bigram_counts = [neg_bigram_counts,pos_bigram_counts]
        bigram_totals = [sum(neg_bigram_counts.values()),sum(pos_bigram_counts.values())]
        bigram_vocab = set(list(neg_bigram_counts.keys()) + list(pos_bigram_counts.keys()))

        bigram_freq = [sorted([(key,val/bigram_totals[i]) 
            for key,val in bigram_counts[i].items()],key=lambda x: -x[1]) 
                        for i in [0,1]]
        bigram_freq_alt = [dict(lst) for lst in bigram_freq]
        bigram_diff = [(bigram,bigram_freq_alt[0].get(bigram,int(0))-bigram_freq_alt[1].get(bigram,int(0))) 
                        for bigram in bigram_vocab]
           
        bigram_diff = sorted(bigram_diff,key = lambda x: -x[1])

        return bigram_freq, bigram_diff


    def word_frequency_graphs(self,unigram=True,min_rank=0,max_rank=25): 
        """
        PURPOSE:
        """
        from bokeh.plotting import figure
        from bokeh.io import show
        from bokeh.models import HoverTool, ColumnDataSource

        if unigram: 
            gram_freq,_ = self._unigram_frequencies_()
            title = 'Unigram'
        else:
            gram_freq,_ = self._bigram_frequencies_()
            title = 'Bigram'
            
        neg_vocab_words,neg_vocab_freq= zip(*gram_freq[0][min_rank:max_rank])
        pos_vocab_words,pos_vocab_freq= zip(*gram_freq[1][min_rank:max_rank])

        source = ColumnDataSource(data=dict(neg_vocab_words=neg_vocab_words,
                                            neg_vocab_freq=neg_vocab_freq,
                                            pos_vocab_words=pos_vocab_words,
                                            pos_vocab_freq=pos_vocab_freq))

        hover_1 = HoverTool(tooltips=[('Word', "$neg_vocab_words"),('Freq','$neg_vocab_freq')]) 
        hover_2 = HoverTool(tooltips=[('Word', "$pos_vocab_words"),('Freq','$pos_vocab_freq')]) 
        
        p1 = figure(x_range=neg_vocab_words,plot_height=300,plot_width=750,
                    tools=[hover_1],title="Appropriate Question {} Frequency".format(title))
        p2 = figure(x_range=pos_vocab_words,plot_height=300,plot_width=750,
                    tools=[hover_2],title="Inappropriate Question {} Frequency".format(title))

        p1.vbar(x='neg_vocab_words',top='neg_vocab_freq',width=0.9,source=source)
        p2.vbar(x='pos_vocab_words',top='pos_vocab_freq',width=0.9,source=source)

        p1.xgrid.grid_line_color = None
        p2.xgrid.grid_line_color = None

        p1.xaxis.major_label_orientation = 1 
        p2.xaxis.major_label_orientation = 1 

        show(p1)
        show(p2)

    def word_count_difference_graph(self,unigram=True,num=10):
        """
        PURPOSE:
        """
        from bokeh.plotting import figure
        from bokeh.io import show

        if unigram: 
            _,gram_diff = self._unigram_frequencies_()
            title='Unigram'
        else:
            _,gram_diff = self._bigram_frequencies_()
            title='Bigram'
        
        words,diff = zip(*(gram_diff[:num] + gram_diff[-num:]))

        p = figure(x_range = words,plot_height=300,plot_width=750,
                   title='{} Frequency Difference: (Appropriate Frequency - Inappropriate Frequency)'.format(title))

        p.vbar(x=words,top=diff,width=0.9)
        p.xgrid.grid_line_color = None
        p.xaxis.major_label_orientation = 1 

        show(p)
        
    def comment_length_graph(self):
        """ 
        PURPOSE: 
        """
        from bokeh.plotting import figure
        from bokeh.io import show
        from bokeh.models import HoverTool, ColumnDataSource

        neg_len, neg_len_counts = zip(*self.sorted_comment_lengths[0])
        pos_len, pos_len_counts = zip(*self.sorted_comment_lengths[1])
        max_length = np.max((neg_len[-1],pos_len[-1]))

        neg_len_freq_plotting = [ neg_len_counts[neg_len.index(i)]/sum(neg_len_counts)
                                    if i in neg_len else 0
                                    for i in range(max_length+1)] 
        pos_len_freq_plotting = [ pos_len_counts[pos_len.index(i)]/sum(pos_len_counts)
                                    if i in pos_len else 0
                                    for i in range(max_length+1)]
        data = {'lengths' : list(range(max_length+1)),
                'neg': neg_len_freq_plotting,
                'pos': pos_len_freq_plotting}

        colors = ["#c9d9d3", "#718dbf"]

        p = figure(plot_width=750, plot_height=400, title='Distribution of Question Lengths')

        p.vbar(x='lengths',top='pos',width=0.9,source=data,color=colors[1],alpha=0.75,legend='Inappropriate')
        p.vbar(x='lengths',top='neg',width=0.9,source=data,color=colors[0],alpha=0.75,legend='Appropriate')
        
        show(p)

