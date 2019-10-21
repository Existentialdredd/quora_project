import re

def canonicalize_digits(word):
    if any([c.isalpha() for c in word]): return word
    word = re.sub("\d", "DG", word)
    if word.startswith("DG"):
        word = word.replace(",", "") # remove thousands separator
    return word

def canonicalize_word(word, wordset=None, digits=True):
    word = word.lower()
    if digits:
        if (wordset != None) and (word in wordset): return word
        word = canonicalize_digits(word) # try to canonicalize numbers
    if (wordset == None) or (word in wordset):
        return word
    else:
        return constants.UNK_TOKEN

def canonicalize_words(words, **kw):
    return [canonicalize_word(word, **kw) for word in words]

def canonicalize_sentence(sentence, **kw): 
    sentence = sentence.lower()
    sentence = re.sub("(?<=\S)[\.\/\?\(\)\[-](?=\S)"," ",sentence)
    sentence = re.sub("(?<=\S)[\.\/\?\(\)\[](?=\s)","",sentence)
    sentence = re.sub("(?<=\s)[\.\/\?\(\)\[](?=\S)","",sentence)
    sentence = re.sub("(?<=\s)[\.\/\?\(\)\[](?=\s)","",sentence)
    sentence = re.sub("['\.\/\?\(\)\[](?=$)","",sentence)
    sentence = re.sub("[^a-zA-Z\d\s]","",sentence)
    sentence = re.sub("\s{2,}"," ",sentence)
    return sentence

def canon_token_sentence(sentence,**kw):
    canon_sentence = canonicalize_sentence(sentence)
    canon_token = canonicalize_words(canon_sentence.split(' '))
    return canon_token

def flatten_sort_listx2_tuple(tpl_list,sort_index=0,ascending='TRUE'):
    tpl_list = [ item for sublist in tpl_list for item in sublist] 
    if ascending: 
        tpl_list = sorted(tpl_list,key = lambda x: -x[sort_index])            
    else: 
        tpl_list = sorted(tpl_list,key = lambda x: x[sort_index])            
    return tpl_list

