import numpy as np
import pickle
import nltk
from gensim.models import KeyedVectors



# param
#data_path = '../data/shakespeare.txt'
data_path = '../data/easy_sample.txt'
n_data = 7
min_sent_size = 1
max_sent_size = 5

# データの読み込み
with open(data_path) as f:
    data = []
    while len(data) < n_data:
        sent = nltk.word_tokenize(f.readline())
        if min_sent_size <= len(sent) and\
           max_sent_size >= len(sent) :
            data.append(sent)

# word -> id の辞書
word_id = {'<UNK>':0, '<END>':1}
for d in data:
    for w in d:
        if w not in word_id: word_id[w] = len(word_id)

# id -> word のリスト
id_word = []
for k in word_id.keys(): id_word.append(k)

#
data = [[word_id[w] for w in s] for s in data]

#
enc_xs = [np.array(    s    ) for s in data]
dec_xs = [np.array([1]+s    ) for s in data]
ts     = [np.array(    s+[1]) for s in data]

#
data  = list(zip(enc_xs, dec_xs, ts))
vocab = (word_id, id_word)

# 保存
with open('../pickle/data.pickle' , 'wb') as f: pickle.dump(data , f)
with open('../pickle/vocab.pickle', 'wb') as f: pickle.dump(vocab, f)
