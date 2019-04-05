import pickle
import numpy as np
import chainer
import chainer.functions as F
import chainer.iterators as I
import chainer.optimizers as O
import chainer.serializers as S
import chainer.training  as T
import chainer.training.extensions as E
import datetime
from Seq2Seq import *
from VRAE    import *


# Parameter
size_batch = 2
size_hidden = 32
size_middle = 1
rate_dropout = 0.1

out_dir = '../result/easy_sample'

# Loading
with open('../pickle/data.pickle' , 'rb') as f:  data = pickle.load(f)
with open('../pickle/vocab.pickle', 'rb') as f:  word_id, id_word = pickle.load(f)
n_data = len(data)
size_vocab = len(id_word)
print('data  :', n_data)
print('vocab :', size_vocab)

# Setup
net = VRAE(size_vocab, size_hidden, size_middle, rate_dropout)
model = VRAELM(net)

path = out_dir + '/' + 'model_epoch-26400'
S.load_npz(path, model)

#
with chainer.using_config('train', False):
    hs = []
    ls = []
    for enc_x, dec_x, t in data:
        h = model.encode([enc_x])
        hs.append(F.hstack(h).data[0])

        l = enc_x
        ls.append(l)

    print(len(hs))
    print(len(ls))

with open('../pickle/hidden.pickle', 'wb') as f: pickle.dump((hs, ls), f)
