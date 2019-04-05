import numpy as np
import chainer
import chainer.functions   as F
import chainer.iterators   as I
import chainer.optimizers  as O
import chainer.serializers as S
import yaml, pickle
import os, shutil
from vrae import *


''' ファイルの読み込み '''
# 設定ファイル
with open('./config.yml', 'r+') as f:  config = yaml.load(f)
# 加工済みデータファイル
with open('../pickle/data.pickle' , 'rb') as f:  data = pickle.load(f)
with open('../pickle/vocab.pickle', 'rb') as f:  word_id, id_word = pickle.load(f)


''' 学習の設定 '''
n_data     = len(data)
max_epoch  = config['max_epoch']
size_batch = config['size_batch']

n_vocab     = len(id_word)
size_hidden = config['size_hidden']


net = RNNLM(Seq2Seq(n_vocab, size_hidden))
opt = O.Adam()
opt.setup(net)


''' 本丸 '''
for i in range(max_epoch):
    '''
    Update
     - バッチ毎に入力
    '''
    epoch_loss = 0
    for xs in I.SerialIterator(data, size_batch, repeat=False, shuffle=True):
        # loss の計算
        model(xs)
        loss = model.loss
        accu = model.accu
        # 更新
        net.cleargrads()
        loss.backward()
        opt.update()

        print('{:>3} | {:>10.5f} | {:>6.1%}'.format(i+1, loss.data, accu.data))
