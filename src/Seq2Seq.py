import numpy as np
from chainer import Variable, Chain
from chainer import functions as F
from chainer import links     as L
from chainer import reporter  as R



class Encoder(Chain):

    def __init__(self, size_vocab, size_hidden=300, size_middle=300, rate_dropout=0.1):
        super(Encoder, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(size_vocab, size_hidden)
            self.n_gru = L.NStepBiGRU(1, size_hidden, size_middle, rate_dropout)

    def forward(self, xs):
        es = [self.embed(x) for x in xs]
        h, _ = self.n_gru(None, es)

        return h


class Decoder(Chain):

    def __init__(self, size_vocab, size_hidden=300, size_middle=300, rate_dropout=0.1):
        super(Decoder, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(size_vocab, size_hidden)
            self.n_gru = L.NStepBiGRU(1, size_hidden, size_middle, rate_dropout)
            self.rev_embed = L.Linear(2*size_middle, size_vocab)

    def forward(self, xs, h):
        es = [self.embed(x) for x in xs]
        h, ys = self.n_gru(h, es)
        ys = [self.rev_embed(y) for y in ys]

        return h, ys


class Seq2Seq(Chain):

    def __init__(self, size_vocab, size_hidden=300, rate_dropout=0.1):
        super(Seq2Seq, self).__init__()
        with self.init_scope():
            self.encoder = Encoder(size_vocab, size_hidden, size_hidden, rate_dropout)
            self.decoder = Decoder(size_vocab, size_hidden, size_hidden, rate_dropout)

    def forward(self, enc_xs, dec_xs):
        h     = self.encoder(enc_xs)
        h, ys = self.decoder(dec_xs, h)

        return ys


class Seq2SeqLM(Chain):

    def __init__(self, network):
        super(Seq2SeqLM, self).__init__(net=network)


    def __call__(self, enc_xs, dec_xs, ts):
        loss = self.loss(enc_xs, dec_xs, ts)
        accu = self.accu(enc_xs, dec_xs, ts)

        return loss


    def loss(self, enc_xs, dec_xs, ts):
        # predict
        ys = self.net(enc_xs, dec_xs)
        # loss
        ys = F.concat(ys, axis=0)
        ts = F.concat(ts, axis=0)
        loss = F.sum(F.softmax_cross_entropy(ys, ts, reduce='no')) / len(ts)#バッチサイズ
        # reporting
        R.report({'loss':loss}, self)

        return loss / len(ts) #バッチサイズで割る


    def accu(self, enc_xs, dec_xs, ts):
        # predict
        ys = self.net(enc_xs, dec_xs)
        # accuracy
        ys = F.concat(ys, axis=0)
        ts = F.concat(ts, axis=0)
        accu = F.sum(F.accuracy(ys, ts))
        # reporting
        R.report({'accu':accu}, self)

        return accu


    def converter(self, batch, device=None):
        # in  : タプルのリスト : [(e0, d0, t0), (e1, d1, t1), (e2, d2, t2)]
        # out : リストのタプル : ([e0, e1, e2], [d0, d1, d2], [t0, d1, t3])
        return zip(*batch)
