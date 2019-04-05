import numpy as np
from chainer import Chain, Variable
from chainer import functions as F
from chainer import links     as L
from chainer import reporter  as R
from Seq2Seq import Encoder, Decoder
UNK = 0
EOS = 1


class VRAE(Chain):

    def __init__(self, size_vocab, size_hidden=300, size_middle=300, rate_dropout=0.1):
        super(VRAE, self).__init__()
        with self.init_scope():
            self.encoder = Encoder(size_vocab, size_hidden, size_middle, rate_dropout)
            self.h_mu = L.Highway(2*size_middle)
            self.h_sg = L.Highway(2*size_middle)
            self.decoder = Decoder(size_vocab, size_hidden, size_middle, rate_dropout)

    def forward(self, enc_xs, dec_xs):
        # encode
        h = self.encoder(enc_xs)
        # Reshape (2, batch, hidden) -> (batch, 2 * hidden)
        h = F.hstack(h)
        # 平均と標準偏差
        mu = self.h_mu(h)
        sg = self.h_sg(h)
        # Reshape (batch, 2 * hidden) -> (2, batch, hidden)
        mu = F.stack(F.split_axis(mu, 2, axis=1))
        # decode
        h, ys = self.decoder(dec_xs, mu)

        return ys


class VRAELM(Chain):

    def __init__(self, network):
        super(VRAELM, self).__init__(net=network)

    def forward(self, enc_xs, dec_xs, ts):
        loss = self.loss(enc_xs, dec_xs, ts)
        accu = self.accu(enc_xs, dec_xs, ts)

        return loss

    def loss(self, enc_xs, dec_xs, ts):
        # Gauusian Forward calculation
        #   Encode
        h = self.net.encoder(enc_xs)
        #   Reshape (2, batch, hidden) -> (batch, 2 * hidden)
        h = F.hstack(h)
        #   平均と標準偏差
        mu = self.net.h_mu(h)
        sg = self.net.h_sg(h)
        #   z = mu + sg * N(0,1)
        z = F.gaussian(mu, sg)
        #   Reshape (batch, 2 * hidden) -> (2, batch, hidden)
        z = F.stack(F.split_axis(z, 2, axis=1))
        #   Decode
        h, ys = self.net.decoder(dec_xs, z)

        # Loss Calculation
        #   Reconstraction Loss
        ys = F.concat(ys, axis=0)
        ts = F.concat(ts, axis=0)
        rec_loss = F.sum(F.softmax_cross_entropy(ys, ts, reduce='no'))
        #   Regularization Loss
        kld_loss = F.loss.vae.gaussian_kl_divergence(mu, sg)
        #   Entire Loss
        size_batch = len(ts)
        loss = (rec_loss + kld_loss) / size_batch

        # Reporting
        R.report({'loss':loss}, self)

        return loss

    def accu(self, enc_xs, dec_xs, ts):
        # Forward calculation
        ys = self.net(enc_xs, dec_xs)
        # Accyracy
        ys = F.concat(ys, axis=0)
        ts = F.concat(ts, axis=0)
        accu = F.sum(F.accuracy(ys, ts))
        # Reporting
        R.report({'accu':accu}, self)

        return accu

    def encode(self, xs):
        h = self.net.encoder(xs)
        # Reshape (2, batch, hidden) -> (batch, 2 * hidden)
        h = F.hstack(h)
        # 平均と標準偏差
        mu = self.net.h_mu(h)
        # Reshape (batch, 2 * hidden) -> (2, batch, hidden)
        mu = F.stack(F.split_axis(mu, 2, axis=1))

        return mu

    # TODO: 同じ単語IDを生成し続ける
    def decode(self, h, max_length=10):
        size_batch = len(h[0])
        #
        xs = [EOS] * size_batch
        ys = []
        for _ in range(max_length):
            # 入力の型変換 int or np.int32 -> array()
            xs = [np.array([x]) for x in xs]
            # decode
            h, xs = self.net.decoder(h, xs)
            # indexに
            xs = [np.argmax(x.data[0]) for x in xs]
            #
            ys.append(xs)
        # 出力の整形
        ys = zip(*ys) #二重リストの転置
        ys = [y[:y.index(EOS)] if EOS in y else y for y in ys] #EOSでカット

        return ys

    def converter(self, batch, device=None):
        # in  : タプルのリスト : [(e0, d0, t0), (e1, d1, t1), (e2, d2, t2)]
        # out : リストのタプル : ([e0, e1, e2], [d0, d1, d2], [t0, d1, t3])
        ex, dx, ts = zip(*batch)
        return (ex, dx, ts)
