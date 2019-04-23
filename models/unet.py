import collections

import chainer
from chainer import initializers
from chainer import links as L
from chainer import functions as F


class CBR(chainer.Chain):
    def __init__(self, ch0, ch1, bn=True, sample='down', activation=F.relu, dropout=False):
        super(CBR, self).__init__()
        w = initializers.normal.HeNormal(scale=1.0)
        self.activation = activation
        self.dropout = dropout
        with self.init_scope():
            if sample == 'down':
                self.conv = L.Convolution2D(ch0, ch1, 4, 2, 1, initialW=w)
            else:
                self.conv = L.Deconvolution2D(ch0, ch1, 4, 2, 1, initialW=w)
            if bn:
                self.bn = L.BatchNormalization(ch1)

    def __call__(self, x):
        h = self.conv(x)
        if self.bn:
            h = self.bn(h)
        if self.dropout:
            h = F.dropout(h)
        h = self.activation(h)
        return h


class Encoder(chainer.Chain):
    def __init__(self, in_ch):
        super(Encoder, self).__init__()
        w = initializers.normal.HeNormal(scale=1.0)
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_ch, 64, 3, 1, 1, initialW=w)
            self.conv2 = CBR(64, 128, sample='down', activation=F.leaky_relu, dropout=False)
            self.conv3 = CBR(128, 256, sample='down', activation=F.leaky_relu, dropout=False)
            self.conv4 = CBR(256, 512, sample='down', activation=F.leaky_relu, dropout=False)
            self.conv5 = CBR(512, 512, sample='down', activation=F.leaky_relu, dropout=False)
            self.conv6 = CBR(512, 512, sample='down', activation=F.leaky_relu, dropout=False)
            self.conv7 = CBR(512, 512, sample='down', activation=F.leaky_relu, dropout=False)
            self.conv8 = CBR(512, 512, sample='down', activation=F.leaky_relu, dropout=False)

    @property
    def functions(self):
        return collections.OrderedDict([
            ('pair1', [self.conv1, F.leaky_relu]),
            ('pair2', [self.conv2]),
            ('pair3', [self.conv3]),
            ('pair4', [self.conv4]),
            ('pair5', [self.conv5]),
            ('pair6', [self.conv6]),
            ('pair7', [self.conv7]),
            ('conv8', [self.conv8])
        ])

    def __call__(self, h):
        maps = {}
        for key, funcs in self.functions.items():
            for func in funcs:
                h = func(h)
            maps[key] = h
        return maps


class Decoder(chainer.Chain):
    def __init__(self, out_ch):
        super(Decoder, self).__init__()
        w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            self.conv1 = CBR(512, 512, sample='up', activation=F.relu, dropout=True)
            self.conv2 = CBR(1024, 512, sample='up', activation=F.relu, dropout=True)
            self.conv3 = CBR(1024, 512, sample='up', activation=F.relu, dropout=True)
            self.conv4 = CBR(1024, 512, sample='up', activation=F.relu, dropout=False)
            self.conv5 = CBR(1024, 256, sample='up', activation=F.relu, dropout=False)
            self.conv6 = CBR(512, 128, sample='up', activation=F.relu, dropout=False)
            self.conv7 = CBR(256, 64, sample='up', activation=F.relu, dropout=False)
            self.conv8 = L.Convolution2D(128, out_ch, 3, 1, 1, initialW=w)

    @property
    def functions(self):
        return collections.OrderedDict([
            ('pair7', [self.conv2]),
            ('pair6', [self.conv3]),
            ('pair5', [self.conv4]),
            ('pair4', [self.conv5]),
            ('pair3', [self.conv6]),
            ('pair2', [self.conv7]),
            ('pair1', [self.conv8])
        ])

    def __call__(self, hs):
        h = self.conv1(hs['conv8'])
        for key, funcs in self.functions.items():
            from IPython.core.debugger import Pdb; Pdb().set_trace()

            h = F.concat([h, hs[key]])
            for func in funcs:
                h = func(h)
        return h


class Unet(chainer.Chain):
    def __init__(self, in_ch=3, out_ch=1):
        super(Unet, self).__init__()
        with self.init_scope():
            self.encoder = Encoder(in_ch)
            self.decoder = Decoder(out_ch)

    def __call__(self, x):
        h = self.encoder(x)
        h = self.decoder(h)
        return h
