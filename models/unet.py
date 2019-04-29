import collections
from math import ceil

import chainer
from chainer import initializers
from chainer import links as L
from chainer import functions as F
from chainercv.experimental.links.model.pspnet.transforms import convolution_crop
from chainercv import transforms
import numpy as np


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

            h = F.concat([h, hs[key]])
            for func in funcs:
                h = func(h)
        return h


class Unet(chainer.Chain):
    def __init__(self, in_ch, out_ch, input_size):
        super(Unet, self).__init__()
        self.input_size = input_size
        self.n_class = out_ch
        self.mean = np.array((123.68, 116.779, 103.939), dtype=np.float32)[:, None, None]
        self.scales = None
        with self.init_scope():
            self.encoder = Encoder(in_ch)
            self.decoder = Decoder(out_ch)

    def __call__(self, x):
        h = self.encoder(x)
        h = self.decoder(h)
        return h

    def _tile_predict(self, img):
        if self.mean is not None:
            img = img - self.mean
        ori_H, ori_W = img.shape[1:]
        long_size = max(ori_H, ori_W)

        if long_size > max(self.input_size):
            stride_rate = 2 / 3
            stride = (int(ceil(self.input_size[0] * stride_rate)),
                      int(ceil(self.input_size[1] * stride_rate)))

            imgs, param = convolution_crop(
                img, self.input_size, stride, return_param=True)

            counts = self.xp.zeros((1, ori_H, ori_W), dtype=np.float32)
            preds = self.xp.zeros((1, self.n_class, ori_H, ori_W),
                                  dtype=np.float32)
            N = len(param['y_slices'])
            for i in range(N):
                img_i = imgs[i:i + 1]
                y_slice = param['y_slices'][i]
                x_slice = param['x_slices'][i]
                crop_y_slice = param['crop_y_slices'][i]
                crop_x_slice = param['crop_x_slices'][i]

                scores_i = self._predict(img_i)
                # Flip horizontally flipped score maps again
                flipped_scores_i = self._predict(
                    img_i[:, :, :, ::-1])[:, :, :, ::-1]

                preds[0, :, y_slice, x_slice] += \
                    scores_i[0, :, crop_y_slice, crop_x_slice]
                preds[0, :, y_slice, x_slice] += \
                    flipped_scores_i[0, :, crop_y_slice, crop_x_slice]
                counts[0, y_slice, x_slice] += 2

            scores = preds / counts[:, None]
        else:
            img, param = transforms.resize_contain(
                img, self.input_size, return_param=True)
            preds1 = self._predict(img[np.newaxis])
            preds2 = self._predict(img[np.newaxis, :, :, ::-1])
            preds = (preds1 + preds2[:, :, :, ::-1]) / 2

            y_start = param['y_offset']
            y_end = y_start + param['scaled_size'][0]
            x_start = param['x_offset']
            x_end = x_start + param['scaled_size'][1]
            scores = preds[:, :, y_start:y_end, x_start:x_end]
        scores = F.resize_images(scores, (ori_H, ori_W))[0].array
        return scores

    def _predict(self, imgs):
        xs = chainer.Variable(self.xp.asarray(imgs))
        with chainer.using_config('train', False):
            scores = F.softmax(self(xs)).array
        return scores

    def predict(self, imgs):
        """Conduct semantic segmentation from images.
        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in CHW and RGB format
                and the range of their values are :math:`[0, 255]`.
        Returns:
            list of numpy.ndarray:
            List of integer labels predicted from each image in the input \
            list.
        """
        labels = []
        for img in imgs:
            with chainer.using_config('train', False), \
                 chainer.function.no_backprop_mode():
                if self.scales is not None:
                    scores = _multiscale_predict(
                        self._tile_predict, img, self.scales)
                else:
                    scores = self._tile_predict(img)
            labels.append(chainer.backends.cuda.to_cpu(
                self.xp.argmax(scores, axis=0).astype(np.int32)))

        return labels


def _multiscale_predict(predict_method, img, scales):
    orig_H, orig_W = img.shape[1:]
    scores = []
    orig_img = img
    for scale in scales:
        img = orig_img.copy()
        if scale != 1.0:
            img = transforms.resize(
                img, (int(orig_H * scale), int(orig_W * scale)))
        # This method should return scores
        y = predict_method(img)[None]
        assert y.shape[2:] == img.shape[1:]

        if scale != 1.0:
            y = F.resize_images(y, (orig_H, orig_W)).array
        scores.append(y)
    xp = chainer.backends.cuda.get_array_module(scores[0])
    scores = xp.stack(scores)
    return scores.mean(0)[0]  # (C, H, W)
