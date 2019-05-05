import chainer
import chainer.functions as F
from chainer import links as L
from chainercv.links import Conv2DBNActiv


class TrainChain(chainer.Chain):
    def __init__(self, model):
        super(TrainChain, self).__init__()
        with self.init_scope():
            self.model = model

    def __call__(self, imgs, labels):
        predicted_labels = self.model(imgs)
        loss = F.softmax_cross_entropy(predicted_labels, labels)
        chainer.reporter.report({'loss': loss}, self)
        return loss


class PSPTrainChain(chainer.Chain):
    def __init__(self, model):
        initial = chainer.initializers.HeNormal()
        super(PSPTrainChain, self).__init__()
        with self.init_scope():
            self.model = model
            self.aux_conv1 = Conv2DBNActiv(
                None, 512, 3, 1, 1, initialW=initial)
            self.aux_conv2 = L.Convolution2D(
                None, model.n_class, 3, 1, 1, False, initialW=initial)

    def __call__(self, imgs, labels):
        h_aux, h_main = self.model.extractor(imgs)
        h_aux = F.dropout(self.aux_conv1(h_aux), ratio=0.1)
        h_aux = self.aux_conv2(h_aux)
        h_aux = F.resize_images(h_aux, imgs.shape[2:])

        h_main = self.model.ppm(h_main)
        h_main = F.dropout(self.model.head_conv1(h_main), ratio=0.1)
        h_main = self.model.head_conv2(h_main)
        h_main = F.resize_images(h_main, imgs.shape[2:])

        aux_loss = F.softmax_cross_entropy(h_aux, labels)
        main_loss = F.softmax_cross_entropy(h_main, labels)
        loss = 0.4 * aux_loss + main_loss

        chainer.reporter.report({'loss': loss}, self)
        return loss
