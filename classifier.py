import chainer
import chainer.functions as F


class TrainChain(chainer.Chain):

    def __init__(self, model):
        initialW = chainer.initializers.HeNormal()
        super(TrainChain, self).__init__()
        with self.init_scope():
            self.model = model

    def __call__(self, imgs, labels):
        predicted_labels = self.model(imgs)
        loss = F.softmax_cross_entropy(predicted_labels, labels)
        chainer.reporter.report({'loss': loss}, self)
        return loss
