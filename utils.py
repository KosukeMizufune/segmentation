import os
from importlib import import_module

from chainer import cuda, optimizers, training, iterators
from chainer.training import extensions, triggers
from chainer.optimizer_hooks import WeightDecay
from chainer.datasets import TransformDataset

import numpy as np

from transform import Transform


def create_iterator(train, valid, crop_size, rotate, horizontal_flip, scale_range, batchsize):

    mean = np.array(
        (123.68, 116.779, 103.939), dtype=np.float32)[:, None, None]
    processed_train = TransformDataset(train, Transform(mean,
                                                        crop_size,
                                                        rotate,
                                                        horizontal_flip,
                                                        scale_range))

    train_iter = iterators.SerialIterator(processed_train, batchsize)
    valid_iter = iterators.SerialIterator(valid, batchsize, repeat=False, shuffle=False)
    return train_iter, valid_iter


def create_model(model_file, model_name, in_ch, n_class, input_size):
    ext = os.path.splitext(model_file)[1]
    mod_path = '.'.join(os.path.split(model_file)).replace(ext, '')
    mod = import_module(mod_path)
    model = getattr(mod, model_name)(in_ch, n_class, input_size)
    return model


def create_trainer(train_iter, net, gpu_id, initial_lr, weight_decay, freeze_layer,
                   small_lr_layers, small_initial_lr, num_epochs_or_iter,
                   epoch_or_iter, save_dir):
    # Optimizer
    if gpu_id >= 0:
        net.to_gpu(gpu_id)
    optimizer = optimizers.MomentumSGD(lr=initial_lr)
    optimizer.setup(net)

    if weight_decay > 0:
        optimizer.add_hook(WeightDecay(weight_decay))
    if freeze_layer:
        freeze_setup(net, optimizer, freeze_layer)

    if small_lr_layers:
        for layer_name in small_lr_layers:
            layer = getattr(net.predictor, layer_name)
            layer.W.update_rule.hyperparam.lr = small_initial_lr
            layer.b.update_rule.hyperparam.lr = small_initial_lr

    # Trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)
    trainer = training.Trainer(
        updater, (num_epochs_or_iter, epoch_or_iter), out=save_dir)
    return trainer


def trainer_extend(trainer, net, evaluator, small_lr_layers, lr_decay_rate,
                   lr_decay_epoch, epoch_or_iter, save_trainer_interval):
    def slow_drop_lr(trainer):
        if small_lr_layers:
            for layer_name in small_lr_layers:
                layer = getattr(net.predictor, layer_name)
                layer.W.update_rule.hyperparam.lr *= lr_decay_rate
                layer.b.update_rule.hyperparam.lr *= lr_decay_rate

    # Learning rate
    trainer.extend(
        slow_drop_lr,
        trigger=triggers.ManualScheduleTrigger(lr_decay_epoch,
                                               epoch_or_iter)
    )
    trainer.extend(extensions.ExponentialShift('lr', lr_decay_rate),
                   trigger=triggers.ManualScheduleTrigger(lr_decay_epoch,
                                                          epoch_or_iter))

    # Observe training
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.observe_lr(), trigger=(1, epoch_or_iter))
    trainer.extend(evaluator, name='val')

    print_report = ["epoch",
                    "main/loss",
                    "val/main/miou",
                    "val/main/pixel_accuracy",
                    "val/main/mean_class_accuracy",
                    "lr",
                    "elapsed_time"]

    trainer.extend(extensions.PrintReport(print_report))

    # save results of training
    trainer.extend(extensions.PlotReport(['main/loss'],
                                         x_key=epoch_or_iter,
                                         file_name='loss.png'))
    trainer.extend(
        extensions.PlotReport(['val/main/miou',
                               'val/main/pixel_accuracy',
                               'val/main/mean_class_accuracy'],
                              x_key=epoch_or_iter,
                              file_name='accuracy.png'))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(
        filename="snapshot_epoch-" + '{.updater.epoch}'),
        trigger=(save_trainer_interval, epoch_or_iter))


class DelGradient(object):
    name = 'DelGradient'

    def __init__(self, deltgt):
        self.deltgt = deltgt

    def __call__(self, opt):
        for name, param in opt.target.namedparams():
            for d in self.deltgt:
                if d in name:
                    grad = param.grad
                    with cuda.get_device(grad):
                        grad = 0


def freeze_setup(net, optimizer, freeze_layer):
    if freeze_layer == 'all':
        net.predictor.base.disable_update()
    elif isinstance(freeze_layer, list):
        optimizer.add_hook(DelGradient(freeze_layer))
