import chainer
from chainer import cuda, optimizers, training
from chainer.training import extensions, triggers
from chainer.optimizer_hooks import WeightDecay


def run_train(train_iter, net, evaluator, **kwargs):
    # Optimizer
    if kwargs['gpu_id'] >= 0:
        net.to_gpu(kwargs['gpu_id'])
    optimizer = optimizers.MomentumSGD(lr=kwargs['lr'])
    optimizer.setup(net)

    if kwargs['l2_lambda'] > 0:
        optimizer.add_hook(WeightDecay(kwargs['l2_lambda']))
    freeze_setup(net, optimizer, kwargs['freeze_layer'])

    if kwargs['changed_lr_layer']:
        for layer in kwargs['changed_lr_layer']:
            layer.update_rule.hyperparam.lr = kwargs['changed_lr']

    # Trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=kwargs['gpu_id'])
    trainer = training.Trainer(
        updater, (kwargs['max_epoch'], 'epoch'), out=kwargs['result_dir'])

    if kwargs['load_dir']:
        chainer.serializers.load_npz(kwargs['load_dir'], trainer)
    trainer_extend(trainer,
                   evaluator,
                   **kwargs)
    trainer.run()


def trainer_extend(trainer, evaluator, **kwargs):
    def slow_drop_lr(trainer):
        if kwargs['changed_lr_layer'] is None:
            pass
        else:
            for layer in kwargs['changed_lr_layer']:
                layer.update_rule.hyperparam.lr *= kwargs['lr_drop_rate']

    # Learning rate
    trainer.extend(
        slow_drop_lr,
        trigger=triggers.ManualScheduleTrigger(kwargs['lr_drop_epoch'], kwargs['unit'])
    )
    trainer.extend(extensions.ExponentialShift('lr', kwargs['lr_drop_rate']),
                   trigger=triggers.ManualScheduleTrigger(kwargs['lr_drop_epoch'], kwargs['unit']))

    # Observe training
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.observe_lr(), trigger=(1, kwargs['unit']))
    trainer.extend(evaluator, name='val')
    trainer.extend(extensions.PrintReport(kwargs['print_report']))

    # save results of training
    trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'],
                                         x_key=kwargs['unit'],
                                         file_name='loss.png'))
    trainer.extend(
        extensions.PlotReport(['val/main/miou',
                               'val/main/pixel_accuracy',
                               'val/main/mean_class_accuracy'],
                              x_key=kwargs['unit'], file_name='accuracy.png'))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(filename=kwargs['snapshot_filename']+'{.updater.epoch}'),
                   trigger=(kwargs['save_trainer_interval'], kwargs['unit']))


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
