import argparse

from chainercv.extensions import SemanticSegmentationEvaluator
from chainercv.datasets import voc_semantic_segmentation_label_names

from classifier import TrainChain
from chainercv.experimental.links import PSPNetResNet50
from utils import create_iterator, create_model, create_trainer, trainer_extend


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default='models/unet.py')
    parser.add_argument('--model_name', type=str, default='Unet')
    parser.add_argument('--gpu_id', type=int, default=-1)

    # Train settings
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--epoch_or_iter', type=str, default='epoch')
    parser.add_argument('--num_epochs_or_iter', type=int, default=500)
    parser.add_argument('--initial_lr', type=float, default=0.05)
    parser.add_argument('--lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--lr_decay_epoch', type=float, default=25)
    parser.add_argument('--freeze_layer', type=str, default=None)
    parser.add_argument('--small_lr_layers', type=str, default=None)
    parser.add_argument('--small_initial_lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0.0005)

    # Save and Load settings
    parser.add_argument('--save_dir', type=str, default='result')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--save_trainer_interval', type=int, default=10)

    # Data augmentation settings
    parser.add_argument('--crop_size', type=int, nargs='*', default=[28, 28])
    parser.add_argument('--scale_range', type=int, nargs='*', default=[0.5, 2.0])
    parser.add_argument('--rotate', type=bool, default=True)
    parser.add_argument('--horizontal_flip', type=bool, default=True)
    args = parser.parse_args()

    label_names = voc_semantic_segmentation_label_names
    n_class = len(label_names)
    train_iter, valid_iter = create_iterator(args.crop_size,
                                             args.rotate,
                                             args.horizontal_flip,
                                             args.scale_range,
                                             args.batchsize)

    in_ch = 3
    # model = create_model(args, in_ch, n_class, args.crop_size)
    model = PSPNetResNet50(n_class, input_size=args.crop_size)
    net = TrainChain(model)

    evaluator = SemanticSegmentationEvaluator(valid_iter, model, label_names)

    trainer = create_trainer(train_iter, net, args.gpu_id, args.initial_lr,
                             args.weight_decay, args.freeze_layer, args.small_lr_layers,
                             args.small_initial_lr, args.num_epochs_or_iter,
                             args.epoch_or_iter, args.save_dir)

    trainer_extend(trainer, net, evaluator, args.small_lr_layers,
                   args.lr_decay_rate, args.lr_decay_epoch,
                   args.epoch_or_iter, args.save_trainer_interval)
    trainer.run()
