import argparse

from chainercv.extensions import SemanticSegmentationEvaluator
from chainercv.datasets.voc.voc_semantic_segmentation_dataset \
    import VOCSemanticSegmentationDataset
from chainercv.datasets import voc_semantic_segmentation_label_names
from chainercv.datasets import ADE20KSemanticSegmentationDataset
from chainercv.datasets import ade20k_semantic_segmentation_label_names
from chainercv.datasets import CityscapesSemanticSegmentationDataset
from chainercv.datasets import cityscapes_semantic_segmentation_label_names

from classifier import TrainChain, PSPTrainChain
from chainercv.experimental.links import PSPNetResNet50
from utils import create_iterator, create_model, create_trainer, trainer_extend


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default='models/unet.py')
    parser.add_argument('--model_name', type=str, default='Unet')
    parser.add_argument('--data_name', type=str, default='VOC')
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
    parser.add_argument('--data_dir', type=str, default='auto')

    # Data augmentation settings
    parser.add_argument('--crop_size', type=int, nargs='*', default=[28, 28])
    parser.add_argument('--scale_range', type=int, nargs='*', default=[0.5, 2.0])
    parser.add_argument('--rotate', type=bool, default=True)
    parser.add_argument('--horizontal_flip', type=bool, default=True)
    args = parser.parse_args()

    if args.data_name == 'VOC':
        train = VOCSemanticSegmentationDataset(
            data_dir=args.data_dir, split='train')
        valid = VOCSemanticSegmentationDataset(
            data_dir=args.data_dir, split='val')
        label_names = voc_semantic_segmentation_label_names
    elif args.data_name == 'ADE':
        train = ADE20KSemanticSegmentationDataset(
            data_dir=args.data_dir, split='train')
        valid = ADE20KSemanticSegmentationDataset(
            data_dir=args.data_dir, split='val')
        label_names = ade20k_semantic_segmentation_label_names
    elif args.data_name == 'Cityscapes':
        train = CityscapesSemanticSegmentationDataset(
            args.data_dir,
            label_resolution='fine', split='train')
        valid = CityscapesSemanticSegmentationDataset(
            args.data_dir,
            label_resolution='fine', split='val')
        label_names = cityscapes_semantic_segmentation_label_names
    else:
        raise ValueError('Invalid model_name')
    n_class = len(label_names)
    train_iter, valid_iter = create_iterator(train, valid,
                                             args.crop_size,
                                             args.rotate,
                                             args.horizontal_flip,
                                             args.scale_range,
                                             args.batchsize)

    in_ch = 3
    # model = create_model(args, in_ch, n_class, args.crop_size)
    model = PSPNetResNet50(n_class, input_size=args.crop_size, pretrained_model='imagenet')
    net = PSPTrainChain(model)

    evaluator = SemanticSegmentationEvaluator(valid_iter, model, label_names)

    trainer = create_trainer(train_iter, net, args.gpu_id, args.initial_lr,
                             args.weight_decay, args.freeze_layer, args.small_lr_layers,
                             args.small_initial_lr, args.num_epochs_or_iter,
                             args.epoch_or_iter, args.save_dir)
    # TODO: https://github.com/chainer/chainercv/blob/fd630326cb148c8a4966a65ccbdaea90900cd8de/examples/pspnet/train_multi.py#L256 

    trainer_extend(trainer, net, evaluator, args.small_lr_layers,
                   args.lr_decay_rate, args.lr_decay_epoch,
                   args.epoch_or_iter, args.save_trainer_interval)
    trainer.run()
