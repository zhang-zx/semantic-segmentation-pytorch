# System libs
import os
import time
# import math
import random
import argparse
from distutils.version import LooseVersion
# Numerical libs
import torch
import torch.nn as nn
# Our libs
from dataset import TrainDataset
from models import ModelBuilder, SegmentationModule
from utils import AverageMeter, parse_devices
from lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback
import lib.utils.data as torchdata


# train one epoch
def train(segmentation_module, iterator, optimizers, history, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()

    segmentation_module.train(not args.fix_bn)

    # main loop
    tic = time.time()
    for i in range(args.epoch_iters):
        batch_data = next(iterator)
        data_time.update(time.time() - tic)

        segmentation_module.zero_grad()

        # forward pass
        loss, acc = segmentation_module(batch_data)
        loss = loss.mean()
        acc = acc.mean()

        # Backward
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_total_loss.update(loss.data.item())
        ave_acc.update(acc.data.item() * 100)

        # calculate accuracy, and display
        if i % args.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_unet: {:.6f}, '
                  'Accuracy: {:4.2f}, Loss: {:.6f}'
                  .format(epoch, i, args.epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.running_lr_unet,
                          ave_acc.average(), ave_total_loss.average()))

            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(loss.data.item())
            history['train']['acc'].append(acc.data.item())

        # adjust learning rate
        cur_iter = i + (epoch - 1) * args.epoch_iters
        adjust_learning_rate(optimizers, cur_iter, args)


def checkpoint(nets, history, args, epoch_num):
    print('Saving checkpoints...')
    (net_unet, crit) = nets
    suffix_latest = 'epoch_{}.pth'.format(epoch_num)

    dict_unet = net_unet.state_dict()

    torch.save(history,
               '{}/history_{}'.format(args.ckpt, suffix_latest))
    torch.save(dict_unet,
               '{}/unet_{}'.format(args.ckpt, suffix_latest))


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def create_optimizers(nets, args):
    (net_unet, crit) = nets
    optimizer_unet = torch.optim.SGD(
        group_weight(net_unet),
        lr=args.lr_unet,
        momentum=args.beta1,
        weight_decay=args.weight_decay)
    return optimizer_unet


def adjust_learning_rate(optimizers, cur_iter, args):
    scale_running_lr = ((1. - float(cur_iter) / args.max_iters) ** args.lr_pow)
    args.running_lr_unet = args.lr_unet * scale_running_lr

    optimizer_unet = optimizers
    for param_group in optimizer_unet.param_groups:
        param_group['lr'] = args.running_lr_unet



def main(args):
    # Network Builders
    builder = ModelBuilder()
    unet = builder.build_unet(n_channels=3, n_classes=args.num_class)

    crit = nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(unet, crit)

    # Dataset and Loader
    dataset_train = TrainDataset(
        args.list_train, args, batch_per_gpu=args.batch_size_per_gpu)

    loader_train = torchdata.DataLoader(
        dataset_train,
        batch_size=len(args.gpus),  # we have modified data_parallel
        shuffle=False,  # we do not use this param
        collate_fn=user_scattered_collate,
        num_workers=int(args.workers),
        drop_last=True,
        pin_memory=True)

    print('1 Epoch = {} iters'.format(args.epoch_iters))

    # create loader iterator
    iterator_train = iter(loader_train)

    # load nets into gpu
    if len(args.gpus) > 1:
        segmentation_module = UserScatteredDataParallel(
            segmentation_module,
            device_ids=args.gpus)
        # For sync bn
        patch_replication_callback(segmentation_module)
    segmentation_module.cuda()

    # Set up optimizers
    nets = (unet, crit)
    optimizers = create_optimizers(nets, args)

    # Main loop
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}}

    for epoch in range(args.start_epoch, args.num_epoch + 1):
        train(segmentation_module, iterator_train, optimizers, history, epoch, args)

        # checkpointing
        checkpoint(nets, history, args, epoch)

    print('Training Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='baseline',
                        help="a name for identifying the model")
    # Path related arguments
    parser.add_argument('--list_train',
                        default='./data/train.odgt')
    parser.add_argument('--list_val',
                        default='./data/validation.odgt')
    parser.add_argument('--root_dataset',
                        default='./data/')

    # optimization related arguments
    parser.add_argument('--gpus', default='0-3',
                        help='gpus to use, e.g. 0-3 or 0,1,2,3')
    parser.add_argument('--batch_size_per_gpu', default=2, type=int,
                        help='input batch size')
    parser.add_argument('--num_epoch', default=20, type=int,
                        help='epochs to train for')
    parser.add_argument('--start_epoch', default=1, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--epoch_iters', default=5000, type=int,
                        help='iterations of each epoch (irrelevant to batch size)')
    parser.add_argument('--optim', default='SGD', help='optimizer')

    parser.add_argument('--lr_unet', default=2e-2, type=float, help='LR')

    parser.add_argument('--lr_pow', default=0.9, type=float,
                        help='power in poly to drop LR')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weights regularizer')
    parser.add_argument('--deep_sup_scale', default=0.4, type=float,
                        help='the weight of deep supervision loss')
    parser.add_argument('--fix_bn', action='store_true',
                        help='fix bn params')

    # Data related arguments
    parser.add_argument('--num_class', default=150, type=int,
                        help='number of classes')
    parser.add_argument('--workers', default=16, type=int,
                        help='number of data loading workers')
    parser.add_argument('--imgSize', default=[300, 375, 450, 525, 600],
                        nargs='+', type=int,
                        help='input image size of short edge (int or list)')
    parser.add_argument('--imgMaxSize', default=1000, type=int,
                        help='maximum input image size of long edge')
    parser.add_argument('--padding_constant', default=8, type=int,
                        help='maxmimum downsampling rate of the network')
    parser.add_argument('--segm_downsampling_rate', default=8, type=int,
                        help='downsampling rate of the segmentation label')
    parser.add_argument('--random_flip', default=True, type=bool,
                        help='if horizontally flip images when training')

    # Misc arguments
    parser.add_argument('--seed', default=304, type=int, help='manual seed')
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')
    parser.add_argument('--disp_iter', type=int, default=20,
                        help='frequency to display')

    args = parser.parse_args()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    # Parse gpu ids
    all_gpus = parse_devices(args.gpus)
    all_gpus = [x.replace('gpu', '') for x in all_gpus]
    args.gpus = [int(x) for x in all_gpus]
    num_gpus = len(args.gpus)
    args.batch_size = num_gpus * args.batch_size_per_gpu

    args.max_iters = args.epoch_iters * args.num_epoch

    args.running_lr_unet = args.lr_unet


    # Model ID
    args.id += 'unet'
    args.id += '-ngpus' + str(num_gpus)
    args.id += '-batchSize' + str(args.batch_size)
    args.id += '-imgMaxSize' + str(args.imgMaxSize)
    args.id += '-paddingConst' + str(args.padding_constant)
    args.id += '-segmDownsampleRate' + str(args.segm_downsampling_rate)
    args.id += '-LR_UNET' + str(args.lr_unet)
    args.id += '-epoch' + str(args.num_epoch)
    if args.fix_bn:
        args.id += '-fixBN'
    print('Model ID: {}'.format(args.id))

    args.ckpt = os.path.join(args.ckpt, args.id)
    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
