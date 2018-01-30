import argparse

# General model args
parser = argparse.ArgumentParser(description='PyTorch MSDNet implementation')
parser.add_argument('--model', default='msdnet', type=str, metavar='M',
                    help='Model to train the dataset',
                    choices=['msdnet'])
parser.add_argument('--growth', type=str, metavar='GROWTH RATE',
                    help='Per layer growth')
parser.add_argument('--reduction', default=0.5, type=float, metavar='R',
                    help='Transition reduction (default: 0.5)')
parser.add_argument('--savedir', type=str, metavar='PATH', default='results/savedir',
                    help='Path to save result and checkpoint (default: results/savedir)')
parser.add_argument('--resume', action='store_true',
                    help='Use latest checkpoint if have any (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='Use pre-trained model (default: false)')
parser.add_argument('--no-save-model', dest='no_save_model', action='store_true',
                    help='Only save best model (default: false)')
parser.add_argument('--evaluate', action='store_true',
                    help='Evaluate model on validation set (default: false)')
parser.add_argument('--convert-from', default=None, type=str, metavar='PATH',
                    help='Path to saved checkpoint (default: none)')
parser.add_argument('--evaluate-from', default=None, type=str, metavar='PATH',
                    help='Path to saved checkpoint (default: none)')

# General args
parser.add_argument('data', metavar='DIR',
                    help='Path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='Number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='Number of total epochs to run (default: 300)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='Manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate (default: 0.1)')
parser.add_argument('--lr-type', default='cosine', type=str, metavar='T',
                    help='Learning rate strategy (default: cosine)',
                    choices=['cosine', 'multistep'])
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='Momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--manual-seed', default=0, type=int, metavar='N',
                    help='Manual seed (default: 0)')
parser.add_argument('--gpu', help='gpu available')
parser.add_argument('--debug', action='store_true', help='enable debugging')

# MSDNet args
parser.add_argument('--msd-base', type=int, metavar='B', default=4,
                    help='The layer to attach the first classifier (default: 4)')
parser.add_argument('--msd-blocks', type=int, metavar='nB', default=1,
                    help='Number of blocks/classifiers (default: 1)')
parser.add_argument('--msd-stepmode', type=str, metavar='nB', default='even',
                    help='Pattern of span between two adjacent classifers'
                         ' [even|lin_grow] (default: even)')
parser.add_argument('--msd-step', type=int, metavar='S', default=1,
                    help='Span between two adjacent classifers (default: 1)')
parser.add_argument('--msd-bottleneck', default=True, action='store_true',
                    help='Use 1x1 conv layer or not (default: True)')
parser.add_argument('--msd-bottleneck-factor', type=str,
                    metavar='bottleneck rate factor of each sacle',
                    help='Per scale bottleneck', default='1-2-4-4')
parser.add_argument('--msd-growth', type=int, metavar='GROWTH RATE',
                    help='Per layer growth', default=6)
parser.add_argument('--msd-growth-factor', type=str,
                    metavar='growth factor of each sacle',
                    help='Per scale growth', default='1-2-4-4')
parser.add_argument('--msd-prune', type=str,
                    help='Specify how to prune the network', default='max')
parser.add_argument('--msd-join-type', type=str,
                    help='Add or concat for features from different paths',
                    default='concat')
parser.add_argument('--msd-all-gcn', action='store_true',
                    help='Use GCN blocks for all MSDNet layers')
parser.add_argument('--msd-gcn', action='store_true',
                    help='Use GCN block for the first MSDNet layer')
parser.add_argument('--msd-share-weights', action='store_true',
                    help='Use GCN blocks for MSDNet')
parser.add_argument('--msd-gcn-kernel', type=int, metavar='KERNEL_SIZE',
                    help='GCN Conv2d kernel size', default=7)
parser.add_argument('--msd-kernel', type=int, metavar='KERNEL_SIZE',
                    help='MSD Conv2d kernel size', default=3)

# Init Environment
args = parser.parse_args()
args.growth = list(map(int, args.growth.split('-')))
args.msd_bottleneck_factor = \
    list(map(int, args.msd_bottleneck_factor.split('-')))
args.msd_growth_factor = \
    list(map(int, args.msd_growth_factor.split('-')))

# Num of classes
if args.data == 'cifar10':
    args.num_classes = 10
elif args.data == 'cifar100':
    args.num_classes = 100
else:
    args.num_classes = 1000