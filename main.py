#!/usr/bin/python3
#
# Example run:
# ./main.py --model msdnet -b 2 -j 2 cifar10 --msd-blocks 10 --msd-base 4 --msd-step 2 \
#  --msd-stepmode even --growth 6-12-24 --gpu 0
# For evaluation / resume add: --resume --evaluate

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import os
import shutil
import time
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
from utils import measure_model
from opts import args


# Init Torch/Cuda
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
best_prec1 = 0


def main(**kwargs):

    global args, best_prec1

    # Override if needed
    for arg, v in kwargs.items():
        args.__setattr__(arg, v)

    ### Calculate FLOPs & Param
    model = getattr(models, args.model)(args)

    if args.data in ['cifar10', 'cifar100']:
        IMAGE_SIZE = 32
    else:
        IMAGE_SIZE = 224

    n_flops, n_params = measure_model(model, IMAGE_SIZE, IMAGE_SIZE, args.debug)

    if 'measure_only' in args and args.measure_only:
        return n_flops, n_params

    print('Starting.. FLOPs: %.2fM, Params: %.2fM' % (n_flops / 1e6, n_params / 1e6))
    args.filename = "%s_%s_%s.txt" % \
        (args.model, int(n_params), int(n_flops))
    del(model)

    # Create model
    model = getattr(models, args.model)(args)

    if args.debug:
        print(args)
        print(model)

    model = torch.nn.DataParallel(model).cuda()

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # Resume from a checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args)
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    # Evaluate from a model
    if args.evaluate_from is not None:
        args.evaluate = True
        state_dict = torch.load(args.evaluate_from)['state_dict']
        model.load_state_dict(state_dict)

    cudnn.benchmark = True

    ### Data loading
    if args.data == "cifar10":
        train_set = datasets.CIFAR10('../data', train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                     ]))
        val_set = datasets.CIFAR10('../data', train=False,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                   ]))
    elif args.data == "cifar100":
        train_set = datasets.CIFAR100('../data', train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                     ]))
        val_set = datasets.CIFAR100('../data', train=False,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                   ]))

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Run Forward / Backward passes
    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):

        # Train for one epoch
        tr_prec1, tr_prec5, loss, lr = \
            train(train_loader, model, criterion, optimizer, epoch)

        # Evaluate on validation set
        val_prec1, val_prec5 = validate(val_loader, model, criterion)

        # Remember best prec@1 and save checkpoint
        is_best = val_prec1 < best_prec1
        best_prec1 = max(val_prec1, best_prec1)
        model_filename = 'checkpoint_%03d.pth.tar' % epoch
        save_checkpoint({
            'epoch': epoch,
            'model': args.model,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, args, is_best, model_filename, "%.4f %.4f %.4f %.4f %.4f %.4f\n" %
            (val_prec1, val_prec5, tr_prec1, tr_prec5, loss, lr))

    # TestModel and return
    model = model.cpu().module
    model = nn.DataParallel(model).cuda()
    print(model)
    validate(val_loader, model, criterion)
    n_flops, n_params = measure_model(model, IMAGE_SIZE, IMAGE_SIZE)
    print('Finished training! FLOPs: %.2fM, Params: %.2fM' % (n_flops / 1e6, n_params / 1e6))
    print('Please run again with --resume --evaluate flags,'
          ' to evaluate the best model.')

    return


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    ### Switch to train mode
    model.train()

    running_lr = None

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        progress = float(epoch * len(train_loader) + i) / \
            (args.epochs * len(train_loader))
        args.progress = progress

        ### Adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                  nBatch=len(train_loader), method=args.lr_type)
        if running_lr is None:
            running_lr = lr

        ### Measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        ### Compute output
        output = model(input_var, progress)
        if args.model == 'msdnet':
            loss = msd_loss(output, target_var, criterion)
        else:
            loss = criterion(output, target_var)

        ### Measure accuracy and record loss
        if hasattr(output, 'data'):
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        elif args.model == 'msdnet':
            prec1, prec5, _ = msdnet_accuracy(output, target, input)
        else:
            raise NotImplementedError
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        ### Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f}\t'  # ({batch_time.avg:.3f}) '
                  'Data {data_time.val:.3f}\t'  # ({data_time.avg:.3f}) '
                  'Loss {loss.val:.4f}\t'  # ({loss.avg:.4f}) '
                  'Prec@1 {top1.val:.3f}\t'  # ({top1.avg:.3f}) '
                  'Prec@5 {top5.val:.3f}\t'  # ({top5.avg:.3f})'
                  'lr {lr: .4f}'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5, lr=lr))
    return 100. - top1.avg, 100. - top5.avg, losses.avg, running_lr


def msd_loss(output, target_var, criterion):
    losses = []
    for out in range(0, len(output)):
        losses.append(criterion(output[out], target_var))
    mean_loss = sum(losses) / len(output)
    return mean_loss


def msdnet_accuracy(output, target, x, val=False):
    """
    Calculates multi-classifier accuracy

    :param output: A list in the length of the number of classifiers,
                   including output tensors of size (batch, classes)
    :param target: a tensor of length batch_size, including GT
    :param x: network input input
    :param val: A flag to print per class validation accuracy
    :return: mean precision of top1 and top5
    """

    top1s = []
    top5s = []
    prec1 = torch.FloatTensor([0]).cuda()
    prec5 = torch.FloatTensor([0]).cuda()
    for out in output:
        tprec1, tprec5 = accuracy(out.data, target, topk=(1, 5))
        prec1 += tprec1
        prec5 += tprec5
        top1s.append(tprec1[0])
        top5s.append(tprec5[0])

    if val:
        for c in range(0, len(top1s)):
            print("Classifier {} top1: {} top5: {}".
              format(c, top1s[c], top5s[c]))
    prec1 = prec1 / len(output)
    prec5 = prec5 / len(output)
    return prec1, prec5, (top1s, top5s)


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1_per_cls = [AverageMeter() for i in range(0, model.module.num_blocks)]
    top5_per_cls = [AverageMeter() for i in range(0, model.module.num_blocks)]

    ### Switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        ### Compute output
        output = model(input_var, 0.0)
        if args.model == 'msdnet':
            loss = msd_loss(output, target_var, criterion)
        else:
            loss = criterion(output, target_var)

        ### Measure accuracy and record loss
        if hasattr(output, 'data'):
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        elif args.model == 'msdnet':
            prec1, prec5, _ = msdnet_accuracy(output, target, input)
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        ### Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1, top5=top5))
        
        _, _, (ttop1s, ttop5s) = msdnet_accuracy(output, target, input,
                                             val=True)
        for c in range(0,len(top1_per_cls)):
            top1_per_cls[c].update(ttop1s[c], input.size(0))
            top5_per_cls[c].update(ttop5s[c], input.size(0))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    for c in range(0, len(top1_per_cls)):
        print(' * For class {cls}: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                  .format(cls=c,top1=top1_per_cls[c], top5=top5_per_cls[c]))
    return 100. - top1.avg, 100. - top5.avg


def load_checkpoint(args):

    if args.evaluate_from:
        print("Evaluating from model: ", args.evaluate_from)
        model_filename = args.evaluate_from
    else:
        model_dir = os.path.join(args.savedir, 'save_models')
        latest_filename = os.path.join(model_dir, 'latest.txt')
        if os.path.exists(latest_filename):
            with open(latest_filename, 'r') as fin:
                model_filename = fin.readlines()[0].strip()
        else:
            return None
    print("=> loading checkpoint '{}'".format(model_filename))
    state = torch.load(model_filename)
    print("=> loaded checkpoint '{}'".format(model_filename))
    return state


def save_checkpoint(state, args, is_best, filename, result):
    print(args)
    result_filename = os.path.join(args.savedir, args.filename)
    model_dir = os.path.join(args.savedir, 'save_models')
    model_filename = os.path.join(model_dir, filename)
    latest_filename = os.path.join(model_dir, 'latest.txt')
    best_filename = os.path.join(model_dir, 'model_best.pth.tar')
    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)
        os.makedirs(model_dir)

    # For mkdir -p when using python3
    # os.makedirs(args.savedir, exist_ok=True)
    # os.makedirs(model_dir, exist_ok=True)

    print("=> saving checkpoint '{}'".format(model_filename))
    with open(result_filename, 'a') as fout:
        fout.write(result)
    torch.save(state, model_filename)
    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    if args.no_save_model:
        shutil.move(model_filename, best_filename)
    elif is_best:
        shutil.copyfile(model_filename, best_filename)

    print("=> saved checkpoint '{}'".format(model_filename))
    return


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args, batch=None,
                         nBatch=None, method='cosine'):
    if method == 'cosine':
        T_total = args.epochs * nBatch
        T_cur = (epoch % args.epochs) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        if args.data in ['cifar10', 'cifar100']:
            lr, decay_rate = args.lr, 0.1
            if epoch >= args.epochs * 0.75:
                lr *= decay_rate**2
            elif epoch >= args.epochs * 0.5:
                lr *= decay_rate
        else:
            """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
            lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()