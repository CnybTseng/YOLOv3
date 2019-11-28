# -*- coding: utf-8 -*-
# file: train.py
# brief: YOLOv3 implementation based on PyTorch
# author: Zeng Zhiwei
# date: 2019/7/18

from __future__ import print_function
import os
import torch
import utils
import darknet
import argparse
import numpy as np
import shufflenetv2
import dataset as ds
import torch.utils.data
import evaluate as eval
from progressbar import *
import multiprocessing as mp
from functools import partial

def train_one_epoch(model, optimizer, lr_scheduler, data_loader, epoch, interval, shared_size, scale_sampler, sparsity=False, lamb=0.01):
    model.train()
    msgs = []
    widgets = ['Training epoch %d: ' % (epoch+1), Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=len(data_loader)).start()
    
    for batch_id, (images, target) in enumerate(data_loader):
        x = torch.cat(tensors=images, dim=0)    # Nx(1xCxHxW)=>NxCxHxW
        loss, metrics = model(x, target)
        
        total_batches = epoch * len(data_loader) + batch_id
        if total_batches % interval == 0:
            optimizer.zero_grad()

        loss.backward()
        if sparsity: model.correct_bn_grad(lamb)
        if total_batches % interval == interval - 1:
            lr_scheduler.step()
            with open('log/lr.txt', 'a') as file:
                file.write(f'{lr_scheduler.get_lr()[0]}\n')
                file.close()
            optimizer.step()
        
        msgs.append((loss.detach().cpu().item(), metrics))
        pbar.update(batch_id + 1)
        size = scale_sampler(total_batches + 1)
        shared_size[0], shared_size[1] = size[0], size[1]
    
    pbar.finish()
    return msgs

def main(args):
    try: mp.set_start_method('spawn')
    except RuntimeError: pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_size = [int(insz) for insz in args.in_size.split(',')]
    scale_step = [int(ss) for ss in args.scale_step.split(',')]
    anchors = np.loadtxt(os.path.join(args.dataset, 'anchors.txt'))
    scale_sampler = utils.TrainScaleSampler(scale_step, args.rescale_freq)
    shared_size = torch.IntTensor(in_size).share_memory_()

    dataset_train = ds.CustomDataset(args.dataset, 'train')
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=partial(ds.collate_fn, in_size=shared_size, train=True))
    
    dataset_valid = ds.CustomDataset(args.dataset, 'test')
    data_loader_valid = torch.utils.data.DataLoader(
        dataset=dataset_valid,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=partial(ds.collate_fn, in_size=torch.IntTensor(in_size), train=False))

    if args.checkpoint:
        print(f'load {args.checkpoint}')
        model = torch.load(args.checkpoint).to(device)
    else:
        print('please set fine tune model first!')
        return
    
    if args.test_only:
        mAP = eval.evaluate(model, data_loader_valid, device, args.num_classes)
        print(f'mAP of current model on validation dataset:%.2f%%' % (mAP * 100))
        return
    
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    if args.resume:
        trainer_state = torch.load('checkpoint/trainer-ckpt.pth')
        optimizer.load_state_dict(trainer_state['optimizer'])
 
    milestones = [int(ms) for ms in args.milestones.split(',')]
    def lr_lambda(iter):
        if iter < args.warmup:
            return pow(iter / args.warmup, 4)
        factor = 1
        for i in milestones:
            factor *= pow(args.lr_gamma, int(iter > i))
        return factor

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if args.resume:
        start_epoch = trainer_state['epoch'] + 1
        lr_scheduler.load_state_dict(trainer_state['lr_scheduler'])
    else:
        start_epoch = 0
    print(f'Start training from epoch {start_epoch}')

    best_mAP = 0
    for epoch in range(start_epoch, args.epochs):
        msgs = train_one_epoch(model, optimizer, lr_scheduler, data_loader, epoch, args.interval, shared_size, scale_sampler, args.sparsity, args.lamb)
        utils.print_training_message(epoch + 1, msgs, args.batch_size)
        torch.save(model, f"checkpoint/{args.savename}-ckpt-%03d.pth" % epoch)
        torch.save({
            'epoch' : epoch,
            'optimizer' : optimizer.state_dict(),
            'lr_scheduler' : lr_scheduler.state_dict()}, 'checkpoint/trainer-ckpt.pth')
        
        if epoch >= args.eval_epoch:
            mAP = eval.evaluate(model, data_loader_valid, device, args.num_classes)
            with open('log/mAP.txt', 'a') as file:
                file.write(f'{epoch} {mAP}\n')
                file.close()
            print(f'Current mAP:%.2f%%' % (mAP * 100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-size', type=str, default='416,416', help='network input size')
    parser.add_argument('--num-classes', type=int, default=20, help='number of classes')
    parser.add_argument('--resume', help='resume training', action='store_true')
    parser.add_argument('--checkpoint', type=str, default='', help='checkpoint model file')
    parser.add_argument('--dataset', type=str, default='dataset', help='dataset path')
    parser.add_argument('--batch-size', type=int, default=2, help='training batch size')
    parser.add_argument('--interval', type=int, default=32, help='update weights every #interval batches')
    parser.add_argument('--scale-step', type=str, default='320,608,10', help='scale step for multi-scale training')
    parser.add_argument('--rescale-freq', type=int, default=320, help='image rescaling frequency')
    parser.add_argument('--epochs', type=int, default=200, help='number of total epochs to run')
    parser.add_argument('--warmup', type=int, default=1000, help='warmup iterations')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--optim', type=str, default='sgd', help='optimization algorithms[adam or sgd]')
    parser.add_argument('--lr', type=float, default=0.00001, help='initial learning rate')
    parser.add_argument('--milestones', type=str, default='7829,11744', help='list of batch indices, must be increasing')
    parser.add_argument('--lr-gamma', type=float, default=0.1, help='factor of decrease learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--savename', type=str, default='yolov3', help='filename of trained model')
    parser.add_argument('--test-only', help='only test the model', action='store_true')
    parser.add_argument('--eval-epoch', type=int, default=10, help='epoch beginning evaluate')
    parser.add_argument('--sparsity', help='enable sparsity training', action='store_true')
    parser.add_argument('--lamb', type=float, default=0.01, help='sparsity factor')   
    args = parser.parse_args()
    print(args)
    main(args)