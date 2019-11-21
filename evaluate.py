# -*- coding: utf-8 -*-
# file: evaluate.py
# brief: YOLOv3 implementation based on PyTorch
# author: Zeng Zhiwei
# date: 2019/8/22

from __future__ import print_function
import os
import re
import glob
import torch
import utils
import darknet
import argparse
import numpy as np
import shufflenetv2
import dataset as ds
import torch.utils.data
from progressbar import *
import multiprocessing as mp
from functools import partial

def evaluate(model, data_loader, device, num_classes):
    dets = list()
    gts = list()
    widgets = ['Evaluate: ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=len(data_loader)).start()
    for batch_id, (images, targets) in enumerate(data_loader):
        model.eval()
        x = torch.cat(tensors=images, dim=0)
        with torch.no_grad():
            y = model(x)
            z = utils.get_network_boxes(y, thresh=0.25)
            nms = utils.nms_obj(z)
        dets.append(nms)
        gts.append(targets)
        pbar.update(batch_id + 1)
    
    mAP = utils.calc_mAP(dets, gts, num_classes, False)
    pbar.finish()
    return mAP

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-size', dest='in_size', type=str, default='416,416', help='network input size')
    parser.add_argument('--model', type=str, default='', help='model file')
    parser.add_argument('--num-classes', dest='num_classes', type=int, default=20, help='number of classes')
    parser.add_argument('--dataset', type=str, default='', help='dataset path')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--eval-epoch', dest='eval_epoch', type=int, default=10, help='epoch beginning evaluate')
    args = parser.parse_args()
    print(args)
    
    try: mp.set_start_method('spawn')
    except RuntimeError: pass
    
    in_size = [int(insz) for insz in args.in_size.split(',')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = ds.CustomDataset(args.dataset, 'test')
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=partial(ds.collate_fn, in_size=torch.IntTensor(in_size), train=False))
    
    if os.path.isdir(args.model):
        paths = list(sorted(glob.glob(os.path.join(args.model, '*.pth'))))
        mAPs = list()
        for path in paths:
            if 'trainer' in path: continue
            segments = re.split(r'[-,.]', path)
            if int(segments[-2]) < args.eval_epoch: continue
            
            anchors = np.loadtxt(os.path.join(args.dataset, 'anchors.txt'))
            model = darknet.DarkNet(anchors, in_size, num_classes=args.num_classes).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            mAP = evaluate(model, data_loader, device, args.num_classes)
            mAPs.append(mAP)
            
            with open('log/evaluation.txt', 'a') as file:
                file.write(f'{int(segments[-2])} {mAP}\n')
                file.close()
            
            print(f'mAP of {path} on validation dataset:%.2f%%' % (mAP * 100))
        mAPs = np.array(mAPs)
        epoch = np.argmax(mAPs)
        print(f'Best model is ckpt-{epoch+args.eval_epoch}, best mAP is %.2f%%' % (mAPs[epoch] * 100))
    else:
        anchors = np.loadtxt(os.path.join(args.dataset, 'anchors.txt'))
        model = darknet.DarkNet(anchors, in_size, num_classes=args.num_classes).to(device)
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()
        mAP = evaluate(model, data_loader, device, args.num_classes)
        print(f'mAP of {args.model} on validation dataset:%.2f%%' % (mAP * 100))