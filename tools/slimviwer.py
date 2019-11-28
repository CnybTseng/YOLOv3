# -*- coding: utf-8 -*-
# file: slimviwer.py
# brief: YOLOv3 implementation based on PyTorch
# author: Zeng Zhiwei
# date: 2019/11/27

import os
import re
import cv2
import sys
import glob
import torch
import numpy as np
sys.path.append('.')
import utils
import darknet
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-size', type=str, default='416,416', help='network input size')
    parser.add_argument('--model', type=str, default='', help='model file')
    parser.add_argument('--dataset', type=str, default='', help='dataset path')
    parser.add_argument('--num-classes', type=int, default=3, help='number of classes')
    parser.add_argument('--eval-epoch', type=int, default=50, help='epoch beginning evaluate')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_size = [int(insz) for insz in args.in_size.split(',')]
    anchors = np.loadtxt(os.path.join(args.dataset, 'anchors.txt'))
    
    paths = list(sorted(glob.glob(os.path.join(args.model, '*.pth'))))
    weights = {}
    minimum =  1000000
    maximum = -1000000
    
    for path in paths:
        if 'trainer' in path: continue
        segments = re.split(r'[-,.]', path)
        if int(segments[-2]) < args.eval_epoch: continue
        
        model = darknet.DarkNet(anchors, in_size, num_classes=args.num_classes).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.load_prune_permit('model/prune_permit.json')

        print(f'process {path}')
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.BatchNorm2d):
                continue
            else:
                if not model.prune_permit[name][1]:
                    continue
            
            if name not in weights:
                weights[name] = list()
            weight = module.weight.detach().cpu().abs().numpy()
            local_minimum = np.amin(weight)
            local_maximum = np.amax(weight)
            minimum = local_minimum if minimum > local_maximum else minimum
            maximum = local_maximum if maximum < local_maximum else maximum
            weights[name].append(weight.tolist())
    
    print(f'minimum is {minimum}, maximum is {maximum}')
    for layer, weight in weights.items():
        weight_image = np.stack(weight)
        print(f'process weight of {layer}, size {weight_image.shape}')
        minimum, maximum = np.amin(weight), np.amax(weight)
        weight_image = 255 * (weight_image - minimum)/(maximum - minimum + 1e-6)
        weight_image = weight_image.astype(np.uint8)
        cv2.imwrite(os.path.join('log', f'{layer}.jpg'), weight_image)