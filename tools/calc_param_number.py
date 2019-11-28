# -*- coding: utf-8 -*-
# file: calc_param_number.py
# brief: YOLOv3 implementation based on PyTorch
# author: Zeng Zhiwei
# date: 2019/11/27

import os
import sys
import torch
import numpy as np
sys.path.append('.')
import darknet
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-size', type=str, default='416,416', help='network input size')
    parser.add_argument('--model', type=str, default='', help='model file')
    parser.add_argument('--dataset', type=str, default='', help='dataset path')
    parser.add_argument('--num-classes', type=int, default=3, help='number of classes')
    parser.add_argument('--pruned-model', '-pm', action='store_true')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not args.pruned_model:
        in_size = [int(insz) for insz in args.in_size.split(',')]
        anchors = np.loadtxt(os.path.join(args.dataset, 'anchors.txt'))
        
        model = darknet.DarkNet(anchors, in_size, num_classes=args.num_classes).to(device)
        model.load_state_dict(torch.load(args.model, map_location=device))
    else:
        model = torch.load(args.model, map_location=device)
    
    model.eval()
    nparams = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            nparams += module.weight.numel()
            if module.bias is not None:
                nparams += module.bias.numel()
        if isinstance(module, torch.nn.BatchNorm2d):
            nparams += module.weight.numel()
            nparams += module.bias.numel()
            nparams += module.running_mean.numel()
            nparams += module.running_var.numel()
    
    print(f'total number of parameters is {nparams}')