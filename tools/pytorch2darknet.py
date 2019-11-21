# -*- coding: utf-8 -*-
# file: pytorch2darknet.py
# brief: YOLOv3 implementation based on PyTorch
# author: Zeng Zhiwei
# date: 2019/11/6

import os
import sys
import torch
sys.path.append('.')
import darknet
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-size', type=str, default='416,416', help='network input size')
    parser.add_argument('--pytorch-model', '-pm', type=str, dest='pm', help='pytorch-format model file')
    parser.add_argument('--dataset', type=str, default='', help='dataset path')
    parser.add_argument('--num-classes', type=int, default=3, help='number of classes')
    parser.add_argument('--darknet-model', '-dm', type=str, dest='dm', default='darknet.weights', help='darknet-format model file')
    args = parser.parse_args()
    
    tiny = 'tiny' in args.pm
    in_size = [int(insz) for insz in args.in_size.split(',')]
    anchors = np.loadtxt(os.path.join(args.dataset, 'anchors.txt'))
    model = darknet.DarkNet(anchors, in_size=in_size, num_classes=args.num_classes, tiny=tiny)
    model.load_state_dict(torch.load(args.pm, map_location='cpu'))
    
    with open(args.dm, 'wb') as file:
        header = np.asarray([0,2,0,0,0], dtype=np.int32)
        header.tofile(file)
        last_conv = None
        last_name = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
                last_name = name
            elif isinstance(module, torch.nn.BatchNorm2d):
                print(f'read from {name}')
                module.bias.data.cpu().numpy().tofile(file)
                module.weight.data.cpu().numpy().tofile(file)
                module.running_mean.data.cpu().numpy().tofile(file)
                module.running_var.data.cpu().numpy().tofile(file)
                if last_conv is not None:
                    print(f'read from {last_name}')
                    last_conv.weight.data.cpu().numpy().tofile(file)
                    last_conv = None
                else:
                    print(f"the module before {name} isn't Conv2d, that's impossible!!!")
                    break
            else:
                if last_conv is not None:
                    if last_conv.bias is not None:
                        print(f'read from {last_name}.bias')
                        last_conv.bias.data.cpu().numpy().tofile(file)
                    print(f'read from {last_name}')
                    last_conv.weight.data.cpu().numpy().tofile(file)
                last_conv = None
        file.close()