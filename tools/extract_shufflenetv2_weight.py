# -*- coding: utf-8 -*-
# file: extract_shufflenetv2_weight.py
# brief: YOLOv3 implementation based on PyTorch
# author: Zeng Zhiwei
# date: 2019/11/12

import os
import sys
import torch
import argparse
import numpy as np
sys.path.append('.')
import shufflenetv2
from collections import OrderedDict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-model', '-s', type=str, help='source model file')
    parser.add_argument('--dst-model', '-d', type=str, help='destination model file')
    parser.add_argument('--in-size', type=str, default='416,416', help='network input size')
    parser.add_argument('--num-classes', type=int, default=20, help='number of classes')
    args = parser.parse_args()
    print(args)
    
    if '0.5x' in args.src_model:
        model_size = '0.5x'
    elif '1.0x' in args.src_model:
        model_size = '1.0x'
    elif '1.5x' in args.src_model:
        model_size = '1.5x'
    elif '2.0x' in args.src_model:
        model_size = '2.0x'
    
    in_size = [int(insz) for insz in args.in_size.split(',')]
    anchors = np.random.randint(low=10, high=150, size=(9,2))
    model = shufflenetv2.ShuffleNetV2(anchors, in_size, num_classes=args.num_classes, model_size=model_size)
    
    checkpoint = torch.load(args.src_model, map_location='cpu')
    if 'state_dict' in checkpoint:
        checkpoint = dict(checkpoint['state_dict'])
    
    src_param = list()
    for k, v in checkpoint.items():
        if 'conv_last' in k: break
        if not 'num_batches_tracked' in k:
            src_param.append(v)
    
    src_layer_index = 0
    for name, module in model.named_modules():
        if 'stage5' in name:
            if src_layer_index != len(src_param):
                print(f'something is wrong when reading parameters from source model!')
            break
        
        if isinstance(module, torch.nn.Conv2d):
            assert module.weight.data.size() == src_param[src_layer_index].data.size()
            module.weight.data = src_param[src_layer_index].data.clone()
            src_layer_index = src_layer_index + 1
        elif isinstance(module, torch.nn.BatchNorm2d):
            assert module.weight.data.size() == src_param[src_layer_index].data.size()
            module.weight.data = src_param[src_layer_index].data.clone()
            src_layer_index = src_layer_index + 1
            assert module.bias.data.size() == src_param[src_layer_index].data.size()
            module.bias.data = src_param[src_layer_index].data.clone()
            src_layer_index = src_layer_index + 1
            assert module.running_mean.data.size() == src_param[src_layer_index].data.size()
            module.running_mean.data = src_param[src_layer_index].data.clone()
            src_layer_index = src_layer_index + 1
            assert module.running_var.data.size() == src_param[src_layer_index].data.size()
            module.running_var.data = src_param[src_layer_index].data.clone()
            src_layer_index = src_layer_index + 1
    
    torch.save(model.state_dict(), f'{args.dst_model}')