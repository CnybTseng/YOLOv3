# -*- coding: utf-8 -*-
# file: ncnn.py
# brief: YOLOv3 implementation based on PyTorch
# author: Zeng Zhiwei
# date: 2019/8/8

from __future__ import print_function
import argparse
import numpy as np
import os
import sys
from os.path import splitext
import torch
sys.path.append('.')
import darknet as net

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', help='model file')
    parser.add_argument('--num-classes', dest='num_classes', type=int, default=80, help='number of classes')
    parser.add_argument('--insize', type=str, default='416,416', help='input size')
    parser.add_argument('--dataset', type=str, default='', help='dataset path')
    parser.add_argument('--onnxname', type=str, default='model.onnx', help='exported ONNX filename')
    parser.add_argument('--pruned-model', action='store_true', help='the model has been pruned')
    args = parser.parse_args()
    
    insize = [int(s) for s in args.insize.split(',')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not args.pruned_model:
        anchors = np.loadtxt(os.path.join(args.dataset, 'anchors.txt'))    
        model = net.DarkNet(anchors, insize, args.num_classes).to(device)
        model.load_state_dict(torch.load(args.model, map_location=device))
    else:
        model = torch.load(args.model, map_location=device)
    
    input_names = ["data"]
    dummy_input = torch.randn(1, 3, insize[0], insize[1], device=device)
    torch.onnx.export(model, dummy_input, args.onnxname, verbose=True, input_names=input_names)

    # filename, extension = splitext(args.onnxname)
    # os.system(f'python -m onnxsim {args.onnxname} {args.onnxname}')
    # os.system(f'./model/onnx/onnx2ncnn {args.onnxname} {filename}.param {filename}.bin')