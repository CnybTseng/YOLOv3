# -*- coding: utf-8 -*-
# file: test.py
# brief: YOLOv3 implementation based on PyTorch
# author: Zeng Zhiwei
# date: 2019/7/12

import re
import os
import cv2
import sys
import glob
import utils
import torch
import yolov3
import dataset
import darknet
import argparse
import shufflenetv2
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-size', type=str, default='416,416', help='network input size')
    parser.add_argument('--model', type=str, default='', help='model file')
    parser.add_argument('--dataset', type=str, default='', help='dataset path')
    parser.add_argument('--image', type=str, default='', help='test image filename')
    parser.add_argument('--thresh', type=float, default=0.5, help='objectness threshold')
    parser.add_argument('--store', help='store the detection result or not', action='store_true')
    parser.add_argument('--backbone', type=str, default='darknet53', help='backbone architecture[darknet53(default),shufflenetv2]')
    parser.add_argument('--pruned-model', action='store_true')
    args = parser.parse_args()
    print(args)
    
    in_size = [int(insz) for insz in args.in_size.split(',')]
    classnames = utils.load_class_names(os.path.join(args.dataset, 'classes.txt'))
    anchors = np.loadtxt(os.path.join(args.dataset, 'anchors.txt'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    norm = 255 if args.backbone == 'darknet53' else 1
    
    if not args.pruned_model:
        if args.backbone == 'darknet53':
            model = darknet.DarkNet(anchors, in_size=in_size, num_classes=len(classnames)).to(device)
        elif args.backbone == 'shufflenetv2':
            model = shufflenetv2.ShuffleNetV2(anchors, in_size=in_size, num_classes=len(classnames)).to(device)
        else:
            print('unknown backbone architecture!')
            sys.exit(0)
        model.load_state_dict(torch.load(args.model, map_location=device))
    else:
        model = torch.load(args.model, map_location=device)
    model.eval()
    
    decoder = yolov3.YOLOv3EvalDecoder(in_size, len(classnames), anchors)
    transform = dataset.get_transform(train=False, net_w=in_size[0], net_h=in_size[1])
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        
    def process_single_image(filename):
        bgr = cv2.imread(filename, cv2.IMREAD_COLOR)
        assert bgr is not None, 'cv2.imread({}) fail'.format(filename)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x, _ = transform(rgb, None)
        x = x.type(FloatTensor) / norm
        ys = model(x)
        dets = decoder(ys)
        dets = utils.get_network_boxes(dets, bgr.shape[:2], thresh=args.thresh)
        dets = utils.do_nms_sort(dets)
        if args.store:
            utils.save_detection_result(filename, rgb.shape[:2], dets, classnames)        
        result = utils.overlap_detection(bgr, dets, classnames)
        filename = re.split(r'[\\,/]', filename)
        result_name = os.path.join('detection', filename[-1])
        cv2.imwrite(result_name, result)           
    
    if os.path.isfile(args.image):
        if os.path.splitext(args.image)[-1] != '.txt':
            process_single_image(args.image)
        else:
            paths = open(args.image).read().split()
            image_paths = paths[0::2]
            for path in image_paths:
                process_single_image(path)
    elif os.path.isdir(args.image):
        images = glob.glob(os.path.join(args.image, '*.jpg')) + glob.glob(os.path.join(args.image, '*.JPG'))
        for image in images:
            process_single_image(image)
    else:
        print('no input file!')