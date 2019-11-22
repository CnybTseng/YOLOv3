# -*- coding: utf-8 -*-
# file: test.py
# brief: YOLOv3 implementation based on PyTorch
# author: Zeng Zhiwei
# date: 2019/7/12

from __future__ import print_function
import re
import os
import cv2
import glob
import utils
import torch
import dataset
import darknet
import argparse
import numpy as np
import shufflenetv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-size', type=str, default='416,416', help='network input size')
    parser.add_argument('--model', type=str, default='', help='model file')
    parser.add_argument('--dataset', type=str, default='', help='dataset path')
    parser.add_argument('--image', type=str, default='', help='test image filename')
    parser.add_argument('--thresh', type=float, default=0.5, help='objectness threshold')
    args = parser.parse_args()
    print(args)
    
    in_size = [int(insz) for insz in args.in_size.split(',')]
    class_names = utils.load_class_names(os.path.join(args.dataset, 'classes.txt'))
    anchors = np.loadtxt(os.path.join(args.dataset, 'anchors.txt'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = darknet.DarkNet(anchors, in_size=in_size, num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    
    transform = dataset.get_transform(train=False, net_w=in_size[0], net_h=in_size[1])
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    
    def process_single_image(filename):
        bgr = cv2.imread(filename, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x, _ = transform(rgb, None)
        x = x.type(FloatTensor) / 255.0
        y = model(x)
        z = utils.get_network_boxes(y.clone(), bgr.shape[:2], thresh=args.thresh)
        nms = utils.nms_obj(z)
        return utils.overlap_detection(bgr, nms, class_names)
    
    if os.path.isfile(args.image):
        result = process_single_image(args.image)
        cv2.imwrite('detection.jpg', result)
    elif os.path.isdir(args.image):
        images = glob.glob(os.path.join(args.image, '*.jpg')) + glob.glob(os.path.join(args.image, '*.JPG'))
        for image in images:
            result = process_single_image(image)
            im_nm = re.split(r'[\\,/]', image)
            result_name = os.path.join('detection', im_nm[-1])
            cv2.imwrite(result_name, result)
            print(f'detect {image} -> {result_name}')
    else:
        print('no input file!')