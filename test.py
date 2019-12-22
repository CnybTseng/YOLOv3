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
import box
import utils
import torch
import yolov3
import dataset
import darknet
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-size', type=str, default='416,416', help='network input size')
    parser.add_argument('--model', type=str, default='', help='model file')
    parser.add_argument('--dataset', type=str, default='', help='dataset path')
    parser.add_argument('--image', type=str, default='', help='test image filename')
    parser.add_argument('--thresh', type=float, default=0.5, help='objectness threshold')
    parser.add_argument('--store', help='store the detection result or not', action='store_true')
    args = parser.parse_args()
    print(args)
    
    in_size = [int(insz) for insz in args.in_size.split(',')]
    class_names = utils.load_class_names(os.path.join(args.dataset, 'classes.txt'))
    anchors = np.loadtxt(os.path.join(args.dataset, 'anchors.txt'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = darknet.DarkNet(anchors, in_size=in_size, num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    
    decoder = yolov3.YOLOv3EvalDecoder(in_size, len(class_names), anchors)
    transform = dataset.get_transform(train=False, net_w=in_size[0], net_h=in_size[1])
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    
    def save_detection_result(filename, im_size, dets, classnames):
        num_classes = dets[:,5:].shape[1]
        segments = re.split(r'[\\,/]', filename)
        filename = segments[-1].split('.')[0]
        for det in dets:
            if det[4] == 0 or np.max(det[5:]) == 0:
                continue
            classid = np.argmax(det[5:])
            classname = classnames[classid]
            minx, miny, maxx, maxy = det[:4] + 1
            minx = max(1, minx)
            miny = max(1, miny)
            maxx = min(im_size[1], maxx)
            maxy = min(im_size[0], maxy)
            for c in range(num_classes):
                if det[5+c] > 0:
                    with open(f"results/comp4_det_test_{classname}.txt", "a") as file:
                        file.write(f"{filename} {det[5+c]} {minx} {miny} {maxx} {maxy}\n")
    
    def process_single_image(filename):
        bgr = cv2.imread(filename, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x, _ = transform(rgb, None)
        x = x.type(FloatTensor) / 255
        ys = model(x)
        dets = decoder(ys)
        dets = box.get_network_boxes(dets, bgr.shape[:2], thresh=args.thresh)
        dets = box.do_nms_sort(dets)
        if args.store:
            save_detection_result(filename, rgb.shape[:2], dets, class_names)
        return box.overlap_detection(bgr, dets, class_names), dets            
    
    if os.path.isfile(args.image):
        result, dets = process_single_image(args.image)
        cv2.imwrite('detection.jpg', result)
    elif os.path.isdir(args.image):
        images = glob.glob(os.path.join(args.image, '*.jpg')) + glob.glob(os.path.join(args.image, '*.JPG'))
        for image in images:
            result, dets = process_single_image(image)
            im_nm = re.split(r'[\\,/]', image)
            result_name = os.path.join('detection', im_nm[-1])
            cv2.imwrite(result_name, result)
            print(f'detect {image} -> {result_name}')
    else:
        print('no input file!')