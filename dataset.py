# -*- coding: utf-8 -*-
# file: dataset.py
# brief: YOLOv3 implementation based on PyTorch
# author: Zeng Zhiwei
# date: 2019/7/17

import os
import glob
import numpy as np
import cv2
import torch
from pascalvoc import PascalVocReader as pvr
import utils
import sys
import transforms as T
import torch.nn.functional as tnf

def get_transform(train, net_w=416, net_h=416):
    transforms = []
    transforms.append(T.ToTensor())
    if train == True:
        transforms.append(T.RandomSpatialJitter(jitter=0.3,net_w=net_w,net_h=net_h))
        transforms.append(T.RandomColorJitter(hue=0.1,saturation=1.5,exposure=1.5))
        transforms.append(T.RandomHorizontalFlip(prob=0.5))
    else:
        transforms.append(T.MakeLetterBoxImage(width=net_w,height=net_h))
    return T.Compose(transforms)

def collate_fn(batch, in_size=torch.IntTensor([416,416]), train=False):
    # with open(os.path.join('log', 'collate_fn.txt'), 'a') as file:
    #     file.write(f'{os.getpid()} {in_size}\n')
    transformed_batch = []
    transforms = get_transform(train, in_size[0].item(), in_size[1].item())
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    for b in batch:
        image, target = transforms(b[0], b[1])
        image = image.type(FloatTensor) / 255.0
        transformed_batch.append((image, target))
    return tuple(zip(*transformed_batch))

class CustomDataset(object):
    '''定制的数据集.
    '''
    
    def __init__(self, root, file='train'):
        self.root = root
        path = open(os.path.join(root, f'{file}.txt')).read().split()
        self.images_path = path[0::2]
        self.annocations_path = path[1::2]
        self.class_names = self.__load_class_names(os.path.join(root, 'classes.txt'))
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')
        self.FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

    def __getitem__(self, index):
        image_path = self.images_path[index]
        annocation_path = self.annocations_path[index]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        org_size = image.shape[:2]  # HxWxC => HxW
        annocation = pvr(annocation_path).getShapes()
        
        boxes = []
        labels = []
        for an in annocation:
            name = an[0]
            xmin = an[1][0][0]
            ymin = an[1][0][1]
            xmax = an[1][1][0]
            ymax = an[1][2][1]
            self.__assert_bbox(xmin, ymin, xmax, ymax, org_size, image_path)
            bx, by, bw, bh = self.__xyxy_to_xywh(xmin, ymin, xmax, ymax, org_size)
            boxes.append([bx, by, bw, bh])
            labels.append(self.class_names.index(name))

        boxes = torch.as_tensor(boxes, dtype=torch.float32, device=self.device)
        labels = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
        target = {'boxes':boxes, 'labels':labels}

        return image, target
    
    def __len__(self):
        return len(self.images_path)
        
    def __load_class_names(self, path):
        class_names = []
        with open(path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                name = line.rstrip('\n')
                if not name: continue
                class_names.append(name)
            file.close()
        return class_names
    
    def __xyxy_to_xywh(self, xmin, ymin, xmax, ymax, size):
        bx = (xmin + xmax) / 2 / size[1]
        by = (ymin + ymax) / 2 / size[0]
        bw = (xmax - xmin + 1) / size[1]
        bh = (ymax - ymin + 1) / size[0]
        return bx, by, bw, bh
    
    def __assert_bbox(self, xmin, ymin, xmax, ymax, size, filename):
        if xmin < 0 or xmin > size[1]:
            print(f'BAD BOUNDING BOX! xmin={xmin}, size={size}, {filename}')
            sys.exit()
        if xmin > 0: xmin -= 1
        if ymin < 0 or ymin > size[0]:
            print(f'BAD BOUNDING BOX! ymin={ymin}, size={size}, {filename}')
            sys.exit()
        if ymin > 0: ymin -= 1
        if xmax < 0 or xmax > size[1]:
            print(f'BAD BOUNDING BOX! xmax={xmax}, size={size}, {filename}')
            sys.exit()
        xmax -= 1
        if ymax < 0 or ymax > size[0]:
            print(f'BAD BOUNDING BOX! ymax={ymax}, size={size}, {filename}')
            sys.exit()
        ymax -= 1
        