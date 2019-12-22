# -*- coding: utf-8 -*-
# file: box.py
# brief: YOLOv3 implementation based on PyTorch
# author: Zeng Zhiwei
# date: 2019/9/12

import os
import cv2
import torch
import numpy as np
import copy

def get_network_boxes(dets, im_size, thresh=0.5):
    dets = dets.squeeze()
    dets = dets[dets[:,4]>thresh,:]
    dets[:,5:] = dets[:,4].view(dets[:,4].size(0), 1) * dets[:,5:]
    dets[:,5:][dets[:,5:]<=thresh] = 0
    long_side = max(im_size)
    for det in dets:
        x, y, w, h = det[:4] * long_side
        if im_size[0] == long_side:
            x = x - (long_side - im_size[1])/2
        else:
            y = y - (long_side - im_size[0])/2
        det[0] = x - w/2
        det[1] = y - h/2
        det[2] = x + w/2
        det[3] = y + h/2
    return dets.numpy()

def box_iou(box1, box2, eps=1e-16):
    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])    
    inter_w = max(0, xmax - xmin)
    inter_h = max(0, ymax - ymin)
    inter_a = inter_w * inter_h    
    box1_a = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_a = (box2[2] - box2[0]) * (box2[3] - box2[1])    
    return inter_a / (box1_a + box2_a - inter_a + eps)

def do_nms_sort(dets, ovthresh=0.45):
    num_dets = dets.shape[0]
    num_classes = dets[:, 5:].shape[1]
    for c in range(num_classes):
        sorted_indices = np.argsort(dets[:, 5+c])[::-1]
        dets = dets[sorted_indices, :]
        for i in range(num_dets):
            if dets[i, 5+c] == 0:
                continue
            for j in range(i+1, num_dets):
                iou = box_iou(dets[i, :4], dets[j, :4])
                if iou > ovthresh:
                    dets[j, 5+c] = 0
    return dets

def overlap_detection(im, dets, class_names):    
    for det in dets:                       
        if det[4] == 0 or np.max(det[5:]) == 0:
            continue

        # 获取字符串尺寸
        id = np.argmax(det[5:])
        [size, baseline] = cv2.getTextSize(text=class_names[id], fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, thickness=1)
        
        # 字符串水平位置限界
        minx, miny, maxx, maxy = det[:4].astype(np.int32)
        tw, th = size[0], size[1]
        tx = minx if minx >= 0 else 0
        tx = tx if tx < im.shape[1]-tw else im.shape[1]-tw-1
        
        # 字符串垂直位置限界.
        # 字符串的高度等于getTextSize返回的高度加上baseline
        ty = miny-baseline if miny-baseline >= th else th 
        ty = ty if ty < im.shape[0]-baseline else im.shape[0]-baseline-1
        
        # 叠加类别标签
        cv2.rectangle(img=im, pt1=(tx,ty-th), pt2=(tx+tw-1,ty+baseline), color=(0,255,255), thickness=-1)
        cv2.putText(img=im, text=class_names[id], org=(tx,ty), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(0,0,255), thickness=1)
        cv2.rectangle(img=im, pt1=(minx,miny), pt2=(maxx,maxy), color=(0,255,255), thickness=1)

    return im

if __name__ == '__main__':
    dets = torch.rand(1, 10, 8)
    print(dets)
    dets = get_network_boxes(dets, (576,720))
    print(dets)