# -*- coding: utf-8 -*-
# file: calanchor.py
# brief: YOLOv3 implementation based on PyTorch
# author: Zeng Zhiwei
# date: 2019/9/27

import os
import sys
import cv2
import copy
import argparse
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(".")
from pascalvoc import PascalVocReader as pvr

def cal_iou(wh, centroids):
    def iou(wh, centroid):
        w = np.minimum(wh[:,0], centroid[0])
        h = np.minimum(wh[:,1], centroid[1])
        inter = w * h
        return inter / (wh[:,0] * wh[:,1] + centroid[0] * centroid[1] - inter)
    return np.stack([iou(wh, centroid) for centroid in centroids])

def assign_centroid(wh, centroids):
    ious = cal_iou(wh, centroids)
    nearest_centroid_idx = ious.argmax(axis=0)
    return nearest_centroid_idx, np.mean(ious.max(axis=0))

def update_centroid(wh, assignments, k, old_centroids):
    centroids = np.zeros((k, wh.shape[1]))
    for idx, centroid in enumerate(centroids):
        indices = assignments==idx
        if np.sum(indices):
            centroids[idx,:] = np.mean(wh[indices,:], axis=0)
        else:
            print(f'keep centroid {old_centroids[idx,:]} unchanged')
            centroids[idx,:] = old_centroids[idx,:]
    
    return centroids

def kmeans(wh, k, max_iters):
    idx = np.random.randint(low=0, high=wh.shape[0], size=(k,))
    centroids = wh[idx,:]
    assignments, avg_iou = assign_centroid(wh, centroids)
    
    iter = 0
    debug_info = []
    while iter < max_iters:
        centroids = update_centroid(wh, assignments, k, centroids)
        next_assignments, avg_iou = assign_centroid(wh, centroids)
        num_reassignment = sum(next_assignments!=assignments)
        
        debug_info.append((centroids, avg_iou, num_reassignment))
        if num_reassignment == 0:
            break
        
        assignments = copy.deepcopy(next_assignments)
        iter = iter + 1
    
    idx = np.argsort(centroids[:,0] * centroids[:,1])
    return centroids[idx,:].astype(np.int32), debug_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=416, help='network input size')
    parser.add_argument('--dataset', type=str, default='dataset', help='dataset path')
    parser.add_argument('--k', type=int, default=9, help='number of clusters')
    parser.add_argument('--max-iters', dest='max_iters', type=int, default=1000, help='maximum number of iterations')
    parser.add_argument('--workspace', type=str, default='workspace', help='workspace path')
    args = parser.parse_args()
    
    print('read annocation...', end='')
    path = open(os.path.join(args.dataset, 'train.txt')).read().split()
    image_path = path[0::2]
    label_path = path[1::2]
    
    wh = []
    for ip, lp in zip(image_path, label_path):
        image = cv2.imread(ip, cv2.IMREAD_COLOR)
        s = args.size / np.max(image.shape[:2])
        annocation = pvr(lp).getShapes()
        for an in annocation:            
            w = int(s * (an[1][1][0] - an[1][0][0] + 1))
            h = int(s * (an[1][2][1] - an[1][0][1] + 1))
            if w <= 1 or h <= 1:
                print(f'Box ERROR: {ip} {lp}')
                exit(-1)
            wh.append((w, h))
    
    print('done\ncluster...', end='')
    wh = np.array(wh)
    centroids, debug_info = kmeans(wh, args.k, args.max_iters)
        
    avg_iou = [di[1] for di in debug_info]
    num_reassignment = [di[2] for di in debug_info]
    print(f'done\nbest centroids: {centroids}, avg_iou: {avg_iou[-1]}')
    
    np.savetxt(f'{args.workspace}/log/anchors.txt', centroids, '%d')
    
    fig, ax1 = plt.subplots()
    ax1.set_title('Anchor Cluster')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('avg_iou')
    ax1.plot(avg_iou, 'r-')
    ax1.tick_params(axis='y')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('num_reassignment')
    ax2.plot(num_reassignment, 'b-')
    ax2.tick_params(axis='y')
    plt.savefig(f'{args.workspace}/log/cluster.jpg', dpi=800)
    
    fig2 = plt.figure()
    ax3 = fig2.add_subplot(111)
    ax3.set_title('W/H-centroids')
    ax3.set_xlabel('width')
    ax3.set_ylabel('height')
    ax3.scatter(x=wh[:,0], y=wh[:,1], c='r', marker='x')
    ax3.plot(centroids[:,0], centroids[:,1], 'bo', fillstyle=None)
    fig2.savefig(f'{args.workspace}/log/wh.jpg', dpi=800)