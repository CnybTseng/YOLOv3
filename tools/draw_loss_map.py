# -*- coding: utf-8 -*-
# file: metric.py
# brief: YOLOv3 implementation based on PyTorch
# author: Zeng Zhiwei
# date: 2019/9/12

import matplotlib.pyplot as plt
import numpy as np
import os

loss = np.loadtxt('log/loss.txt')
min_loss_epoch = np.argmin(loss)

fig, ax1 = plt.subplots()
ax1.set_title('Training metrics')
ax1.set_xlim([0,max(200,len(loss))])
ax1.set_ylim([0,np.max(loss)*1.1])
ax1.set_xlabel('Training Epoch')
ax1.set_ylabel('Total Loss')
ax1.plot(loss, 'r-')
ax1.tick_params(axis='y')

title = 'Training metrics'
title += f'\nmin loss:%.2f, epoch:{min_loss_epoch}' % loss[min_loss_epoch]

ax2 = ax1.twinx()
ax2.set_ylim([0,1])
ax2.set_ylabel('mAP')

if os.path.isfile('log/mAP.txt'):
    mAP = np.loadtxt('log/mAP.txt')
    max_mAP_epoch = np.argmax(mAP[:,1])
    # ax2.bar(mAP[:,0], mAP[:,1], width=0.2)
    ax2.plot(mAP[:,0], mAP[:,1], 'y-')
    ax2.tick_params(axis='y')

    mme = np.int(mAP[max_mAP_epoch,0])
    title += f'\nmax mAP:%.2f, epoch:{mme}' % mAP[max_mAP_epoch,1]

if os.path.isfile('log/evaluation.txt'):
    mAP = np.loadtxt('log/evaluation.txt')
    max_mAP_epoch = np.argmax(mAP[:,1])
    ax2.plot(mAP[:,0], mAP[:,1], 'b-')
    ax2.tick_params(axis='y')

    mme = np.int(mAP[max_mAP_epoch,0])
    title += f'\nmax mAP(eval-only):%.2f, epoch:{mme}' % mAP[max_mAP_epoch,1]

ax1.set_title(title)
fig.tight_layout()
plt.show()