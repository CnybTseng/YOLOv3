# -*- coding: utf-8 -*-
# file: remove_class.py
# brief: Remove specific class from annotations.
# author: Zeng Zhiwei
# date: 2019/10/28

import os
import re
import cv2
import sys
import glob
import copy
import argparse
import numpy as np
sys.path.append(".")
from pascalvoc import PascalVocReader as pvr
from pascalvoc import PascalVocWriter as pvw

class ClassRemover(object):
    def __init__(self, class_list):
        self.class_list = class_list
    
    def __call__(self, src_path, dst_path):
        image_filenames = list(sorted(glob.glob(os.path.join(src_path, '*.jpg'))))
        label_filenames = list(sorted(glob.glob(os.path.join(src_path, '*.xml'))))
        reg = re.compile(re.escape('.xml'), re.IGNORECASE)
        for im_fn, lb_fn in zip(image_filenames, label_filenames):
            print(f'remove {self.class_list} from {lb_fn}...', end='')
            _, name = os.path.split(lb_fn)
            name = reg.sub('', name)
            labels = pvr(lb_fn).getShapes()
            image  = cv2.imread(im_fn, cv2.IMREAD_COLOR)
            writer = pvw(dst_path, name + '.jpg', image.shape, localImgPath=os.path.join(dst_path, name + '.jpg'))
            for lb in labels:
                if lb[0] not in self.class_list:
                    writer.addBndBox(lb[1][0][0], lb[1][0][1], lb[1][1][0], lb[1][2][1], lb[0], False)
            writer.save(os.path.join(dst_path, name + '.xml'))
            print('done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-path', type=str, default='', help='source image and label path')
    parser.add_argument('--dst-path', type=str, default='', help='destination image and label path')
    parser.add_argument('--classes', type=str, default='', help='classes list for removing')
    args = parser.parse_args()
    print(args)
    
    class_list = [c for c in args.classes.split(',')]
    
    remover = ClassRemover(class_list)
    remover(args.src_path, args.dst_path)