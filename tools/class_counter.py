# -*- coding: utf-8 -*-
# file: class_counter.py
# brief: Tile image into sub-images with overlap.
# author: Zeng Zhiwei
# date: 2019/9/26

import os
import sys
import glob
import argparse
from collections import defaultdict
sys.path.append('.')
from pascalvoc import PascalVocReader as pvr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='', help='labels path')
    args = parser.parse_args()
    
    label_names = glob.glob(os.path.join(args.path, '*.xml'))
    class_counter = defaultdict(int)
    for lab_nm in label_names:
        label = pvr(lab_nm).getShapes()
        for l in label:
            class_counter[l[0]] += 1
    
    print(f'class statistics: {class_counter}')
            