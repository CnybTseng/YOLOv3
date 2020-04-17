import os
import re
import cv2
import sys
import glob
import copy
import shutil
import argparse
import numpy as np
sys.path.append(".")
from pascalvoc import PascalVocReader as pvr
from pascalvoc import PascalVocWriter as pvw

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-path', type=str, default='', help='source image and label path')
    parser.add_argument('--dst-path', type=str, default='', help='destination image and label path')
    parser.add_argument('--classes', type=str, default='', help='classes list for removing')
    args = parser.parse_args()
    print(args)
    
    class_list = [c for c in args.classes.split(',')]
    
    image_filenames  = []
    image_filenames += list(sorted(glob.glob(os.path.join(args.src_path, '*.jpg'))))
    image_filenames += list(sorted(glob.glob(os.path.join(args.src_path, '*.jpeg'))))
    image_filenames += list(sorted(glob.glob(os.path.join(args.src_path, '*.JPG'))))
    image_filenames += list(sorted(glob.glob(os.path.join(args.src_path, '*.png'))))
    
    for image_filename in image_filenames:
        root, ext = os.path.splitext(image_filename)
        label_filename = root + '.xml'
        labels = pvr(label_filename).getShapes()
        need_remove = False
        for lb in labels:
            if lb[0] in class_list:
                need_remove = True
                break
        if need_remove:
            # shutil.copy2(image_filename, args.dst_path)
            # shutil.copy2(label_filename, args.dst_path)
            os.system(f"move {image_filename} {args.dst_path}")
            os.system(f"move {label_filename} {args.dst_path}")