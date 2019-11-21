import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='the path of the dataset')
parser.add_argument('--image-path', type=str, help='the path of the test image copies')
args = parser.parse_args()

paths = open(os.path.join(args.dataset, 'test.txt')).read().split()
image_paths = paths[0::2]
for path in image_paths:
    os.system(f"cp '{path}' {args.image_path}")