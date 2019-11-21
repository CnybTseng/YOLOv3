import os
import sys
import glob
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to training samples')
parser.add_argument('--train-ratio', '-tr', dest='tr', default=1, type=float, help='the ratio of training samples')
args = parser.parse_args()

assert args.path
assert args.tr > 0.5 and args.tr < 1.000001

image_filenames = list(sorted(glob.glob(os.path.join(args.path, '*.jpg')))) + list(sorted(glob.glob(os.path.join(args.path, '*.JPG'))))
label_filenames = list(sorted(glob.glob(os.path.join(args.path, '*.xml'))))

num_samples = len(image_filenames)
rand_index = np.random.permutation(num_samples)
num_train = int(args.tr * num_samples)
num_test = num_samples - num_train

with open('train.txt', 'w') as file:
    for i in range(num_train):
        file.write(f"{image_filenames[rand_index[i]]} {label_filenames[rand_index[i]]}\n")
    file.close()

if num_test < 1:
    sys.exit()

with open('test.txt', 'w') as file:
    for i in range(num_train, num_samples):
        file.write(f"{image_filenames[rand_index[i]]} {label_filenames[rand_index[i]]}\n")
    file.close()