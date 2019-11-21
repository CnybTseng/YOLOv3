import os
from os import listdir, getcwd
from os.path import join
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='./', help='the path of VOCdevkit')
args = parser.parse_args()

sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test'), ('2012', 'train'), ('2012', 'val')]

for year, image_set in sets:
    image_ids = open(join(args.dir, 'VOCdevkit/VOC%s/ImageSets/Main/%s.txt' % (year, image_set))).read().strip().split()
    file = open('%s_%s.txt' % (year, image_set), 'w')
    for image_id in image_ids:
        file.write(join(args.dir, 'VOCdevkit', f'VOC{year}', 'JPEGImages', f'{image_id}.jpg') + ' ' + \
            join(args.dir, 'VOCdevkit', f'VOC{year}', 'Annotations', f'{image_id}.xml') + '\n')
    file.close()