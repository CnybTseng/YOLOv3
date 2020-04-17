import os
import glob
import argparse

def rename_file(path, id):
    head, tail = os.path.split(path)
    suffix = os.path.splitext(tail)[1]
    new_path = os.path.join(head, f"%06d{suffix}" % id)
    print(f"rename {path} to {new_path}")
    os.rename(path, new_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to the dataset')
    parser.add_argument('--start', type=int, default=0, help='start number to use')
    args = parser.parse_args()
    
    formats = ['*.jpg', '*.jpeg', '*.png', '*.tif']
    
    image_paths = []
    for p in args.path.split(','):
        for f in formats:
            image_paths += list(sorted(glob.glob(os.path.join(p, f))))
    
    for i, ipath in enumerate(image_paths):
        rename_file(ipath, i + args.start)
        root, ext = os.path.splitext(ipath)
        lpath = root + '.xml'
        rename_file(lpath, i + args.start)