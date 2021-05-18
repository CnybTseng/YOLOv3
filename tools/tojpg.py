import os
import cv2
import glob
import shutil
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to the images')
    parser.add_argument('--backup-dir', type=str, default='', help='path to the backup images')
    args = parser.parse_args()

    for format in ['*.png', '*.jpg', '*.jpeg']:
        paths = list(sorted(glob.glob(os.path.join(args.path, format))))
        for path in paths:
            print('{}'.format(path))
            im = cv2.imread(path)
            root, ext = os.path.splitext(path)
            
            if os.path.exists(args.backup_dir):
                shutil.move(path, args.backup_dir)
            else:
                os.remove(path)
            
            cv2.imwrite(root + '.jpg', im)