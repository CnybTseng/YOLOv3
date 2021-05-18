import os
import shutil
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', help='root directory')
    args = parser.parse_args()
    
    fullpaths = []
    for dirpath, dirnames, filenames in os.walk(args.root_dir):
        if dirnames:
            continue
        for filename in filenames:
            fullpath = os.path.join(dirpath, filename)
            fullpaths.append(fullpath)

    print(f"total number of files is {len(fullpaths)}")