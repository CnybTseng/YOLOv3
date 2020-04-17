import os
import re
import argparse

def main(args):
    paths = open(args.path).read().split()
    paths = paths[0::2]
    with open("imagesetfile.txt", 'w') as file:
        for path in paths:
            segments = re.split(r'[\\,/,.]', path)
            file.write(f'{segments[-2]}\n')
        file.close()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to the dataset file')
    args = parser.parse_args()
    main(args)