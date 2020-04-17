import os
import glob
import argparse

def main(args):
    xml_paths = glob.glob(os.path.join(args.path, '*.xml'))
    img_paths = []
    for f in ['*.jpg', '*.JPG', '*.png', '*.tif']:
        img_paths += glob.glob(os.path.join(args.path, f))
    for img_path in img_paths:
        filename, extension = os.path.splitext(img_path)
        xml_path = filename + '.xml'
        if xml_path not in xml_paths:
            print(f"find missing xml: {xml_path}")
            with open(xml_path, 'w') as file:
                file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to the annotations')
    args = parser.parse_args()
    main(args)