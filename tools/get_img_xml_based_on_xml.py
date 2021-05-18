import os
import glob
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-path', type=str, help='path to the source data')
    parser.add_argument('--dst-path', type=str, help='path to the destination data')
    args = parser.parse_args()
    
    if not os.path.exists(args.dst_path):
        os.makedirs(args.dst_path)
    
    paths = list(sorted(glob.glob(os.path.join(args.src_path, '*.*'))))
    
    xml_names = []
    img_paths = []
    for path in paths:
        root, ext = os.path.splitext(path)
        if ext == '.xml':
            xml_names.append(root)
        else:
            img_paths.append(path)
    
    for xml_name in xml_names:
        for img_path in img_paths:
            if xml_name in img_path:
                os.system(f"copy {xml_name}.xml {args.dst_path}")
                os.system(f"copy {img_path} {args.dst_path}")
                break
        else:
            print(f"not found expected image:{xml_name}")