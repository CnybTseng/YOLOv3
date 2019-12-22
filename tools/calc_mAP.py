import os
import argparse
from voc_eval import voc_eval

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detpath', help='Path to detections')
    parser.add_argument('--annopath', help='Path to annotations')
    parser.add_argument('--imagesetfile', help='Text file containing the list of images, one image per line')
    parser.add_argument('--ovthresh', type=float, default=0.45, help='Overlap threshold [0.45]')
    parser.add_argument('--use_07_metric', help="Whether to use VOC07's 11 point AP computation [False]", action='store_true')
    args = parser.parse_args()
    print(args)
    
    dets = os.listdir(args.detpath)
    detpath = os.path.join(args.detpath, 'comp4_det_test_{}.txt')
    annopath = os.path.join(args.annopath, '{}.xml')
    APs = []
    for det in dets:
        classname = det.split('.')[0].split('_')[-1]
        rec, prec, ap = voc_eval(detpath, annopath, args.imagesetfile, classname, '.', args.ovthresh, args.use_07_metric)
        APs.append(ap)
        print("AP of {} is {}".format(classname, ap))
    print("mAP is {}".format(sum(APs)/len(APs)))