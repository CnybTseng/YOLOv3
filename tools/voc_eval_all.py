from voc_eval import voc_eval

import os

current_path = '/home/image/tseng/project/darknet'
results_path = current_path+"/results"
sub_files = os.listdir(results_path)

mAP = []
for i in range(len(sub_files)):
    class_name = sub_files[i].split(".txt")[0]
    detpath = '/home/image/tseng/project/darknet/results/{}.txt'
    annopath = '/home/image/tseng/dataset/VOCdevkit/VOC2007/Annotations/{}.xml'
    imagesetfile = '/home/image/tseng/dataset/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
    rec, prec, ap = voc_eval(detpath, annopath, imagesetfile, class_name, '.')
    print("{} :\t {} ".format(class_name, ap))
    mAP.append(ap)

mAP = tuple(mAP)

print("***************************")
print("mAP :\t {}".format( float( sum(mAP)/len(mAP)) ))