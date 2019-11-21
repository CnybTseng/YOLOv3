# -*- coding: utf-8 -*-
# file: utils.py
# brief: YOLOv3 implementation based on PyTorch
# author: Zeng Zhiwei
# date: 2019/9/12

import os
import cv2
import torch
import numpy as np
import copy

class TrainScaleSampler(object):
    def __init__(self, scale_step=(320,608,10), rescale_freq=320):
        self.scale_step = scale_step
        self.rescale_freq = rescale_freq
        self.size = [416,416]
    
    def __call__(self, num_batches=0):
        if num_batches % self.rescale_freq == 0:
            sizes = np.linspace(start=self.scale_step[0], stop=self.scale_step[1], num=self.scale_step[2], dtype=np.int32)
            rand_size = sizes[np.random.randint(len(sizes))]
            self.size = [rand_size.item(), rand_size.item()]
            # with open(os.path.join('log', 'TrainScaleSampler.txt'), 'a') as file:
            #     file.write(f'update scale to {self.size}/{num_batches}\n')
        return self.size

def print_training_message(epoch, msgs, batch_size):
    with open('log/verbose.txt', 'a') as file:
        losses = 0
        num_batches = len(msgs)
        num_yolo = len(msgs[0][1])
        for loss, metrics in msgs:
            for metric in metrics:
                for k, v in metric.items():
                    if isinstance(v, int): file.write(f'{k}:{v} ')
                    else : file.write(f'{k}:%.5f ' % v)
                file.write('\n')
            losses += loss
            file.write('\n')
        losses /= (num_batches * num_yolo * batch_size)
        file.write(f'Epoch {epoch} done, total losses:%.5f\n' % losses)
        print(f'Epoch {epoch} done, total losses:%.5f' % losses)
        file.close()
    
    with open('log/loss.txt', 'a') as file:
        file.write(f'{losses}\n')
        file.close()

def cal_av_loss(msgs, batch_size):
    losses = 0
    num_batches = len(msgs)
    num_yolo = len(msgs[0][1])
    for loss, metrics in msgs:
        losses += loss
    losses /= (num_batches * num_yolo * batch_size)
    return losses

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def lr_lambda(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)      

def load_class_names(path):
    class_names = []
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            name = line.rstrip('\n')
            if not name: continue
            class_names.append(name)
        file.close()
    return class_names

def make_letterbox_image(im, in_size=(416,416)):
    fx = in_size[1] / im.shape[1]
    fy = in_size[0] / im.shape[0]
    s = fx if fx < fy else fy

    rz_im = cv2.resize(im, (0, 0), fx=s, fy=s).astype(float) / 255
    
    lb_im = np.full((in_size[0],in_size[1],3), 0.5, dtype=np.float)
    dx = np.int32((lb_im.shape[1] - rz_im.shape[1])/2)
    dy = np.int32((lb_im.shape[0] - rz_im.shape[0])/2)
    lb_im[dy:dy+rz_im.shape[0],dx:dx+rz_im.shape[1],:] = rz_im[:,:,:]

    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    norm_im = torch.from_numpy(lb_im).type(dtype=FloatTensor)
    norm_im = torch.unsqueeze(norm_im, 0)
    x = norm_im.permute(0, 3, 1, 2).contiguous()

    return x

def xyxy_to_xywh(minx, miny, maxx, maxy, org_size):
    lb_size = max(org_size)
    dx = (lb_size - org_size[1])/2
    dy = (lb_size - org_size[0])/2

    lb_minx = minx + dx
    lb_miny = miny + dy
    lb_maxx = maxx + dx
    lb_maxy = maxy + dy
    
    bx = ((lb_minx + lb_maxx)/2) / lb_size
    by = ((lb_miny + lb_maxy)/2) / lb_size
    bw = (lb_maxx - lb_minx) / lb_size
    bh = (lb_maxy - lb_miny) / lb_size
    
    return bx, by, bw, bh

def xywh_to_xyxy(box):
    xmin = box[0] - box[2]/2
    ymin = box[1] - box[3]/2
    xmax = box[0] + box[2]/2
    ymax = box[1] + box[3]/2
    
    return xmin, ymin, xmax, ymax

def get_network_boxes(prediction, im_size=(1,1), thresh=0.5):
    '''
    参数
    ----
    prediction : Tensor
        YOLO层输出的预测值.预测值大小为1*#cells*(4+1+#classes).
        #cells = (3*cellw*cellh)*#YOLOs.
    im_size : tuple or list
        im_size=(height,width)
    thresh : float
        置信度和类别概率阈值.
    '''
    
    prediction = prediction.squeeze()
    square_size = max(im_size)
    prediction = prediction[prediction[:,4] > thresh,:]
    
    detection = {'bbox':list(), 'objectness':list(), 'prob':list()}
    for pred in prediction:
        if square_size > 1:
            x, y, w, h = pred[:4] * square_size
            
            if im_size[0] == square_size:
                x = x - int((square_size - im_size[1])/2)
            else:
                y = y - int((square_size - im_size[0])/2)
            
            minx = int(x - w/2)
            miny = int(y - h/2)
            maxx = int(x + w/2)
            maxy = int(y + h/2)
            
            minx = max(0, minx)
            miny = max(0, miny)
            maxx = min(im_size[1]-1, maxx)
            maxy = min(im_size[0]-1, maxy)
        else:
            minx, miny, maxx, maxy = xywh_to_xyxy(pred[:4])
        
        bbox = (minx, miny, maxx, maxy)
        objectness = pred[4].numpy()
        prob = pred[4] * pred[5:]
        prob[prob < thresh] = 0
        prob = prob.numpy()
        detection['bbox'].append(bbox)
        detection['objectness'].append(objectness)
        detection['prob'].append(prob)
    
    if len(detection['bbox']):
        detection['bbox'] = np.stack(detection['bbox'])
        detection['objectness'] = np.stack(detection['objectness'])
        detection['prob'] = np.stack(detection['prob'])
    
    return detection

def cal_iou(box1, box2, eps=1e-16):
    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])
    
    inter_w = max(0, xmax - xmin)
    inter_h = max(0, ymax - ymin)
    inter_a = inter_w * inter_h
    
    box1_a = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_a = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return inter_a / (box1_a + box2_a - inter_a + eps)

def nms_sort(detection, thresh=0.45):
    if len(detection['bbox']) == 0:
        return detection
    
    bbox = detection['bbox']
    objectness = detection['objectness']
    prob = detection['prob']
    num_classes = prob.shape[1]
    num_dets = prob.shape[0]

    for c in range(num_classes):
        sorted_indices = np.argsort(prob[:,c])[::-1]
        bbox = bbox[sorted_indices,:]
        objectness = objectness[sorted_indices]
        prob = prob[sorted_indices,:]

        for i in range(num_dets):
            if prob[i,c] == 0:
                continue
            
            for j in range(i+1, num_dets):
                iou = cal_iou(bbox[i,:], bbox[j,:])
                if iou > thresh:
                    prob[j,c] = 0
    
    return {'bbox':bbox, 'objectness':objectness, 'prob':prob}

def nms_obj(detection, thresh=0.45):
    if len(detection['bbox']) == 0:
        return detection
    
    bbox = detection['bbox']
    objectness = detection['objectness']
    prob = detection['prob']
    num_classes = prob.shape[1]
    num_dets = prob.shape[0]

    sorted_indices = np.argsort(objectness)[::-1]
    bbox = bbox[sorted_indices,:]
    objectness = objectness[sorted_indices]
    prob = prob[sorted_indices,:]

    for i in range(num_dets):
        if objectness[i] == 0:
            continue
        
        class1 = np.argmax(prob[i])
        for j in range(i+1, num_dets):
            class2 = np.argmax(prob[j])
            if objectness[j] == 0 or class1 != class2:
                continue
            
            iou = cal_iou(bbox[i,:], bbox[j,:])
            if iou > thresh:
                objectness[j] = 0
                prob[j,:] = 0
    
    return {'bbox':bbox, 'objectness':objectness, 'prob':prob}

def overlap_detection(im, detection, class_names):
    im = copy.deepcopy(im)
    bbox = detection['bbox']
    objectness = detection['objectness']
    prob = detection['prob']
    
    for det in zip(bbox, objectness, prob):                       
        if det[1] == 0 or np.max(det[2]) == 0:
            continue

        # 获取字符串尺寸
        id = np.argmax(det[2])
        [size, baseline] = cv2.getTextSize(text=class_names[id],\
            fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, thickness=1)
        
        # 字符串水平位置限界
        minx, miny, maxx, maxy = det[0]
        tw, th = size[0], size[1]
        tx = minx if minx >= 0 else 0
        tx = tx if tx < im.shape[1]-tw else im.shape[1]-tw-1
        
        # 字符串垂直位置限界.
        # 字符串的高度等于getTextSize返回的高度加上baseline
        ty = miny-baseline if miny-baseline >= th else th 
        ty = ty if ty < im.shape[0]-baseline else im.shape[0]-baseline-1
        
        # 叠加类别标签
        cv2.rectangle(img=im, pt1=(tx,ty-th), pt2=(tx+tw-1,ty+baseline),\
            color=(0,255,255), thickness=-1)
        cv2.putText(img=im, text=class_names[id], org=(tx,ty),\
            fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1,\
            color=(0,0,255), thickness=1)
        cv2.rectangle(img=im, pt1=(minx,miny), pt2=(maxx,maxy),\
            color=(0,255,255), thickness=1)

    return im

def save_tensor(tensor, filename):
    data = tensor.flatten().detach().cpu().numpy()
    np.savetxt(filename, data, delimiter=' ', fmt='%.6f')

def __prediction_and_truth_to_list(outputs, ground_truth, num_classes):
    pred_objs = list()
    pred_boxes = list()
    pred_classes = list()
    truth_indices = list()
    truth_boxes = list()
    truth_classes = list()
    num_preds = [0] * num_classes
    num_truths = [0] * num_classes

    for i, (output, target) in enumerate(zip(outputs, ground_truth)):
        # 统计检测结果中的各类物体
        for det in zip(output['bbox'], output['objectness'], output['prob']):
            if det[1] == 0 or np.max(det[2]) == 0:
                continue

            pred_objs.append(det[1])
            pred_boxes.append(list(det[0]))
            class_id = np.argmax(det[2])
            pred_classes.append(class_id)
            truth_indices.append(i)
            num_preds[class_id] += 1
        
        # 统计事实的各类物体
        boxes = target[0]['boxes'].detach().cpu().numpy().tolist()
        labels = target[0]['labels'].detach().cpu().numpy().tolist()
        truth_boxes.append(boxes)
        truth_classes.append(labels)
        for label in labels:
            num_truths[label] += 1
    
    return pred_objs, pred_boxes, pred_classes, truth_indices, truth_boxes, \
        truth_classes, num_preds, num_truths

def __calc_TP_and_FP(pred_boxes, pred_classes, truth_indices, truth_boxes,
    truth_classes, num_preds, num_classes):
    
    TP = list()     # Truth Positive
    FP = list()     # False Positive
    for c in range(num_classes):
        TP.append([0] * num_preds[c])
        FP.append([0] * num_preds[c])
    
    # 事实物体是否分配的掩模
    truth_mask = list()
    for t in range(len(truth_classes)):
        truth_mask.append([0] * len(truth_classes[t]))

    nobjs = [0] * num_classes   # 各类的计数器
    for i, (pb, pc) in enumerate(zip(pred_boxes, pred_classes)):
        t = truth_indices[i]  
        best_iou = 0
        resp_idx = 0
        
        # 寻找重叠率最大的事实物体
        for j, (tb, tc) in enumerate(zip(truth_boxes[t], truth_classes[t])):
            if pc != tc: continue
            iou = cal_iou(pb, xywh_to_xyxy(tb))
            if iou > best_iou:
                best_iou = iou
                resp_idx = j

        # 真阳: 存在未分配的重叠率够大的事实物体
        if best_iou >= 0.5 and truth_mask[t][resp_idx] == 0:
            TP[pc][nobjs[pc]] = 1
            truth_mask[t][resp_idx] = 1
        else:
            FP[pc][nobjs[pc]] = 1
        
        # 更新对应类别的计数器
        nobjs[pc] += 1
    
    # 真阳和假阳的数值积分
    for c in range(num_classes):
        accum = 0
        for idx, val in enumerate(TP[c]):
            TP[c][idx] += accum
            accum += val

        accum = 0
        for idx, val in enumerate(FP[c]):
            FP[c][idx] += accum
            accum += val

    return TP, FP

def __calc_AP(TP, FP, num_classes, num_truths):
    # 计算精确率和召回率
    recall = copy.deepcopy(TP)
    precision = copy.deepcopy(TP)
    for c in range(num_classes):
        for idx, val in enumerate(TP[c]):
            recall[c][idx] = TP[c][idx] / num_truths[c]

        for idx, val in enumerate(TP[c]):
            precision[c][idx] = TP[c][idx] / (TP[c][idx] + FP[c][idx])

    # 计算平均精确率
    measured_recall = copy.deepcopy(recall)
    measured_precision = copy.deepcopy(precision)
    AP = [0] * num_classes
    for c in range(num_classes):
        measured_precision[c].insert(0, 0.0)
        measured_precision[c].append(0.0)
        
        # 将精确率折线做单调下降处理
        for i in range(len(measured_precision[c])-2, -1, -1):
            measured_precision[c][i] = max(measured_precision[c][i],
                measured_precision[c][i+1])
        
        measured_recall[c].insert(0, 0.0)
        measured_recall[c].append(1.0)
        inflection_points = list()
        
        # 寻找召回率折线的拐点
        for i in range(1, len(measured_recall[c])):
            if measured_recall[c][i] != measured_recall[c][i-1]:
                inflection_points.append(i)
        
        # 计算平均精确率
        for i in inflection_points:
            w = measured_recall[c][i] - measured_recall[c][i-1]
            AP[c] += (measured_precision[c][i] * w)
    
    return AP, recall, precision

def __prediction_truth_to_file(pred_objs, pred_boxes, pred_classes, \
    truth_indices, truth_boxes, truth_classes):
    for i, (po, pb, pc) in enumerate(zip(pred_objs, pred_boxes, pred_classes)):
        with open(f'../mAP/input/detection-results/%0004d.txt' % truth_indices[i], 'a') as file:
            file.write(f'{pc} {po} {pb[0]} {pb[1]} {pb[2]} {pb[3]}\n')
            file.close()
    
    for i, (tbs, tcs) in enumerate(zip(truth_boxes, truth_classes)):
        with open(f'../mAP/input/ground-truth/%0004d.txt' % i, 'w') as file:
            for (tb, tc) in zip(tbs, tcs):
                tb = xywh_to_xyxy(tb)
                file.write(f'{tc} {tb[0]} {tb[1]} {tb[2]} {tb[3]}\n')
            file.close()

def calc_mAP(outputs, ground_truth, num_classes, debug=False):    
    pred_objs, \
    pred_boxes, \
    pred_classes, \
    truth_indices, \
    truth_boxes, \
    truth_classes, \
    num_preds, \
    num_truths = __prediction_and_truth_to_list(outputs, ground_truth, num_classes)

    if debug:
        __prediction_truth_to_file(pred_objs, pred_boxes, pred_classes, truth_indices, \
            truth_boxes, truth_classes)

    if len(pred_objs) == 0:
        return 0
    
    sorted_indices = np.argsort(pred_objs)[::-1]
    pred_objs = np.array(pred_objs)[sorted_indices]
    pred_boxes = np.array(pred_boxes)[sorted_indices,:]
    pred_classes = np.array(pred_classes)[sorted_indices]
    truth_indices = np.array(truth_indices)[sorted_indices]

    TP, FP = __calc_TP_and_FP(pred_boxes, pred_classes, truth_indices, truth_boxes,
        truth_classes, num_preds, num_classes)

    AP, recall, precision = __calc_AP(TP, FP, num_classes, num_truths)
    mAP = sum(AP)/num_classes
    
    if debug:
        print(f'AP={AP}')
        print(f'mAP={mAP}')
        with open('myresult.txt', 'w') as file:
            file.write('# AP and precision/recall per class\n')
            for i in range(num_classes):
                rounded_prec = ['%.2f' % elem for elem in precision[i]]
                rounded_rec = ['%.2f' % elem for elem in recall[i]]
                file.write(f'%.2f%% = {i} AP\n' % (AP[i] * 100))
                file.write(f' Precision: {rounded_prec}\n')
                file.write(f' Recall :{rounded_rec}\n\n')
            file.write('\n# mAP of all classes\n')
            file.write(f'mAP = %.2f%%\n\n' % (mAP * 100))
            file.write('# Number of ground-truth objects per class\n')
            for i in range(num_classes):
                file.write(f'{i}: {num_truths[i]}\n')
            file.write('\n# Number of detected objects per class\n')
            for i in range(num_classes):
                file.write(f'{i}: {num_preds[i]} (tp:{TP[i][-1]}, fp:{FP[i][-1]})\n')
    
    return mAP