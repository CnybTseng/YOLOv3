# -*- coding: utf-8 -*-
# file: yolo.py
# brief: YOLOv3 implementation based on PyTorch
# author: Zeng Zhiwei
# date: 2019/7/12

from __future__ import print_function
import torch
import numpy as np

class Yolo(torch.nn.Module):
    '''目标检测层
    '''
    
    def __init__(self, num_classes, anchors, anchor_mask, in_size):
        '''初始化目标检测层.
        
        参数
        ----
        num_classes : int
            类别数量.
        anchors : tuple
            锚框参数.参数格式为((w1,h1),(w2,h2),...,(wn,hn)),n为锚框的个数.
        in_size : tuple
            神经网络输入图像的尺寸.参数格式为(h,w).
        '''
        
        super(Yolo, self).__init__()
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')
        if int(torch.__version__.replace('.', '').split('+')[0]) >= 120:
            self.BoolTensor = torch.cuda.BoolTensor if self.cuda else torch.BoolTensor
        else:
            self.BoolTensor = torch.cuda.ByteTensor if self.cuda else torch.ByteTensor
        self.LongTensor = torch.cuda.LongTensor if self.cuda else torch.LongTensor
        self.FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        self.num_classes = num_classes
        self.anchors = torch.tensor(anchors).type(self.FloatTensor)
        self.anchor_mask = torch.tensor(anchor_mask).type(self.LongTensor)
        self.num_anchors = len(anchor_mask)
        self.in_size = in_size
        self.ignore_thresh = 0.5
            
    def __cal_bounding_box(self, px, py, pw, ph):
        # 网格大小: 13x13或26x26.
        [gy, gx] = torch.meshgrid(torch.arange(py.size(2)), torch.arange(px.size(3)))
        
        # 网格整形.整形之后大小为: 1x1x13x13或1x1x26x26
        gy = gy.view(1, 1, gy.size(0), gy.size(1)).type(self.FloatTensor)
        gx = gx.view(1, 1, gx.size(0), gx.size(1)).type(self.FloatTensor)
        
        # 计算x,y的归一化值.张量大小: 1x3x13x13或1x3x26x26
        npy = (py.data + gy) / py.size(2)   # 相对网格的归一化纵坐标
        npx = (px.data + gx) / px.size(3)   # 相对网格的归一化横坐标

        # 锚框尺寸张量整形.整形后的大小为: 1x3x1x1
        achr_w = self.anchors[self.anchor_mask, 0].view(1, self.num_anchors, 1, 1)
        achr_h = self.anchors[self.anchor_mask, 1].view(1, self.num_anchors, 1, 1)
        
        # 计算w,h的归一化值.张量大小为: 1x3x13x13或1x3x26x26
        npw = torch.exp(pw.data) * achr_w / self.in_size[1]
        nph = torch.exp(ph.data) * achr_h / self.in_size[0]
        
        # 打包边界框.张量大小为: 1x3x13x13x4或1x3x26x26x4
        bbox = self.FloatTensor(npx.size(0), npx.size(1), npx.size(2), npx.size(3), 4)
        bbox[..., 0] = npx
        bbox[..., 1] = npy
        bbox[..., 2] = npw
        bbox[..., 3] = nph
        
        return bbox
    
    def __xywh_to_xyxy(self, xywh):
        '''转换边界框的描述形式.
        
        参数
        ----
        xywh : Tensor
            以中心坐标,宽,高形式描述的边界框.
            xywh = torch.tensor([[x1,y1,w1,h1],[x2,y2,w2,h2],...,[xn,yn,wn,yn]])
        ----
        返回: 以最左,上,右,下坐标方式描述的边界框.
            xyxy = torch.tensor([[xmin1,ymin1,xmax1,ymax1],[xmin2,ymin2,xmax2,ymax2],
            ...,[xminn,yminn,xmaxn,ymaxn]])
        '''
        
        xmin = xywh[:,0] - xywh[:,2]/2
        xmax = xywh[:,0] + xywh[:,2]/2
        ymin = xywh[:,1] - xywh[:,3]/2
        ymax = xywh[:,1] + xywh[:,3]/2
        
        return torch.stack((xmin,ymin,xmax,ymax), dim=1)
    
    def __cal_intersection_area(self, box1, box2):
        '''计算两个边界框的重叠面积.
        
        参数
        ----
        box1 : Tensor
            边界框.box1=Tensor([minx,miny,maxx,maxy]),或者box1=Tensor(
            [xmin1,xmin2,...,xminn],[ymin1,ymin2,..,yminn],[xmax1,xmax2,...,xmaxn],
            [ymax1,ymax2,...,ymaxn])
        box2 : Tensor
            边界框.box2=Tensor([minx,miny,maxx,maxy]),或者box2=Tensor(
            [xmin1,xmin2,...,xminn],[ymin1,ymin2,..,yminn],[xmax1,xmax2,...,xmaxn],
            [ymax1,ymax2,...,ymaxn])
        '''
        
        minx = torch.max(box1[0], box2[0])
        miny = torch.max(box1[1], box2[1])
        maxx = torch.min(box1[2], box2[2])
        maxy = torch.min(box1[3], box2[3])
        w = torch.max(maxx-minx, self.FloatTensor([0]))
        h = torch.max(maxy-miny, self.FloatTensor([0]))        
        
        return w * h
    
    def __cal_IoU(self, box1, box2, eps=1e-16):
        '''计算两个边界框的重叠率.两个边界框中心对齐.
        
        参数
        ----
        box1 : Tensor
            边界框.box1=Tensor([minx,miny,maxx,maxy]),或者box1=Tensor(
            [xmin1,xmin2,...,xminn],[ymin1,ymin2,..,yminn],[xmax1,xmax2,...,xmaxn],
            [ymax1,ymax2,...,ymaxn])
        box2 : Tensor
            边界框.box2=Tensor([minx,miny,maxx,maxy]),或者box2=Tensor(
            [xmin1,xmin2,...,xminn],[ymin1,ymin2,..,yminn],[xmax1,xmax2,...,xmaxn],
            [ymax1,ymax2,...,ymaxn])
        '''

        intersection = self.__cal_intersection_area(box1, box2)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / (area1 + area2 - intersection + eps)
    
    def __cal_IoU_lrca(self, box1, box2, eps=1e-16):
        '''计算两个边界框的重叠率.两个边界框左下角对齐.
        
        参数
        ----
        box1 : Tensor
            边界框.box1=Tensor([w,h])或box1=Tensor([w1,w2,...,wn],[h1,h2,...,hn])
        box2 : Tensor
            边界框.box2=Tensor([w,h])或box1=Tensor([w1,w2,...,wn],[h1,h2,...,hn])
        '''

        intersection = torch.min(box1[0], box2[0]) * torch.min(box1[1], box2[1])
        area1 = box1[0] * box1[1]
        area2 = box2[0] * box2[1]
        
        return intersection / (area1 + area2 - intersection + eps)
    
    def __merge_truth_prob(self, cellx_indices, celly_indices, truth_pro):
        '''融合落在同一细胞内的类别概率期望值.
        
        参数
        ----
        cellx_indices : Tensor
            细胞水平坐标索引.
        celly_indices : Tensor
            细胞垂直坐标索引.
        truth_pro : Tensor
            类别概率期望值.
        '''
        
        assert cellx_indices.size() == celly_indices.size()
        cell_indices = torch.stack((cellx_indices, celly_indices), dim=0).type(self.LongTensor)

        num_cells = cell_indices.size(1)
        for i in range(num_cells):
            for j in range(i+1, num_cells):
                if torch.equal(cell_indices[:,i], cell_indices[:,j]):
                    mask_i = truth_pro[i,:] == 1
                    mask_j = truth_pro[j,:] == 1
                    truth_pro[i, mask_j] = 1
                    truth_pro[j, mask_i] = 1
                    # with open('log/merge_truth_prob.txt', 'a') as file:
                    #     file.write(f'{cell_indices[:,i]} == {cell_indices[:,j]}\n')
                    #     file.write(f'mask_i={mask_i}, mask_j={mask_j}\n')
                    #     file.write(f'truth_pro_i:{truth_pro[i,:]}, truth_pro_j={truth_pro[j,:]}\n')
                    #     file.close()

        return truth_pro
    
    def __parse_truth_data(self, target, grid_size):
        '''解析真实数据.
        
        参数
        ----
        target : tuple of dict
            目标值.包括真框和类别标签.
            target=({'boxes':boxes1,'labels':labels1},{'boxes':boxes2,'labels':labels2},
            ...,{'boxes':boxesn,'labels':labelsn}),n为批量大小.
        grid_size : Tensor
            网格尺寸.
        ----
        返回: 物体序号索引,锚框索引,细胞坐标索引,类别索引,以及物体坐标,
            尺度,置信度,类别概率,边界框的期望预测值.
        '''
        
        batch_indices = self.LongTensor([])
        anchor_indices = self.LongTensor([])
        celly_indices = self.LongTensor([])
        cellx_indices = self.LongTensor([])
        class_indices = self.LongTensor([])
        truth_x = self.FloatTensor([])
        truth_y = self.FloatTensor([])
        truth_w = self.FloatTensor([])
        truth_h = self.FloatTensor([])
        truth_obj = self.FloatTensor([])
        truth_pro = self.FloatTensor([])
        truth_box = self.FloatTensor([])
        truth_box_all = self.FloatTensor([])
        anchors = self.anchors / self.in_size[0]
        
        for (b, data) in enumerate(target):
            boxes = data['boxes']
            labels = data['labels'].type(self.LongTensor)
            if boxes.numel() == 0 or labels.numel() == 0: continue
            truth_box_all = torch.cat((truth_box_all, boxes))

            # 在所有锚框中搜索和真框最匹配的锚框
            ious = torch.stack([self.__cal_IoU_lrca(box, boxes[:,2:].t()) for box in anchors])
            max_ious, resp_ids = ious.max(dim=0)
            
            # print(f'boxes={boxes}')
            # print(f'labels={labels}')
            # print(f'ious={ious}')
            # print(f'max_ious={max_ious}')
            # print(f'resp_ids={resp_ids} at grid_size {grid_size}')
            
            # 搜索最匹配的锚框属于本层的物体
            resp_objs = [i for i in range(len(resp_ids)) \
                if len((self.anchor_mask==resp_ids[i]).nonzero())]
            num = len(resp_objs)
            if num == 0: continue
            
            # print(f'resp_objs={resp_objs} at grid_size {grid_size}')
            
            # 排除不由本层负责预测的真框
            resp_objs = torch.tensor(resp_objs).type(self.LongTensor)
            boxes = boxes[resp_objs, :]
            labels = labels[resp_objs]
            ious = ious[:, resp_objs]
            resp_ids = resp_ids[resp_objs]
            
            # print(f'resp boxes={boxes}')
            # print(f'resp labels={labels}')
            # print(f'resp ious={ious}')
            # print(f'resp resp_ids={resp_ids}')

            # 批号索引,一张图里所有目标具有一样的批号
            batch_indices = torch.cat((batch_indices, self.LongTensor(num).fill_(b)))
            
            # print(f'batch_indices={batch_indices}')
            
            # 本层和真框最匹配的锚框索引
            anchor_indices = torch.cat((anchor_indices, resp_ids % self.num_anchors))
            
            # print(f'anchor_indices={anchor_indices}')
            
            # 细胞索引
            bx = boxes[:,0] * grid_size
            by = boxes[:,1] * grid_size
            cellx = bx.floor()
            celly = by.floor()
            cellx_indices = torch.cat((cellx_indices, cellx.type(self.LongTensor)))
            celly_indices = torch.cat((celly_indices, celly.type(self.LongTensor)))
            
            # 类别索引
            class_indices = torch.cat((class_indices, labels))
            
            # 真框位置的期望值
            truth_x = torch.cat((truth_x, bx - cellx))
            truth_y = torch.cat((truth_y, by - celly))
            
            # 真框尺度的期望值
            truth_w = torch.cat((truth_w, torch.log(boxes[:,2] / anchors[resp_ids,0])))
            truth_h = torch.cat((truth_h, torch.log(boxes[:,3] / anchors[resp_ids,1])))
            
            # 物体置信度的期望值
            truth_obj = torch.cat((truth_obj, self.FloatTensor(num).fill_(1)))
            
            # print(f'truth_obj={truth_obj}')
            
            # 物体类别概率的期望值
            label_mask = self.FloatTensor(num, self.num_classes).fill_(0)
            label_mask[torch.arange(num, dtype=torch.int64, device=self.device), labels] = 1
            label_mask = self.__merge_truth_prob(cellx, celly, label_mask)
            truth_pro = torch.cat((truth_pro, label_mask))
            
            # print(f'truth_pro={truth_pro}')
            
            # 本层负责处理的真框
            truth_box = torch.cat((truth_box, boxes))

        return ( \
            batch_indices,\
            anchor_indices,\
            celly_indices,\
            cellx_indices,\
            class_indices,\
            truth_x,\
            truth_y,\
            truth_w,\
            truth_h,\
            truth_obj,\
            truth_pro,\
            truth_box,\
            truth_box_all)

    def __cal_ignore_mask(self, truth_boxes, pred_boxes):
        '''计算不需要参与背景损失计算的掩膜.
        
        参数
        ----
        truth_boxes : Tensor
            真实框.
        pred_boxes : Tensor
            预测框.预测框张量大小为: Nx3x13x13x4.
        ----
        返回: 不需要参与背景损失计算的掩膜.
        '''

        size = pred_boxes.size()
        if truth_boxes.numel() == 0:
            return self.BoolTensor(size[0], size[1], size[2], size[3]).fill_(0)
        
        truth_boxes = self.__xywh_to_xyxy(truth_boxes)
        pred_boxes = self.__xywh_to_xyxy(pred_boxes.view(-1, 4))
        ious = torch.stack([self.__cal_IoU(box, pred_boxes.t()) for box in truth_boxes])
        best_ious, resp_ids = ious.max(dim=0)
        ignore_mask = (best_ious > self.ignore_thresh).view(size[:4])
        
        # print(f'truth_boxes.size={truth_boxes.size()}')
        # print(f'pred_boxes.size={size}, view size={pred_boxes.size()}')
        # print(f'ious.size={ious.size()}')
        # print(f'best_ious.size={best_ious.size()}')
        # print(f'ignore_mask={ignore_mask.size()}')
        # print(f'ignore_mask 1s = {ignore_mask.sum()}')
        # print(f'ignore_mask percentage = {ignore_mask.sum()/ignore_mask.numel()}')
        
        return ignore_mask

    def __safe_mse_loss(self, input, target):
        assert input.numel() == target.numel()
        if input.numel() == 0 or target.numel() == 0:
            return self.FloatTensor([0]).requires_grad_()
        return torch.nn.MSELoss(reduction='sum')(input, target)
    
    def __safe_bce_loss(self, input, target, weight=None):
        assert input.numel() == target.numel()
        if input.numel() == 0 or target.numel() == 0:
            return self.FloatTensor([0]).requires_grad_()
        return torch.nn.BCELoss(weight=weight, reduction='sum')(input, target)
    
    def __cal_loss(self, prediction, target, bbox):
        '''计算损失.
        
        参数
        ----
        prediction : List of Tensor
            预测值.包括边界框(x,y,w,h),物体置信度和类别概率.
        target : tuple of dict
            目标值.包括边界框(x,y,w,h)和类别标签.
        bbox : Tensor
            神经网络输出的归一化的物体边界框.
        ----
        返回: 损失和指标信息.
        '''
        
        pxywh, objectness, probabilities = prediction
        
        n = probabilities.size(0)   # 批大小
        a = probabilities.size(1)   # 锚框个数
        h = probabilities.size(2)   # 特征图高度
        w = probabilities.size(3)   # 特征图宽度
        c = probabilities.size(4)   # 类别数量
        
        # 解析物体的真实信息.
        batch_indices,\
        anchor_indices,\
        celly_indices,\
        cellx_indices,\
        class_indices,\
        truth_x,\
        truth_y,\
        truth_w,\
        truth_h,\
        truth_obj,\
        truth_pro,\
        truth_box,\
        truth_box_all = self.__parse_truth_data(target, w)

        # 背景掩膜
        bkg_mask = self.BoolTensor(n, a, h, w).fill_(1)
        bkg_mask[batch_indices, anchor_indices, celly_indices, cellx_indices] = 0
        ignore_mask = self.__cal_ignore_mask(truth_box_all, bbox)
        bkg_mask[ignore_mask] = 0
        
        # 背景置信度的期望值和预测值
        truth_bkg = self.FloatTensor(n, a, h, w).fill_(0)[bkg_mask]
        pred_bkg = objectness[bkg_mask]
        
        # print(f'batch_indices={batch_indices}')
        # print(f'anchor_indices={anchor_indices}')
        # print(f'celly_indices={celly_indices}')
        # print(f'cellx_indices={cellx_indices}')
        # print(f'class_indices={class_indices}')
        # print(f'truth_x={truth_x}')
        # print(f'truth_y={truth_y}')
        # print(f'truth_w={truth_w}')
        # print(f'truth_h={truth_h}')
        # print(f'truth_obj={truth_obj}')
        # print(f'truth_pro={truth_pro}')
        # print(f'truth_box={truth_box}')
        
        # 获取物体坐标,尺度,置信度和类别概率的预测值
        pred_x = pxywh[batch_indices, anchor_indices, celly_indices, cellx_indices, 0]
        pred_y = pxywh[batch_indices, anchor_indices, celly_indices, cellx_indices, 1]
        pred_w = pxywh[batch_indices, anchor_indices, celly_indices, cellx_indices, 2]
        pred_h = pxywh[batch_indices, anchor_indices, celly_indices, cellx_indices, 3]
        pred_obj = objectness[batch_indices, anchor_indices, celly_indices, cellx_indices]
        pred_pro = probabilities[batch_indices, anchor_indices, celly_indices, cellx_indices]
        
        # print(f'pred_x={pred_x}')
        # print(f'pred_y={pred_y}')
        # print(f'pred_w={pred_w}')
        # print(f'pred_h={pred_h}')
        # print(f'pred_obj={pred_obj}')
        # print(f'pred_pro={pred_pro}')

        # 计算损失
        s = 2 - truth_box[:,2] * truth_box[:,3] if truth_box.numel() > 0 \
            else self.FloatTensor([0])
        s = torch.sqrt(s)

        loss_x = self.__safe_mse_loss(s * pred_x, s * truth_x)
        loss_y = self.__safe_mse_loss(s * pred_y, s * truth_y)
        loss_w = self.__safe_mse_loss(s * pred_w, s * truth_w)
        loss_h = self.__safe_mse_loss(s * pred_h, s * truth_h)
        loss_obj = self.__safe_bce_loss(pred_obj, truth_obj)
        loss_bkg = self.__safe_bce_loss(pred_bkg, truth_bkg)
        loss_pro = self.__safe_bce_loss(pred_pro, truth_pro)
        losses = loss_x + loss_y + loss_w + loss_h + loss_obj + loss_bkg + loss_pro
       
        # 性能指标
        if batch_indices.numel() > 0:
            class_correctness = (pred_pro.argmax(-1) == class_indices).float()     
            obj50_pp = (objectness > 0.5).float()
            obj50_tp = (pred_obj > 0.5).float()

            # print(f'obj50_pp={obj50_pp.sum()}')
            # print(f'objectness.numel={objectness.numel()}')
            
            # 真框和真框位置预测框的重叠率
            pred_boxes = bbox[batch_indices, anchor_indices, celly_indices, cellx_indices, :4]
            truth_boxes = self.__xywh_to_xyxy(truth_box)
            pred_boxes = self.__xywh_to_xyxy(pred_boxes)
            ious = self.__cal_IoU(truth_boxes.t(), pred_boxes.t())
            ious50 = (ious > 0.50).float()
            ious75 = (ious > 0.75).float()
            
            # print(f'pred_boxes={pred_boxes}')
            # print(f'ious.size={ious.size()}')
            # print(f'ious={ious}')

            class_accuracy = class_correctness.mean().detach().cpu().item()
            object_confidence = pred_obj.mean().detach().cpu().item()
            background_confidence = pred_bkg.mean().detach().cpu().item()
            precision = obj50_tp.sum() / (obj50_pp.sum() + 1e-16)
            recall50 = ious50.mean().detach().cpu().item()
            recall75 = ious75.mean().detach().cpu().item()
            aviou = ious.mean().detach().cpu().item()
            avcat = pred_pro[:,class_indices].mean().detach().cpu().item()
        else:
            class_accuracy = 0.0
            object_confidence = 0.0
            background_confidence = pred_bkg.mean().detach().cpu().item()
            precision = 0.0
            recall50 = 0.0
            recall75 = 0.0
            aviou = 0.0
            avcat = 0.0
        
        metrics = {
            "Grid" : w,
            "Lx" : loss_x.detach().cpu().item(),
            "Ly" : loss_y.detach().cpu().item(),
            "Lw" : loss_w.detach().cpu().item(),
            "Lh" : loss_h.detach().cpu().item(),
            "Lobj" : loss_obj.detach().cpu().item(),
            "Lbkg" : loss_bkg.detach().cpu().item(),
            "Lc" : loss_pro.detach().cpu().item(),
            "Loss" : losses.detach().cpu().item(),
            "Class" : class_accuracy,
            "Obj" : object_confidence,
            "Bkg" : background_confidence,
            "P" : precision,
            "R50" : recall50,
            "R75" : recall75,
            "Iou" : aviou,
            "Cat" : avcat}

        return losses, metrics
    
    def forward(self, x, target=None, in_size=None):
        '''前向传播
        
        参数
        ----
        x : Tensor
            特征输入张量.张量大小为N x C x H x W.其中C = num_anchors x (5 + num_classes).
        target : Tensor
            目标输入张量.
        in_size : tuple
            输入图像的分辨率.
        '''
        
        # 更新神经网络输入分辨率
        if in_size is not None: self.in_size = in_size 
        
        n = x.size(0)   # tensor memory layout: NCHW
        h = x.size(2)
        w = x.size(3)

        # 特征输入张量整形.张量z的最终大小为: Nx3x13x13x85或Nx3x26x26x85.
        z = x.view(n, self.num_anchors, 5 + self.num_classes, h, w) # 4D->5D
        z = z.permute(0, 1, 3, 4, 2).contiguous()   # 维度交换
        
        # 计算x,y,w,h的预测值.四个张量大小都为: Nx3x13x13或Nx3x26x26.
        px = torch.sigmoid(z[..., 0])
        py = torch.sigmoid(z[..., 1])
        pw = z[..., 2]
        ph = z[..., 3]
        
        # 计算检测框的位置和大小.返回张量的大小为: Nx3x13x13x4或Nx3x26x26x4.
        bbox = self.__cal_bounding_box(px, py, pw, ph)
        
        # 计算含有物体的置信度和物体的类别概率.
        # 含有物体的置信度张量大小为: Nx3x13x13或Nx3x26x26
        objectness = torch.sigmoid(z[..., 4]).type(self.FloatTensor)
        # 物体类别概率张量大小为: Nx3x13x13x80或Nx3x26x26x80
        probabilities = torch.sigmoid(z[..., 5:]).type(self.FloatTensor)

        # 推断模式返回预测值
        if target == None: return torch.cat(tensors=(
            bbox.view(n, -1, 4),
            objectness.view(n, -1, 1),
            probabilities.view(n, -1, self.num_classes)), dim=-1)
        
        # 计算损失
        pxywh = torch.stack((px, py, pw, ph), dim=-1)
        loss, metrics = self.__cal_loss(
            prediction=[pxywh,objectness,probabilities], target=target, bbox=bbox)
        
        return loss, metrics
