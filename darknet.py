# -*- coding: utf-8 -*-
# file: darknet.py
# brief: YOLOv3 implementation based on PyTorch
# author: Zeng Zhiwei
# date: 2019/9/24

import yolo
import json
import torch
import numpy as np
import torch.nn.functional as F

class ConvBnReLU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, momentum, negative_slope):
        super(ConvBnReLU, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = torch.nn.BatchNorm2d(num_features=out_channels, momentum=momentum)
        self.relu = torch.nn.LeakyReLU(negative_slope, inplace=True)
    
    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

class Residual(torch.nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, momentum, negative_slope):
        super(Residual, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, padding=0, bias=False)
        self.norm1 = torch.nn.BatchNorm2d(num_features=mid_channels, momentum=momentum)
        self.relu1 = torch.nn.LeakyReLU(negative_slope, inplace=True)
        self.conv2 = torch.nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = torch.nn.BatchNorm2d(num_features=out_channels, momentum=momentum)
        self.relu2 = torch.nn.LeakyReLU(negative_slope, inplace=True)
        
    def forward(self, x):
        y = self.relu1(self.norm1(self.conv1(x)))
        y = self.relu2(self.norm2(self.conv2(y)))
        return x + y

class Upsample(torch.nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
    
    def forward(self, x):
        return F.interpolate(input=x, scale_factor=self.scale_factor, mode=self.mode)

class Route(torch.nn.Module):
    def __init__(self):
        super(Route, self).__init__()
        
    def forward(self, tensors):
        return torch.cat(tensors=tensors, dim=1)
       
class DarkNet(torch.nn.Module):
    def __init__(self, anchors, in_size=(416,416), num_classes=80, tiny=False, disable_yolo=False):
        super(DarkNet, self).__init__()
        self.anchors=anchors
        self.num_classes = num_classes
        self.disable_yolo = disable_yolo
        self.momentum = 0.01
        self.negative_slope = 0.1
        self.detection_channels = (5 + self.num_classes) * 3
        self.prune_permit = {}
        
        '''backbone'''
        
        self.cbrl1 = ConvBnReLU(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, momentum=self.momentum, negative_slope=self.negative_slope)
        self.zpad1 = torch.nn.ZeroPad2d(padding=(1, 0, 1, 0))
        self.cbrl2 = ConvBnReLU(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0, momentum=self.momentum, negative_slope=self.negative_slope)
        
        self.stage1 = Residual(in_channels=64, mid_channels=32, out_channels=64, momentum=self.momentum, negative_slope=self.negative_slope)
        self.zpad2 = torch.nn.ZeroPad2d(padding=(1, 0, 1, 0))
        self.cbrl3 = ConvBnReLU(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0, momentum=self.momentum, negative_slope=self.negative_slope)
        
        self.stage2 = []
        for repeate in range(2):
            self.stage2.append(Residual(in_channels=128, mid_channels=64, out_channels=128, momentum=self.momentum, negative_slope=self.negative_slope))
        self.stage2 = torch.nn.Sequential(*self.stage2)
        self.zpad3 = torch.nn.ZeroPad2d(padding=(1, 0, 1, 0))
        self.cbrl4 = ConvBnReLU(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=0, momentum=self.momentum, negative_slope=self.negative_slope)
        
        self.stage3 = []
        for repeate in range(8):
            self.stage3.append(Residual(in_channels=256, mid_channels=128, out_channels=256, momentum=self.momentum, negative_slope=self.negative_slope))
        self.stage3 = torch.nn.Sequential(*self.stage3)
        self.zpad4 = torch.nn.ZeroPad2d(padding=(1, 0, 1, 0))
        self.cbrl5 = ConvBnReLU(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0, momentum=self.momentum, negative_slope=self.negative_slope)
        
        self.stage4 = []
        for repeate in range(8):
            self.stage4.append(Residual(in_channels=512, mid_channels=256, out_channels=512, momentum=self.momentum, negative_slope=self.negative_slope))
        self.stage4 = torch.nn.Sequential(*self.stage4)
        self.zpad5 = torch.nn.ZeroPad2d(padding=(1, 0, 1, 0))
        self.cbrl6 = ConvBnReLU(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=0, momentum=self.momentum, negative_slope=self.negative_slope)
        
        self.stage5 = []
        for repeate in range(4):
            self.stage5.append(Residual(in_channels=1024, mid_channels=512, out_channels=1024, momentum=self.momentum, negative_slope=self.negative_slope))
        self.stage5 = torch.nn.Sequential(*self.stage5)
        
        '''YOLO1'''
        
        self.pair1 = []
        for repeate in range(2):
            self.pair1.append(ConvBnReLU(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, momentum=self.momentum, negative_slope=self.negative_slope))
            self.pair1.append(ConvBnReLU(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, momentum=self.momentum, negative_slope=self.negative_slope))
        self.pair1 = torch.nn.Sequential(*self.pair1)
        
        self.cbrl7 = ConvBnReLU(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, momentum=self.momentum, negative_slope=self.negative_slope)
        self.cbrl8 = ConvBnReLU(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, momentum=self.momentum, negative_slope=self.negative_slope)
        self.conv1 = torch.nn.Conv2d(in_channels=1024, out_channels=self.detection_channels, kernel_size=1, padding=0, bias=True)
        self.yolo1 = yolo.Yolo(num_classes=self.num_classes, anchors=self.anchors, anchor_mask=(6,7,8), in_size=in_size)

        '''YOLO2'''

        self.route1 = Route()
        self.cbrl9 = ConvBnReLU(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, momentum=self.momentum, negative_slope=self.negative_slope)
        self.upsample1 = Upsample(scale_factor=2)
        self.route2 = Route()
        self.cbrl10 = ConvBnReLU(in_channels=768, out_channels=256, kernel_size=1, stride=1, padding=0, momentum=self.momentum, negative_slope=self.negative_slope)
        
        self.pair2 = []
        for repeate in range(2):
            self.pair2.append(ConvBnReLU(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, momentum=self.momentum, negative_slope=self.negative_slope))
            self.pair2.append(ConvBnReLU(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, momentum=self.momentum, negative_slope=self.negative_slope))
        self.pair2 = torch.nn.Sequential(*self.pair2)

        self.cbrl11 = ConvBnReLU(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, momentum=self.momentum, negative_slope=self.negative_slope)
        self.conv2 = torch.nn.Conv2d(in_channels=512, out_channels=self.detection_channels, kernel_size=1, padding=0, bias=True)
        self.yolo2 = yolo.Yolo(num_classes=self.num_classes, anchors=self.anchors, anchor_mask=(3,4,5), in_size=in_size)
               
        '''YOLO3'''

        self.route3 = Route()
        self.cbrl12 = ConvBnReLU(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, momentum=self.momentum, negative_slope=self.negative_slope)
        self.upsample2 = Upsample(scale_factor=2)
        self.route4 = Route()
        self.cbrl13 = ConvBnReLU(in_channels=384, out_channels=128, kernel_size=1, stride=1, padding=0, momentum=self.momentum, negative_slope=self.negative_slope)
        
        self.pair3 = []
        for repeate in range(2):
            self.pair3.append(ConvBnReLU(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, momentum=self.momentum, negative_slope=self.negative_slope))
            self.pair3.append(ConvBnReLU(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, momentum=self.momentum, negative_slope=self.negative_slope))
        self.pair3 = torch.nn.Sequential(*self.pair3)
        
        self.cbrl14 = ConvBnReLU(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, momentum=self.momentum, negative_slope=self.negative_slope)
        self.conv3 = torch.nn.Conv2d(in_channels=256, out_channels=self.detection_channels, kernel_size=1, padding=0, bias=True)
        self.yolo3 = yolo.Yolo(num_classes=self.num_classes, anchors=self.anchors, anchor_mask=(0,1,2), in_size=in_size)

        self.__init_weights()
        self.__init_prune_permit()
    
    def __init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.normal_(module.weight.data)
                module.weight.data *= (2.0/module.weight.numel())
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias.data, 0)
            elif isinstance(module, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(module.weight.data, 1)
                torch.nn.init.constant_(module.bias.data, 0)
                torch.nn.init.constant_(module.running_mean.data, 0)
                torch.nn.init.constant_(module.running_var.data, 0)  

    def forward(self, x, target=None):
        '''前向传播.
        
        参数
        ----
        x : Tensor
            输入图像张量.
        target : Tensor
            输入目标张量.
        '''
        
        outputs, metrics, losses = [], [], []
        tensors1, tensors2, tensors3, tensors4 = [], [], [], []
        if target is not None:
            in_size = (x.size(2), x.size(3))
        
        # backbone
        x = self.cbrl1(x)
        x = self.zpad1(x)
        x = self.cbrl2(x)
        x = self.stage1(x)
        x = self.zpad2(x)
        x = self.cbrl3(x)
        x = self.stage2(x)
        x = self.zpad3(x)
        x = self.cbrl4(x)
        x = self.stage3(x)
        tensors4.insert(0, x.clone())
        x = self.zpad4(x)
        x = self.cbrl5(x)
        x = self.stage4(x)
        tensors2.insert(0, x.clone())
        x = self.zpad5(x)
        x = self.cbrl6(x)
        x = self.stage5(x)
        
        # YOLO1
        x = self.pair1(x)
        x = self.cbrl7(x)
        tensors1.insert(0, x.clone())
        x = self.cbrl8(x)
        x = self.conv1(x)
        if not self.disable_yolo:
            if target == None:
                outputs.append(self.yolo1(x))
            else:
                loss, metric = self.yolo1(x, target, in_size)
                losses.append(loss)
                metrics.append(metric)
        
        # YOLO2
        x = self.route1(tensors1)
        x = self.cbrl9(x)
        x = self.upsample1(x)
        tensors2.insert(0, x.clone())
        x = self.route2(tensors2)
        x = self.cbrl10(x)
        x = self.pair2(x)
        tensors3.insert(0, x.clone())
        x = self.cbrl11(x)
        x = self.conv2(x)
        if not self.disable_yolo:
            if target == None:
                outputs.append(self.yolo2(x))
            else:
                loss, metric = self.yolo2(x, target, in_size)
                losses.append(loss)
                metrics.append(metric)

        # YOLO3
        x = self.route3(tensors3)
        x = self.cbrl12(x)
        x = self.upsample2(x)
        tensors4.insert(0, x.clone())
        x = self.route4(tensors4)
        x = self.cbrl13(x) 
        x = self.pair3(x)
        x = self.cbrl14(x)
        x = self.conv3(x)
        if not self.disable_yolo:
            if target == None:
                outputs.append(self.yolo3(x))
            else:
                loss, metric = self.yolo3(x, target, in_size)
                losses.append(loss)
                metrics.append(metric)

        if not self.disable_yolo:
            if target == None:
                return torch.cat(outputs, dim=1).detach().cpu()
            else:
                return sum(losses), metrics
        else:
            return torch.FloatTensor([0])
    
    def __init_prune_permit(self):
        index = -1
        for name, module in self.named_modules():
            if isinstance(module, (torch.nn.Conv2d, Route, Upsample, yolo.Yolo)):
                index = index + 1
                if 'stage' in name and 'conv2' in name:
                    index = index + 1
            if isinstance(module, torch.nn.BatchNorm2d):
                if 'norm2' in name:
                    self.prune_permit[name] = (index, False)
                else:
                    self.prune_permit[name] = (index, True)
        
        with open('model/prune_permit_init.json', 'w') as file:
            file.write(json.dumps(self.prune_permit))
            file.close()

    def load_prune_permit(self, path):
        with open(path, 'r') as file:
            self.prune_permit = json.load(file)
            file.close()
            print(self.prune_permit)

    def correct_bn_grad(self, lamb=0.01):
        '''修正BN层的梯度.
        
        参数
        ----
        lamb : float
            稀疏化影响因子.
        '''
        
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d) and self.prune_permit[name][1]:
                module.weight.grad.data.add_(lamb * torch.sign(module.weight.data))
       
if __name__ == '__main__':
    anchors = ((10,11),(14,15),(17,14),(18,17),(16,20),(21,22),(24,23),(29,32),(44,43))
    model = DarkNet(anchors)
    x = torch.rand(1, 3, 416, 416)
    y = model(x)
    print(f'y.size = {y.size()}')