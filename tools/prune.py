# -*- coding: utf-8 -*-
# file: prune.py
# brief: YOLOv3 implementation based on PyTorch
# author: Zeng Zhiwei
# date: 2019/8/15

from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from os.path import splitext
from io import StringIO
from functools import partial
import timeit
import json
import copy
import glob
import sys
import cv2
import os
import re
sys.path.append('.')
import utils
import darknet as net
import dataset as ds
from evaluate import evaluate
import yolov3

def save_print_stdout(object, filename):
    old_stdout = sys.stdout
    result = StringIO()
    sys.stdout = result
    print(object)
    with open(filename, 'w') as file:
        file.write(result.getvalue())
        file.close()
    sys.stdout = old_stdout

def save_model_parameter_as_file(model, path):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):                
            if module.bias is not None:
                np.savetxt(f'{path}/{name}.bias.txt', module.bias.data.numpy())
        elif isinstance(module, torch.nn.BatchNorm2d):
            np.savetxt(f'{path}/{name}.weight.txt', module.weight.data.numpy())
            np.savetxt(f'{path}/{name}.bias.txt', module.bias.data.numpy())
            np.savetxt(f'{path}/{name}.running_mean.txt', module.running_mean.data.numpy())
            np.savetxt(f'{path}/{name}.running_var.txt', module.running_var.data.numpy())

def calc_prune_thresh(model, pr, workspace, force_thresh=0):
    gamma = list()
    layer_id = list()
    layer_name = list()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d) and model.prune_permit[name][1]:
            gamma.append(module.weight.cpu().data.abs().numpy().tolist())
            layer_id.append(model.prune_permit[name][0])
            layer_name.append(name)
    
    gamma_all = list()
    max_gammas = list()
    for g in gamma:
        gamma_all += g
        max_gammas.append(np.max(g))
    
    max_thresh = np.min(max_gammas)
    gamma_all = np.sort(np.array(gamma_all))
    thresh_index = np.int32(args.pr * gamma_all.shape[0])
    thresh = gamma_all[thresh_index]
    if force_thresh > 0:
        thresh = force_thresh
        
    with open(f'{workspace}/log/gamma.txt', 'w') as file:
        for g in gamma_all:
            file.write(f'{g}\n')
        file.write(f'thresh_index:{thresh_index}\n')
        file.close()
    
    print(f'original prune threshold: {thresh}')
    print(f'max_gammas = {max_gammas}')
    print(f'prune threshold should be less than {max_thresh}')
    thresh = min(thresh, max_thresh)
    print(f'tuned prune threshold: {thresh}')
    
    num_subplots = len(gamma)
    for k in range(num_subplots):
        num_prune = np.sum(np.array(gamma[k]) < thresh)
        plt.title(f'{layer_name[k]},{num_prune}/{len(gamma[k])},{thresh}')
        plt.plot(gamma[k], 'r-+')
        plt.plot([thresh] * len(gamma[k]), 'b--')
        plt.xlabel('Channel Index')
        plt.ylabel('Gamma Abs. Value')
        plt.savefig(f'{workspace}/log/layer_{layer_id[k]}.jpg', dpi=120)
        plt.clf()
    
    return thresh

def make_prune_config(model, thresh):
    prune_config = {}
    parent_conv = None
    for name, module in model.named_modules():        
        if isinstance(module, torch.nn.Conv2d):
            if parent_conv is not None:
                prune_config[name] = {'in_mask':prune_config[parent_conv]['out_mask']}
                prune_config[name]['parent'] = parent_conv.replace('conv', 'norm')
                parent_conv = None
            elif 'cbrl9' in name:
                prune_config[name] = {'in_mask':prune_config['cbrl8.conv']['in_mask']}
                prune_config[name]['parent'] = prune_config['cbrl8.conv']['parent']
            elif 'cbrl12' in name:
                prune_config[name] = {'in_mask':prune_config['cbrl11.conv']['in_mask']}
                prune_config[name]['parent'] = prune_config['cbrl11.conv']['parent']
        elif isinstance(module, torch.nn.BatchNorm2d) and model.prune_permit[name][1]:
            mask = module.weight.cpu().data.abs().ge(thresh).numpy().tolist()
            parent_conv = name.replace('norm', 'conv')
            if parent_conv in prune_config:
                prune_config[parent_conv]['out_mask'] = mask
            else:
                prune_config[parent_conv] = {'out_mask':mask}
    return prune_config

def find_module(model, name):
    parent_module = model
    hierarchies = name.split(".")
    for h in hierarchies:
        parent_module = parent_module.__getattr__(h)
    return parent_module

def model_slimming(model, prune_config):
    new_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            norm_name = name.replace('conv', 'norm')
            if norm_name in model.prune_permit and model.prune_permit[norm_name][1]:
                out_mask = prune_config[name]['out_mask']
                out_indices = np.argwhere(out_mask)[:,0].tolist()
                out_channels = sum(out_mask)
                kernel_size = module.kernel_size
                stride = module.stride
                padding = module.padding
                bias = module.bias is not None      # biases all be False
                if 'in_mask' in prune_config[name]:
                    in_mask = prune_config[name]['in_mask']
                    in_indices = np.argwhere(in_mask)[:,0].tolist()
                    in_channels = sum(in_mask)
                    new_modules[name] = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
                    new_modules[name].weight.data = module.weight.data[out_indices,:,:,:][:,in_indices,:,:].clone()
                    print(f'copy input and output channels changed {name} done. {sum(in_mask)}/{len(in_mask)} {sum(out_mask)}/{len(out_mask)}')
                else:
                    in_channels = module.in_channels
                    new_modules[name] = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
                    new_modules[name].weight.data = module.weight.data[out_indices,:,:,:].clone()
                    print(f'copy output channels changed {name} done. {sum(out_mask)}/{len(out_mask)}')
            else:
                out_channels = module.out_channels
                kernel_size = module.kernel_size
                stride = module.stride
                padding = module.padding
                bias = module.bias is not None
                if name in prune_config and 'in_mask' in prune_config[name]:
                    in_mask = prune_config[name]['in_mask']
                    in_channels = sum(in_mask)
                    in_indices = np.argwhere(in_mask)[:,0].tolist()
                    new_modules[name] = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
                    new_modules[name].weight.data = module.weight.data[:,in_indices,:,:].clone()
                    print(f'copy input channels changed {name} done. {sum(in_mask)}/{len(in_mask)}')
                    if bias:
                        print(f'the shape of {name} is not changed, but we should update its parameters!')
                        new_modules[name].bias.data = module.bias.data.clone()
                        parent_module = find_module(model, prune_config[name]['parent'])
                        print(f"find parent {prune_config[name]['parent']} {parent_module}")
                        if sum(in_mask) == len(in_mask):
                            print(f'parent module has not been pruned!')
                            continue
                        prune_indices = np.argwhere(1 - np.array(in_mask))[:,0].tolist()
                        residual_bias = parent_module.bias[prune_indices]
                        residual_bias = F.leaky_relu(residual_bias, negative_slope=0.1)
                        filter_sumel = module.weight.data[:,prune_indices,:,:].sum(dim=(2,3))
                        offset = filter_sumel.matmul(residual_bias.reshape(-1,1)).reshape(-1)
                        new_modules[name].bias.data.add_(offset)
                else:
                    print(f'the {name} is not changed absolutely!')
        elif isinstance(module, torch.nn.BatchNorm2d):
            conv_name = name.replace('norm', 'conv')
            if model.prune_permit[name][1]:
                mask = prune_config[conv_name]['out_mask']
                num_features = sum(mask)
                momentum = module.momentum
                indices = np.argwhere(mask)[:,0].tolist()
                new_modules[name] = torch.nn.BatchNorm2d(num_features=num_features, momentum=momentum)
                new_modules[name].bias.data = module.bias.data[indices].clone()
                new_modules[name].weight.data = module.weight.data[indices].clone()
                new_modules[name].running_var.data = module.running_var.data[indices].clone()
                new_modules[name].running_mean.data = module.running_mean.data[indices].clone()
                new_modules[name].num_batches_tracked = module.num_batches_tracked
                print(f'copy {name} done. {sum(mask)}/{len(mask)}')
                if 'parent' in prune_config[conv_name]:
                    print('and we also need to update its parameters')
                    parent_module = find_module(model, prune_config[conv_name]['parent'])
                    print(f"find parent {prune_config[conv_name]['parent']} {parent_module}")
                    parent_conv = prune_config[conv_name]['parent'].replace('norm', 'conv')
                    mask = prune_config[parent_conv]['out_mask']
                    if sum(mask) == len(mask):
                        print(f'parent module has not been pruned!')
                        continue
                    prune_indices = np.argwhere(1 - np.array(mask))[:,0].tolist()
                    residual_bias = parent_module.bias[prune_indices]
                    residual_bias = F.leaky_relu(residual_bias, negative_slope=0.1)
                    conv_module = find_module(model, conv_name)
                    print(f'find current {conv_name} {conv_module}')
                    filter_sumel = conv_module.weight.data[:,prune_indices,:,:].sum(dim=(2,3))
                    offset = filter_sumel.matmul(residual_bias.reshape(-1,1)).reshape(-1)
                    new_modules[name].running_mean.data.sub_(offset[indices])
            else:
                if conv_name in prune_config:
                    print(f'the shape of {name} is not changed, but we should update its parameters!')
                    num_features = module.num_features
                    momentum = module.momentum
                    new_modules[name] = torch.nn.BatchNorm2d(num_features=num_features, momentum=momentum)
                    new_modules[name].bias.data = module.bias.data.clone()
                    new_modules[name].weight.data = module.weight.data.clone()
                    new_modules[name].running_var.data = module.running_var.data.clone()
                    new_modules[name].running_mean.data = module.running_mean.data.clone()
                    new_modules[name].num_batches_tracked = module.num_batches_tracked
                    parent_module = find_module(model, prune_config[conv_name]['parent'])
                    print(f"find parent {prune_config[conv_name]['parent']} {parent_module}")
                    parent_conv = prune_config[conv_name]['parent'].replace('norm', 'conv')
                    mask = prune_config[parent_conv]['out_mask']
                    if sum(mask) == len(mask):
                        print(f'parent module has not been pruned!')
                        continue
                    prune_indices = np.argwhere(1 - np.array(mask))[:,0].tolist()
                    residual_bias = parent_module.bias[prune_indices]
                    residual_bias = F.leaky_relu(residual_bias, negative_slope=0.1)
                    conv_module = find_module(model, conv_name)
                    print(f'find current {conv_name} {conv_module}')
                    filter_sumel = conv_module.weight.data[:,prune_indices,:,:].sum(dim=(2,3))
                    offset = filter_sumel.matmul(residual_bias.reshape(-1,1)).reshape(-1)
                    new_modules[name].running_mean.data.sub_(offset)
                else:
                    print(f'the {name} is not changed absolutely!')
    
    for name in new_modules:
        parent_module = model
        hierarchies = name.split(".")
        if len(hierarchies) == 1:
            model.__setattr__(name, new_modules[name])
            continue

        for h in hierarchies[:-1]:
            parent_module = parent_module.__getattr__(h)

        parent_module.__setattr__(hierarchies[-1], new_modules[name])
    
    return model

def inference(model, decoder, filename, in_size, class_names):
    model.eval()
    transform = ds.get_transform(train=False, net_w=in_size[0], net_h=in_size[1])
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    im = cv2.imread(filename, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    x, _ = transform(rgb, None)
    x = x.type(FloatTensor) / 255.0
    
    start = timeit.default_timer()
    xs = model(x)
    y = decoder(xs)
    end = timeit.default_timer()
    latency = end - start
    
    z = utils.get_network_boxes(y.clone(), im.shape[:2], thresh=0.5)
    nms = utils.do_nms_sort(z)
    result = utils.overlap_detection(im, nms, class_names)
    
    return result, latency, y

def compare_models(model1, model2):
    models_differ = 0
    for key_item1, key_item2 in zip(model1.state_dict().items(), model2.state_dict().items()):
        if torch.equal(key_item1[1], key_item2[1]):
            pass
        else:
            models_differ += 1
            if (key_item1[0] == key_item2[0]):
                print('mismtach found at', key_item1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('models match perfectly!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-size', type=str, default='416,416', help='network input size')
    parser.add_argument('--model', type=str, default='', help='model file')
    parser.add_argument('--dataset', type=str, default='', help='dataset path')
    parser.add_argument('--num-classes', type=int, default=3, help='number of classes')
    parser.add_argument('--prune-ratio', '-pr', dest='pr', type=float, default=0.15, help='prune ratio')
    parser.add_argument('--thresh', type=float, default=0, help='prune threshold')
    parser.add_argument('--image', type=str, default='', help='test image filename')
    parser.add_argument('--test-prune-model', '-test', dest='test', help='test pruned model', action='store_true')
    parser.add_argument('--eval', help='evaluate pruned model', action='store_true')
    parser.add_argument('--eval-epoch', dest='eval_epoch', type=int, default=0, help='epoch beginning evaluate')
    parser.add_argument('--workspace', type=str, default='workspace', help='workspace path')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_size = [int(insz) for insz in args.in_size.split(',')]
    anchors = np.loadtxt(os.path.join(args.dataset, 'anchors.txt'))
    class_names = utils.load_class_names(os.path.join(args.dataset, 'classes.txt'))
    decoder = yolov3.YOLOv3EvalDecoder(in_size, len(class_names), anchors)
    
    if args.test:
        model = torch.load(args.model, map_location=device)
        result, latency, _ = inference(model, decoder, args.image, in_size, class_names)
        print(f'pruned model latency is {latency} seconds.')
        cv2.imwrite('detection/detection-prune.jpg', result)
        sys.exit()
    
    if args.eval:
        dataset = ds.CustomDataset(args.dataset, 'test')
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=partial(ds.collate_fn, in_size=torch.IntTensor(in_size), train=False))
        
        if os.path.isfile(args.model):
            model = torch.load(args.model, map_location=device)
            model.eval()
            mAP = evaluate(model, decoder, data_loader, device, args.num_classes)
            print(f'mAP of {args.model} on validation dataset:%.2f%%' % (mAP * 100))
            sys.exit()
        elif os.path.isdir(args.model):
            paths = list(sorted(glob.glob(os.path.join(args.model, '*.pth'))))
            mAPs = list()
            for path in paths:
                if 'trainer' in path: continue
                segments = re.split(r'[-,.]', path)
                if int(segments[-2]) < args.eval_epoch: continue
                model = torch.load(path, map_location=device)
                model.eval()
                mAP = evaluate(model, decoder, data_loader, device, args.num_classes)
                mAPs.append(mAP)
                with open(f'{args.workspace}/log/evaluation.txt', 'a') as file:
                    file.write(f'{int(segments[-2])} {mAP}\n')
                    file.close()
                print(f'mAP of {path} on validation dataset:%.2f%%' % (mAP * 100))
            mAPs = np.array(mAPs)
            epoch = np.argmax(mAPs)
            print(f'Best model is ckpt-{epoch+args.eval_epoch}, best mAP is %.2f%%' % (mAPs[epoch] * 100))
    
    model = net.DarkNet(anchors, in_size=in_size, num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.load_prune_permit('model/prune_permit.json')
    save_print_stdout(model, f'{args.workspace}/log/model.txt')
    model.eval()
    # save_model_parameter_as_file(model, 'log/0')
    model_copy = copy.deepcopy(model)
    
    if args.image:
        result, latency, y = inference(model, decoder, args.image, in_size, class_names)
        print(f'original model latency is {latency} seconds.')
        cv2.imwrite('detection/detection.jpg', result)
        np.savetxt(f'{args.workspace}/log/0.txt', y.data.numpy().flatten())
    
    thresh = calc_prune_thresh(model, args.pr, args.workspace, force_thresh=args.thresh)    
    prune_config = make_prune_config(model, thresh)
    model = model_slimming(model, prune_config)
    # save_model_parameter_as_file(model, 'log/1')
    torch.save(model.state_dict(), f"{args.workspace}/log/yolov3-prune.pth")
    torch.save(model, f"{args.workspace}/log/yolov3-prune-full.pth")
    
    # compare_models(model_copy, model)
    
    save_print_stdout(model, f'{args.workspace}/log/model-prune.txt')
    with open(f'{args.workspace}/log/prune_config.json', 'w') as file:
        file.write(json.dumps(prune_config))
        file.close()

    if args.image:
        result, latency, y = inference(model, decoder, args.image, in_size, class_names)
        print(f'pruned model latency is {latency} seconds.')
        cv2.imwrite('detection/detection-prune.jpg', result)
        np.savetxt(f'{args.workspace}/log/1.txt', y.data.numpy().flatten())
    else:
        print('test pruned model...', end='')
        x = torch.rand(1, 3, 416, 416)
        ys = model(x)
        for y in ys:
            print(f'done\noutput size is {y.size()}')