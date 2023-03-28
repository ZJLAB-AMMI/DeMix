# Copyright (C) 2020-2023, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""
CAM visualization
"""
import sys
sys.path.append('/home/WLP/pythonProject/uncertainty/SnapMix')

import argparse
import math

import matplotlib.pyplot as plt
import torch

import torchvision
from torchvision import models
# from torchvision.transforms.functional import , resize, to_pil_image, to_tensor

from torchcam.utils import overlay_mask

import os
from PIL import Image
from networks.resnet_ft_cam import get_net
import torch.nn as nn
from torchcam.methods import CAM
from torchvision.transforms.functional import to_tensor, resize, normalize, to_pil_image
from utils import get_config, get_dataloader, load_checkpoint


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

# CONFIG
dataset = 'cub'
netname = 'resnet50'
gpu_ids = '0'
conf = get_config(dataset=dataset, netname=netname, gpu_ids=gpu_ids)
conf.batch_size = 16
conf.workers = 16
conf.weight_decay = 1e-4
if dataset in {'cub'}:
    datadir = '/AMMI_DATA_01/dataset/cub-200-2011/CUB_200_2011'

# DATA
train_loader, val_loader = get_dataloader(conf)

# MODEL
model = get_net(conf)
model = nn.DataParallel(model).cuda()

for i in range(len(val_loader.dataset)):
    item = val_loader.dataset.imgs.iloc[i]
    pil_img = pil_loader(os.path.join(datadir, 'images', item['path']))
    tensor_img = normalize(to_tensor(resize(pil_img, (224, 224))), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).cuda()
    class_idx = int(item['label'])

    mixupflag = False
    cutoutflag = False
    cutmixflag = False
    saliencymixflag = False
    detrmixflag = False

    for mixmethod in ['mixup', 'cutout', 'cutmix', 'saliencymix', 'detrmix']:
        model_file = os.path.join(conf.output_root, conf.dataset.upper(),
                                  conf.netname + '_{}'.format(mixmethod + '_bs16_wd0.0001_0'))
        checkpoint_dict = load_checkpoint(model_file, -1, is_best=True)
        model.load_state_dict(checkpoint_dict['state_dict'])
        model.eval()

        outputs = model(tensor_img.unsqueeze(0))
        pred = outputs.data.argmax().item()

        if mixmethod == 'mixup':
            mixupflag = pred == class_idx
        elif mixmethod == 'cutout':
            cutoutflag = pred == class_idx
        elif mixmethod == 'cutmix':
            cutmixflag = pred == class_idx
        elif mixmethod == 'saliencymix':
            saliencymixflag = pred == class_idx
        else:
            detrmixflag = pred == class_idx
    if detrmixflag and (not mixupflag) and (not cutoutflag) and (not cutmixflag) and (not saliencymixflag):
        print(i)




