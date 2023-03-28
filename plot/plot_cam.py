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
mixmethod = 'mixup'
gpu_ids = '0'
conf = get_config(dataset=dataset, netname=netname, mixmethod=mixmethod, gpu_ids=gpu_ids)
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
model_file = os.path.join(conf.output_root, conf.dataset.upper(),
                          conf.netname + '_{}'.format(conf.mixmethod + '_bs16_wd0.0001_0'))
checkpoint_dict = load_checkpoint(model_file, -1, is_best=True)
model.load_state_dict(checkpoint_dict['state_dict'])

for p in model.parameters():
    p.requires_grad_(False)

# valid idx [120, 255, 270, 424, 437, 439, 449, 552, 601, 622]

for idx in [120, 255, 270, 424, 437, 439, 449, 552, 601, 622]:
    item = val_loader.dataset.imgs.iloc[idx]
    pil_img = pil_loader(os.path.join(datadir, 'images', item['path']))
    tensor_img = normalize(to_tensor(resize(pil_img, (224, 224))), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).cuda()
    tensor_img.requires_grad_(True)

    class_idx = int(item['label'])

    # CAM
    cam_extractor = CAM(model, target_layer='module.conv5', fc_layer='module.classifier')
    cam_extractor._hooks_enabled = True

    model.zero_grad()
    outputs = model(tensor_img.unsqueeze(0))

    activation_map = cam_extractor(class_idx, outputs)[0].squeeze(0).cpu()
    cam_extractor.remove_hooks()
    cam_extractor._hooks_enabled = False

    heatmap = to_pil_image(activation_map, mode="F")
    result = overlay_mask(pil_img, heatmap, alpha=0.5)

    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    axes.imshow(result)
    plt.axis('off')
    img_save_root = '/home/WLP/pythonProject/uncertainty/SnapMix/data'
    fig.savefig(
        os.path.join(img_save_root, '{}_{}_{}_{}_cam_plot.png'.format(idx, conf.dataset, conf.netname, conf.mixmethod)),
        bbox_inches='tight')
