import os
import numpy as np
import torch
from PIL import Image

from datasets.tfs import get_imagenet_transform
import torchvision
from typing import Any, Callable, Optional, Tuple
from easydict import EasyDict


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class ImageFolder(torchvision.datasets.ImageFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            train: bool = True,
            conf: EasyDict = EasyDict()
    ):
        super(ImageFolder, self).__init__(root=root, transform=transform)
        self.train = train
        self.conf = conf

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, tmp = self.samples[index]
        cx, cy, w, h, box_p = -1.0, -1.0, -1.0, -1.0, -1.0
        if self.train and self.conf.mixmethod in {'detrmix', 'pdetrmix', 'normpdetrmix', 'modelpdetrmix',
                                                  'areamodelpdetrmix', 'camdetrmix'}:
            target = tmp[0]
            other_info = tmp[1]
            if len(other_info[0]) > 0:
                i = np.random.randint(len(other_info[0]))
                cx = other_info[0][i]
                cy = other_info[1][i]
                w = other_info[2][i]
                h = other_info[3][i]
                box_p = other_info[4][i]
        elif self.train and self.conf.mixmethod in {'saliencymix'}:
            target = tmp[0]
            other_info = tmp[1]
            if len(other_info) > 0:
                cx = other_info[0]
                cy = other_info[1]
        else:
            target = tmp
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.train and self.conf.mixmethod in {'detrmix', 'pdetrmix', 'normpdetrmix', 'modelpdetrmix',
                                                  'areamodelpdetrmix', 'camdetrmix'}:
            res = (target, cx, cy, w, h, box_p)
        elif self.train and self.conf.mixmethod in {'saliencymix'}:
            res = (target, cx, cy)
        else:
            res = target

        return sample, res


def get_dataset(conf):
    datadir = None
    if conf.dataset == 'imagenet':
        datadir = '/AMMI_DATA_01/Jin/ImageNet/data/ImageNet2012'
        conf['num_class'] = 1000
    elif conf.dataset == 'tiny_imagenet':
        datadir = '/AMMI_DATA_01/dataset/tiny-imagenet-200'
        conf['num_class'] = 200

    transform_train, transform_test = get_imagenet_transform(conf)

    ds = ImageFolder
    ds_train = ds(os.path.join(datadir, "train"), transform=transform_train, conf=conf, train=True)
    ds_test = ds(os.path.join(datadir, "val"), transform=transform_test, train=False)

    # ds_train.samples = ds_train.samples[0:150]
    # ds_train.targets = ds_train.targets[0:150]
    # ds_test.samples = ds_test.samples[0:150]
    # ds_test.targets = ds_test.targets[0:150]

    return ds_train, ds_test
