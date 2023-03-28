import numpy as np
import numpy.random
from random import sample
import torch
from PIL import Image

from datasets.tfs import get_cifar_transform
import torchvision
from typing import Any, Callable, Optional, Tuple
from easydict import EasyDict


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class CIFAR10(torchvision.datasets.CIFAR10):

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            conf: EasyDict = EasyDict()
    ) -> None:
        super(CIFAR10, self).__init__(root, train=train, transform=transform, target_transform=target_transform,
                                      download=download)
        self.conf = conf

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.data[index]
        tmp = self.targets[index]
        cx, cy, w, h, box_p = -1.0, -1.0, -1.0, -1.0, -1.0

        if self.train and self.conf.mixmethod in {'detrmix', 'pdetrmix', 'normpdetrmix', 'modelpdetrmix',
                                                  'areamodelpdetrmix', 'camdetrmix'}:
            target = tmp[0]
            other_info = tmp[1]
            if len(other_info[0]) > 0:
                score_list = [1.0 + np.random.rand(1) if p >= self.conf.cf_min else np.random.rand(1) for p in other_info[4]]
                i = np.array(score_list).argmax()
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

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.train and self.conf.mixmethod in {'detrmix', 'pdetrmix', 'normpdetrmix', 'modelpdetrmix',
                                                  'areamodelpdetrmix', 'camdetrmix'}:
            res = (target, cx, cy, w, h, box_p)
        elif self.train and self.conf.mixmethod in {'saliencymix'}:
            res = (target, cx, cy)
        else:
            res = target

        return img, res


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }


def get_dataset(conf):
    datadir = '/AMMI_DATA_01/WLP/torch_datasets'

    if conf.dataset == 'cifar10':
        conf['num_class'] = 10
        ds = CIFAR10
    else:
        conf['num_class'] = 100
        ds = CIFAR100

    transform_train, transform_test = get_cifar_transform(conf)

    ds_train = None
    if conf.is_train:
        ds_train = ds(datadir, train=True, download=True, transform=transform_train, conf=conf)
    ds_test = ds(datadir, train=False, download=True, transform=transform_test)

    # ds_train.data = ds_train.data[0:150]
    # ds_train.targets = ds_train.targets[0:150]
    # ds_test.data = ds_test.data[0:150]
    # ds_test.targets = ds_test.targets[0:150]

    return ds_train, ds_test
