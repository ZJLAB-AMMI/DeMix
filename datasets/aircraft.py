import os

import numpy as np
import pandas as pd
import torch.utils.data as data
from PIL import Image
from easydict import EasyDict
from torchvision.datasets.folder import default_loader

from datasets.tfs import get_aircraft_transform
from utils.mixmethod import get_object, get_saliency_patch, detrmix, saliencymix


def make_dataset(dir, image_ids, targets):
    assert (len(image_ids) == len(targets))
    images = []
    dir = os.path.expanduser(dir)
    for i in range(len(image_ids)):
        item = (os.path.join(dir, 'data', 'images',
                             '%s.jpg' % image_ids[i]), targets[i])
        images.append(item)
    return images


def find_classes(classes_file):
    # read classes file, separating out image IDs and class names
    image_ids = []
    targets = []
    f = open(classes_file, 'r')
    for line in f:
        split_line = line.split(' ')
        image_ids.append(split_line[0])
        targets.append(' '.join(split_line[1:]))
    f.close()

    # index class names
    classes = np.unique(targets)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    targets = [class_to_idx[c] for c in targets]

    return (image_ids, targets, classes, class_to_idx)


class ImageLoader(data.Dataset):
    """`FGVC-Aircraft <http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft>`_ Dataset.
     Args:
        root (string): Root directory path to dataset.
        class_type (string, optional): The level of FGVC-Aircraft fine-grain classification
            to label data with (i.e., ``variant``, ``family``, or ``manufacturer``).
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g. ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in the root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    class_types = ('variant', 'family', 'manufacturer')
    splits = ('train', 'val', 'trainval', 'test')

    def __init__(self, root='data/aircraft', transform=None,
                 target_transform=None, train=True, loader=default_loader, conf: EasyDict = EasyDict()):
        self.root = os.path.expanduser(root)
        self.class_type = 'variant'
        self.split = 'trainval' if train else 'test'
        self.classes_file = os.path.join(self.root, 'data',
                                         'images_%s_%s.txt' % (self.class_type, self.split))

        (image_ids, targets, classes, class_to_idx) = find_classes(self.classes_file)
        samples = make_dataset(self.root, image_ids, targets)

        paths = []
        labels = []
        for sample in samples:
            path, label = sample
            paths.append(path)
            labels.append(label)

        datadict = {'path': paths, 'label': labels}
        data = pd.DataFrame(datadict)
        imgs = data.reset_index(drop=True)

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.train = train
        self.conf = conf

    def __getitem__(self, index):
        item = self.imgs.iloc[index]
        path = item['path']
        target = item['label']
        img = self.loader(path)

        res = target

        if self.train and self.conf.mixmethod in {'detrmix'}:
            source_idx = np.random.randint(len(self.imgs))
            source_item = self.imgs.iloc[source_idx]
            source_detr_res = source_item['detr_res']
            target_detr_res = item['detr_res']
            r = np.random.rand(1)
            if r < self.conf.prob and len(source_detr_res[0]) > 0 and len(target_detr_res[0]) > 0:
                source_file_path = source_item['path']
                source_img = self.loader(os.path.join(self.root, source_file_path))

                source_patch = get_object(np.array(source_img), source_detr_res)
                img, bbox = detrmix(np.array(img), source_patch)

                bbx1, bby1, bbx2, bby2 = bbox
                lam_b = (bbx2 - bbx1) * (bby2 - bby1) / img.shape[0] / img.shape[1]
                lam_a = 1.0 - lam_b
                target_b = source_item['label']
            else:
                lam_a = 1.0
                lam_b = 0.0
                target_b = target

            res = (target, target_b, lam_a, lam_b)

        if self.train and self.conf.mixmethod in {'saliencymix'}:
            source_idx = np.random.randint(len(self.imgs))
            source_item = self.imgs.iloc[source_idx]
            source_saliency_res = source_item['saliency_res']
            r = np.random.rand(1)
            if r < self.conf.prob and len(source_saliency_res) > 0:
                source_img = self.loader(os.path.join(self.root, source_item['path']))
                patch = get_saliency_patch(np.array(source_img), source_saliency_res)
                img, lam_b = saliencymix(np.array(img), patch, source_saliency_res)
                lam_a = 1.0 - lam_b
                target_b = source_item['label']
            else:
                lam_a = 1.0
                lam_b = 0.0
                target_b = target
            res = (target, target_b, lam_a, lam_b)

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        img = self.transform(img)
        return img, res

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, 'data', 'images')) and \
               os.path.exists(self.classes_file)


def get_dataset(conf):
    datadir = '/AMMI_DATA_01/dataset/fgvc-aircraft-2013b'
    conf['num_class'] = 100

    transform_train, transform_test = get_aircraft_transform(conf)

    ds_train = ImageLoader(datadir, train=True, transform=transform_train, conf=conf)
    ds_test = ImageLoader(datadir, train=False, transform=transform_test)

    return ds_train, ds_test
