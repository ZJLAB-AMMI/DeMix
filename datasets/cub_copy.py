from PIL import Image
import os
import pandas as pd
from datasets.tfs import get_cub_transform
from torch.utils.data import Dataset
from easydict import EasyDict
import numpy as np


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageLoader(Dataset):

    def __init__(self, root, transform=None, target_transform=None, train=False, loader=pil_loader, conf: EasyDict = EasyDict()):
        img_folder = os.path.join(root, "images")
        img_paths = pd.read_csv(os.path.join(root, "images.txt"), sep=" ", header=None, names=['idx', 'path'])
        img_labels = pd.read_csv(os.path.join(root, "image_class_labels.txt"), sep=" ", header=None,
                                 names=['idx', 'label'])
        train_test_split = pd.read_csv(os.path.join(root, "train_test_split.txt"), sep=" ", header=None,
                                       names=['idx', 'train_flag'])
        bounding_box = pd.read_csv(os.path.join(root, "bounding_boxes.txt"), sep=" ", header=None,
                                   names=['idx', 'x', 'y', 'w', 'h'])

        data = img_paths.merge(img_labels, how='left', on='idx')
        data = data.merge(train_test_split, how='left', on='idx')
        data = data.merge(bounding_box, how='left', on='idx')

        data['label'] = data['label'] - 1
        alldata = data.copy()

        data = data[data['train_flag'] == train]
        imgs = data.reset_index(drop=True)

        if len(imgs) == 0:
            raise (RuntimeError("no csv file"))
        self.root = img_folder
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train = train
        self.conf = conf
        print('num of data:{}'.format(len(imgs)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        item = self.imgs.iloc[index]
        file_path = item['path']
        target = item['label']
        cx, cy, w, h, box_p = -1.0, -1.0, -1.0, -1.0, -1.0
        if self.train and self.conf.mixmethod in {'detrmix', 'pdetrmix', 'normpdetrmix', 'modelpdetrmix',
                                                  'areamodelpdetrmix', 'camdetrmix', 'detrmix_v2'}:
            detr_res = item['detr_res']
            if len(detr_res[0]) > 0:
                score_list = [1.0 + np.random.rand(1) if p >= self.conf.cf_min else np.random.rand(1) for p in detr_res[4]]
                i = np.array(score_list).argmax()
                cx = detr_res[0][i]
                cy = detr_res[1][i]
                w = detr_res[2][i]
                h = detr_res[3][i]
                box_p = detr_res[4][i]
        img = self.loader(os.path.join(self.root, file_path))
        img = self.transform(img)

        if self.train and self.conf.mixmethod in {'detrmix', 'pdetrmix', 'normpdetrmix', 'modelpdetrmix',
                                                  'areamodelpdetrmix', 'camdetrmix', 'detrmix_v2'}:
            res = (target, cx, cy, w, h, box_p)
        else:
            res = target

        return img, res

    def __len__(self):
        return len(self.imgs)


def get_dataset(conf):
    datadir = '/AMMI_DATA_01/dataset/cub-200-2011/CUB_200_2011'
    conf['num_class'] = 200

    transform_train, transform_test = get_cub_transform(conf)

    ds_train = ImageLoader(datadir, train=True, transform=transform_train, conf=conf)
    ds_test = ImageLoader(datadir, train=False, transform=transform_test)

    return ds_train, ds_test
