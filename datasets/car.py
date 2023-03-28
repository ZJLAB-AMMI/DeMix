import torch
import numpy as np
from PIL import Image
import os
import pandas as pd
from torch.utils.data.sampler import WeightedRandomSampler
from scipy.io import loadmat
from datasets.tfs import get_car_transform
from easydict import EasyDict
from utils.mixmethod import get_object, get_saliency_patch, get_part_object, get_detr_target_label_weight, detrmix, \
    saliencymix_v2


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def get_mat_frame(path, img_folder):
    results = {}
    tmp_mat = loadmat(path)
    anno = tmp_mat['annotations'][0]
    results['path'] = [os.path.join(img_folder, anno[i][-1][0]) for i in range(anno.shape[0])]
    results['label'] = [anno[i][-2][0, 0] for i in range(anno.shape[0])]
    return results


class ImageLoader(torch.utils.data.Dataset):

    def __init__(self, root='Stanford_Cars', transform=None, target_transform=None, train=False, loader=pil_loader,
                 conf: EasyDict = EasyDict()):
        img_folder = root
        pd_train = pd.DataFrame.from_dict(
            get_mat_frame(os.path.join(root, 'devkit', 'cars_train_annos.mat'), 'cars_train'))
        pd_test = pd.DataFrame.from_dict(
            get_mat_frame(os.path.join(root, 'cars_test_annos_withlabels.mat'), 'cars_test'))
        data = pd.concat([pd_train, pd_test])
        data['train_flag'] = pd.Series(data.path.isin(pd_train['path']))
        data = data[data['train_flag'] == train]
        data['label'] = data['label'] - 1

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

    def __getitem__(self, index):
        item = self.imgs.iloc[index]
        file_path = item['path']
        target = item['label']

        img = self.loader(os.path.join(self.root, file_path))

        res = target

        if self.train and self.conf.mixmethod in {'detrmix', 'pdetrmix', 'sum1pdetrmix', 'campdetrmix'}:
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

                if self.conf.mixmethod in {'pdetrmix', 'sum1pdetrmix'}:
                    target_valid_area_wt, target_background_area_wt, target_valid_box_p = get_detr_target_label_weight(
                        img.shape, target_detr_res, bbox)
                    lam_a = target_valid_area_wt * target_valid_box_p + target_background_area_wt * self.conf.background_wt
                    lam_b = (1.0 - target_valid_area_wt) * source_detr_res[4][0]

                    if self.conf.mixmethod == 'sum1pdetrmix':
                        lam_a = lam_a / (lam_a + lam_b + 1e-8)
                        lam_b = 1.0 - lam_a

                target_b = source_item['label']

                if np.random.rand(1) < 0.0005:
                    img_save_root = '/home/WLP/pythonProject/uncertainty/SnapMix/data'
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), sharex=False, sharey=False)
                    axes.imshow(img)
                    text = f'wa: {lam_a:0.2f}, wb: {lam_b:0.2f}'
                    axes.text(0, 0, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
                    plt.axis('off')
                    plt.show()
                    fig.savefig(os.path.join(img_save_root,
                                             '{}_{}_img_{}.png'.format(self.conf.mixmethod, self.conf.dataset,
                                                                       index % 10)), bbox_inches='tight')
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

                img, lam_b = saliencymix_v2(np.array(img), patch, source_saliency_res)
                lam_a = 1.0 - lam_b

                target_b = source_item['label']

                if np.random.rand(1) < 0.0005:
                    img_save_root = '/home/WLP/pythonProject/uncertainty/SnapMix/data'
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), sharex=False, sharey=False)
                    axes.imshow(img)
                    text = f'wa: {lam_a:0.2f}, wb: {lam_b:0.2f}'
                    axes.text(0, 0, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
                    plt.axis('off')
                    plt.show()
                    fig.savefig(os.path.join(img_save_root, '{}_{}_img_{}.png'.format(self.conf.mixmethod, self.conf.dataset, index % 10)), bbox_inches='tight')
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


def get_dataset(conf):
    datadir = '/AMMI_DATA_01/dataset/stanford_cars'
    conf['num_class'] = 196

    transform_train, transform_test = get_car_transform(conf)

    ds_train = ImageLoader(datadir, train=True, transform=transform_train, conf=conf)
    ds_test = ImageLoader(datadir, train=False, transform=transform_test)

    return ds_train, ds_test
