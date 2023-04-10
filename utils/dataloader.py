import imp
import os
import pickle

from torch.utils import data


def add_detr_res(dataset, conf):
    if conf.dataset in {'cub'}:
        datadir = '/AMMI_DATA_01/dataset/cub-200-2011/CUB_200_2011'
    elif conf.dataset in {'car'}:
        datadir = '/AMMI_DATA_01/dataset/stanford_cars'
    elif conf.dataset in {'aircraft'}:
        datadir = '/AMMI_DATA_01/dataset/fgvc-aircraft-2013b'

    path = os.path.join(datadir, '{}_detr_detection_result_800.pkl'.format(conf.dataset))

    """
    detr_res: [(x_c, y_c, w, h, box_p), (), ...]
            x_c = []
    """

    detr_res = pickle.load(open(path, 'rb'))
    dataset.imgs = dataset.imgs.merge(detr_res, how='left', on='path')
    dataset.imgs = dataset.imgs.reset_index(drop=True)
    del detr_res

    return dataset


def add_saliency_res(dataset, conf):
    if conf.dataset in {'cub'}:
        datadir = '/AMMI_DATA_01/dataset/cub-200-2011/CUB_200_2011'
    elif conf.dataset in {'car'}:
        datadir = '/AMMI_DATA_01/dataset/stanford_cars'
    elif conf.dataset in {'aircraft'}:
        datadir = '/AMMI_DATA_01/dataset/fgvc-aircraft-2013b'

    path = os.path.join(datadir, '{}_saliency_result_224.pkl'.format(conf.dataset))

    """
    saliency_res: [(x_c, y_c), (), ...]
            x_c = []
    """

    saliency_res = pickle.load(open(path, 'rb'))
    dataset.imgs = dataset.imgs.merge(saliency_res, how='left', on='path')
    dataset.imgs = dataset.imgs.reset_index(drop=True)
    del saliency_res

    return dataset


def get_dataloader(conf):
    if conf.dataset in {'cub'}:
        file_name = 'cub.py'
    elif conf.dataset in {'car'}:
        file_name = 'car.py'
    elif conf.dataset in {'aircraft'}:
        file_name = 'aircraft.py'
    else:
        file_name = None
    src_file = os.path.join('datasets', file_name)
    dataimp = imp.load_source('loader', src_file)
    ds_train, ds_test = dataimp.get_dataset(conf)

    if conf.is_train and conf.mixmethod in {'detrmix'}:
        ds_train = add_detr_res(ds_train, conf)
    if conf.is_train and conf.mixmethod in {'saliencymix'}:
        ds_train = add_saliency_res(ds_train, conf)

    train_loader = None
    if ds_train is not None:
        train_loader = data.DataLoader(ds_train, batch_size=conf.batch_size, shuffle=True, num_workers=conf.workers,
                                       pin_memory=True)
    val_loader = data.DataLoader(ds_test, batch_size=conf.batch_size, shuffle=False, num_workers=conf.workers,
                                 pin_memory=True)
    return train_loader, val_loader
