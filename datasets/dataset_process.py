import os
import sys

sys.path.append('/home/WLP/pythonProject/uncertainty/SnapMix')

import torch
import numpy as np
import pickle
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor
from torchvision.datasets.folder import pil_loader

from datasets.cub import ImageLoader as CubImageLoader
from datasets.car import ImageLoader as CarImageLoader
from datasets.aircraft import ImageLoader as AircraftImageLoader
import pandas as pd


def compute_detr_res(dataset_name='cub', im_size=800):
    def detr_detect(detr_model, img, topk=10):
        img = img.unsqueeze(0)
        outputs = detr_model(img.cuda())
        probs = outputs['pred_logits'].softmax(-1)[0, :, :-1].cpu().detach()
        cfp = probs.max(-1).values
        topk_idx = torch.sort(cfp, dim=-1, descending=True).indices[0:topk]
        box_p = cfp[topk_idx]
        boxes = outputs['pred_boxes'][0, topk_idx].cpu().detach()
        x_c, y_c, w, h = boxes.unbind(1)
        res = (
            x_c.numpy().tolist(), y_c.numpy().tolist(), w.numpy().tolist(), h.numpy().tolist(), box_p.numpy().tolist())
        return res

    if dataset_name in {'cub'}:
        datadir = '/AMMI_DATA_01/dataset/cub-200-2011/CUB_200_2011'
        dataset = CubImageLoader(datadir, train=True, transform=transforms.Compose([transforms.ToTensor()]))
    elif dataset_name in {'car'}:
        datadir = '/AMMI_DATA_01/dataset/stanford_cars'
        dataset = CarImageLoader(datadir, train=True, transform=transforms.Compose([transforms.ToTensor()]))
    elif dataset_name in {'aircraft'}:
        datadir = '/AMMI_DATA_01/dataset/fgvc-aircraft-2013b'
        dataset = AircraftImageLoader(datadir, train=True, transform=transforms.Compose([transforms.ToTensor()]))

    image_normalizer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    detr = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
    detr.eval()
    detr.cuda()

    num_samples = len(dataset)
    print('total num samples: {}'.format(num_samples))

    res_list = []
    path_list = []
    for i in range(num_samples):
        print(i)
        if dataset_name == 'cub':
            path = os.path.join(datadir, 'images', dataset.imgs.iloc[i]['path'])
        else:
            path = os.path.join(datadir, dataset.imgs.iloc[i]['path'])
        img_resize = transforms.Resize([im_size])
        img = image_normalizer(img_resize(to_tensor(pil_loader(path))))
        path_list.append(dataset.imgs.iloc[i]['path'])
        res = detr_detect(detr, img)
        res_list.append(res)

    file = open(os.path.join(datadir, '{}_detr_detection_result_{}.pkl'.format(dataset_name, im_size)), 'wb')
    pickle.dump(pd.DataFrame({'path': path_list, 'detr_res': res_list}), file)
    file.close()


def compute_salient_res(dataset_name='cub', im_size=800):
    import cv2

    def img_process(idx):
        if dataset_name == 'cub':
            path = os.path.join(datadir, 'images', dataset.imgs.iloc[idx]['path'])
        else:
            path = os.path.join(datadir, dataset.imgs.iloc[idx]['path'])
        img_resize = transforms.Resize([im_size])
        img = img_resize(to_tensor(pil_loader(path)))
        img = img.cpu().numpy().transpose(1, 2, 0)
        img = (img[:, :, ::-1] * 255).astype("uint8")
        return img

    if dataset_name in {'cub'}:
        datadir = '/AMMI_DATA_01/dataset/cub-200-2011/CUB_200_2011'
        dataset = CubImageLoader(datadir, train=True, transform=transforms.Compose([transforms.ToTensor()]))
    elif dataset_name in {'car'}:
        datadir = '/AMMI_DATA_01/dataset/stanford_cars'
        dataset = CarImageLoader(datadir, train=True, transform=transforms.Compose([transforms.ToTensor()]))
    elif dataset_name in {'aircraft'}:
        datadir = '/AMMI_DATA_01/dataset/fgvc-aircraft-2013b'
        dataset = AircraftImageLoader(datadir, train=True, transform=transforms.Compose([transforms.ToTensor()]))

    saliency = cv2.saliency.StaticSaliencyFineGrained_create()

    res_list = []
    path_list = []
    for idx in range(len(dataset)):
        print(idx)
        img = img_process(idx)
        (success, saliencyMap) = saliency.computeSaliency(img)
        saliencyMap = (saliencyMap * 255).astype("uint8")
        maximum_indices = np.unravel_index(np.argmax(saliencyMap, axis=None), saliencyMap.shape)
        res_list.append((maximum_indices[1] / saliencyMap.shape[1], maximum_indices[0] / saliencyMap.shape[0]))
        path_list.append(dataset.imgs.iloc[idx]['path'])
    file = open(os.path.join(datadir, '{}_saliency_result_{}.pkl'.format(dataset_name, im_size)), 'wb')
    pickle.dump(pd.DataFrame({'path': path_list, 'saliency_res': res_list}), file)
    file.close()

# os.environ['CUDA_VISIBLE_DEVICES'] = "6"
# dataset_name_set = ['cub', 'car', 'aircraft']
# for dataset_name in dataset_name_set:
#     compute_detr_res(dataset_name=dataset_name)
#     compute_salient_res(dataset_name=dataset_name, im_size=224)
