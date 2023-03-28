import os
import sys

sys.path.append('/home/WLP/pythonProject/uncertainty/SnapMix')

import torch
import numpy as np
import pickle
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor, resize
from torchvision.datasets.folder import pil_loader

from datasets.cub import ImageLoader as CubImageLoader
from datasets.car import ImageLoader as CarImageLoader
from datasets.aircraft import ImageLoader as AircraftImageLoader
import pandas as pd


def get_patch(img: np.array, detr_res: tuple):
    img_shape = img.shape
    H = img_shape[0]
    W = img_shape[1]

    cx = int(detr_res[0][0] * W)
    cy = int(detr_res[1][0] * H)
    w = int(detr_res[2][0] * W)
    h = int(detr_res[3][0] * H)

    bbx1 = np.clip(cx - w // 2, 0, W)
    bby1 = np.clip(cy - h // 2, 0, H)
    bbx2 = np.clip(cx + w // 2, 0, W)
    bby2 = np.clip(cy + h // 2, 0, H)

    return img[bby1: bby2, bbx1: bbx2, :]


def compute_detr_res(dataset_name='cifar10', im_size=800):
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

    if dataset_name in {'cifar10', 'cifar100'}:
        datadir = '/AMMI_DATA_01/WLP/torch_datasets'
        ds = getattr(torchvision.datasets, dataset_name.upper())
        image_normalizer = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        dataset = ds(datadir, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

        # dataset.data = dataset.data[0:150]
        # dataset.targets = dataset.targets[0:150]

        images = torch.from_numpy(
            np.transpose(
                np.stack(dataset.data, axis=0),
                axes=[0, 3, 1, 2]
            )
        ) / 255.0
    elif dataset_name in {'cub'}:
        datadir = '/AMMI_DATA_01/dataset/cub-200-2011/CUB_200_2011'
        dataset = CubImageLoader(datadir, train=True, transform=transforms.Compose([transforms.ToTensor()]))
        image_normalizer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    elif dataset_name in {'car'}:
        datadir = '/AMMI_DATA_01/dataset/stanford_cars'
        dataset = CarImageLoader(datadir, train=True, transform=transforms.Compose([transforms.ToTensor()]))
        image_normalizer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    elif dataset_name in {'aircraft'}:
        datadir = '/AMMI_DATA_01/dataset/fgvc-aircraft-2013b'
        dataset = AircraftImageLoader(datadir, train=True, transform=transforms.Compose([transforms.ToTensor()]))
        image_normalizer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    else:
        if dataset_name == 'tiny_imagenet':
            datadir = '/AMMI_DATA_01/dataset/tiny-imagenet-200'
        else:
            datadir = '/AMMI_DATA_01/Jin/ImageNet/data/ImageNet2012'
        ds = getattr(torchvision.datasets, 'ImageFolder')
        image_normalizer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        dataset = ds(os.path.join(datadir, "train"), transform=transforms.Compose([transforms.ToTensor()]))

    detr = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
    detr.eval()
    detr.cuda()

    num_samples = len(dataset)
    print('total num samples: {}'.format(num_samples))

    # for idx in range(1):
    #     i = 100 * idx

    # i = 0
    # path = dataset.imgs['path'][i]
    # print(path)
    # if dataset_name == 'car':
    #     x = pil_loader(os.path.join(datadir, path))
    # elif dataset_name == 'cub':
    #     x = pil_loader(os.path.join(datadir, 'images', path))
    # else:
    #     x = pil_loader(os.path.join(datadir, path))

    # i = 6
    # path = dataset.imgs['path'][i]
    # print(path)
    # if dataset_name == 'car':
    #     x1 = pil_loader(os.path.join(datadir, path))
    # elif dataset_name == 'cub':
    #     x1 = pil_loader(os.path.join(datadir, 'images', path))
    # else:
    #     x1 = pil_loader(os.path.join(datadir, path))
    #
    # #     testing -----
    # x = np.array(x)
    # x1 = np.array(x1)
    # patch = x1[0:10, 0:10, :]
    # print(patch)
    # patch = Image.fromarray(patch).resize((20, 40))
    # print(np.array(patch).shape)

    # x[0:10, 0:10, :] =
    #
    # print(x.shape, x1[0:10, 0:10, :].shape)

    #     testing -----

    # x_shape = np.array(x).shape  # [height, width, channel]
    #
    # img_resize = transforms.Resize([im_size])
    # im = image_normalizer(img_resize(to_tensor(x)))
    # # print(im.shape)
    # y = detr_detect(detr, im)
    #
    #
    # path = get_patch(np.array(x), y)
    # path = Image.fromarray(path).resize((500, 50))
    # path = np.array(path)
    #
    #
    # xc = y[0][0]
    # yc = y[1][0]
    # w = y[2][0]
    # h = y[3][0]
    # xmin, ymin, w, h = (xc - 0.5 * w) * x_shape[1], (yc - 0.5 * h) * x_shape[0], w * x_shape[1], h * x_shape[0]
    # # print(xmin, ymin, w, h)
    #
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), sharex=False, sharey=False)
    # axes.imshow(path)
    # # axes.imshow(x)
    # # axes.add_patch(
    # #     plt.Rectangle((xmin, ymin), w, h, fill=False, color='r', linewidth=3)
    # # )
    # plt.axis('off')
    # plt.show()
    # fig.savefig(os.path.join('/home/WLP/pythonProject/uncertainty/SnapMix/data', '{}_test_{}.png'.format(dataset_name, 0)), bbox_inches='tight')

    res_list = []
    path_list = []
    for i in range(num_samples):
        print(i)
        if dataset_name in {'cifar10', 'cifar100'}:
            img_resize = transforms.Resize([im_size])
            img = image_normalizer(img_resize(images[i]))
        elif dataset_name in {'cub', 'car', 'aircraft'}:
            if dataset_name == 'cub':
                path = os.path.join(datadir, 'images', dataset.imgs.iloc[i]['path'])
            else:
                path = os.path.join(datadir, dataset.imgs.iloc[i]['path'])
            img_resize = transforms.Resize([im_size])
            img = image_normalizer(img_resize(to_tensor(pil_loader(path))))
            path_list.append(dataset.imgs.iloc[i]['path'])
        else:
            img_resize = transforms.Resize([im_size])
            img = image_normalizer(img_resize(to_tensor(pil_loader(dataset.samples[i][0]))))
        res = detr_detect(detr, img)
        res_list.append(res)

    if dataset_name in {'cub', 'car', 'aircraft'}:
        file = open(os.path.join(datadir, '{}_detr_detection_result_{}.pkl'.format(dataset_name, im_size)), 'wb')
        pickle.dump(pd.DataFrame({'path': path_list, 'detr_res': res_list}), file)
        file.close()
    else:
        file = open(os.path.join(datadir, '{}_detr_detection_result_{}.pkl'.format(dataset_name, im_size)), 'wb')
        pickle.dump(res_list, file)
        file.close()


def compute_salient_res(dataset_name='cifar10', im_size=800):
    import cv2

    def img_process(idx):
        if dataset_name in {'cifar10', 'cifar100'}:
            img = dataset.data[idx]
            img = img[:, :, ::-1]
        elif dataset_name in {'tiny_imagenet', 'imagenet'}:
            img = dataset.samples[idx][0]
            img = to_tensor(pil_loader(img))
            img = resize(img, [224, 224])
            img = img.cpu().numpy().transpose(1, 2, 0)
            img = (img[:, :, ::-1] * 255).astype("uint8")
        elif dataset_name in {'cub', 'car', 'aircraft'}:
            if dataset_name == 'cub':
                path = os.path.join(datadir, 'images', dataset.imgs.iloc[idx]['path'])
            else:
                path = os.path.join(datadir, dataset.imgs.iloc[idx]['path'])
            img_resize = transforms.Resize([im_size])
            img = img_resize(to_tensor(pil_loader(path)))
            img = img.cpu().numpy().transpose(1, 2, 0)
            img = (img[:, :, ::-1] * 255).astype("uint8")
        return img

    if dataset_name in {'cifar10', 'cifar100'}:
        datadir = '/AMMI_DATA_01/WLP/torch_datasets'
        ds = getattr(torchvision.datasets, dataset_name.upper())
        dataset = ds(datadir, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    elif dataset_name in {'cub'}:
        datadir = '/AMMI_DATA_01/dataset/cub-200-2011/CUB_200_2011'
        dataset = CubImageLoader(datadir, train=True, transform=transforms.Compose([transforms.ToTensor()]))
    elif dataset_name in {'car'}:
        datadir = '/AMMI_DATA_01/dataset/stanford_cars'
        dataset = CarImageLoader(datadir, train=True, transform=transforms.Compose([transforms.ToTensor()]))
    elif dataset_name in {'aircraft'}:
        datadir = '/AMMI_DATA_01/dataset/fgvc-aircraft-2013b'
        dataset = AircraftImageLoader(datadir, train=True, transform=transforms.Compose([transforms.ToTensor()]))
    else:
        if dataset_name == 'tiny_imagenet':
            datadir = '/AMMI_DATA_01/dataset/tiny-imagenet-200'
        else:
            datadir = '/AMMI_DATA_01/Jin/ImageNet/data/ImageNet2012'
        ds = getattr(torchvision.datasets, 'ImageFolder')
        dataset = ds(os.path.join(datadir, "train"), transform=transforms.Compose([transforms.ToTensor()]))

    saliency = cv2.saliency.StaticSaliencyFineGrained_create()

    # for idx in range(1):
    #     i = 100 * idx
    #
    # i = 700
    # path = dataset.imgs['path'][i]
    # print(path)
    # if dataset_name == 'car':
    #     x = pil_loader(os.path.join(datadir, path))
    # elif dataset_name == 'cub':
    #     x = pil_loader(os.path.join(datadir, 'images', path))
    # else:
    #     x = pil_loader(os.path.join(datadir, path))
    #
    # x_shape = np.array(x).shape  # [height, width, channel]
    # print(x_shape)
    # img_resize = transforms.Resize([im_size])
    # im = img_resize(to_tensor(x))
    # print(im.shape)
    # #
    # im = im.cpu().numpy().transpose(1, 2, 0)
    # print(im)
    # print(im.shape)
    # im = (im[:, :, ::-1] * 255).astype("uint8")
    # print(im)
    # print(im.shape)
    #
    # (success, saliencyMap) = saliency.computeSaliency(im)
    # print(saliencyMap)
    # print(saliencyMap.shape)
    # saliencyMap = (saliencyMap * 255).astype("uint8")
    # print(np.argmax(saliencyMap, axis=None))
    # maximum_indices = np.unravel_index(np.argmax(saliencyMap, axis=None), saliencyMap.shape)
    # yc = maximum_indices[0] / saliencyMap.shape[0]
    # xc = maximum_indices[1] / saliencyMap.shape[1]
    # w = 0.4
    # h = 0.4
    # xmin, ymin, w, h = (xc - 0.5 * w) * x_shape[1], (yc - 0.5 * h) * x_shape[0], w * x_shape[1], h * x_shape[0]
    # print(xmin, ymin, w, h)
    #
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), sharex=False, sharey=False)
    # axes.imshow(x)
    # axes.add_patch(
    #     plt.Rectangle((xmin, ymin), w, h, fill=False, color='r', linewidth=3)
    # )
    # plt.axis('off')
    # plt.show()
    # fig.savefig(os.path.join('/home/WLP/pythonProject/uncertainty/SnapMix/data', '{}_saliency_test_{}.png'.format(dataset_name, i)), bbox_inches='tight')

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

    if dataset_name in {'cub', 'car', 'aircraft'}:
        file = open(os.path.join(datadir, '{}_saliency_result_{}.pkl'.format(dataset_name, im_size)), 'wb')
        pickle.dump(pd.DataFrame({'path': path_list, 'saliency_res': res_list}), file)
        file.close()

# # dataset_name_set = {'cifar100', 'cifar10', 'tiny_imagenet'}
# dataset_name_set = ['cifar100', 'cifar10']

# os.environ['CUDA_VISIBLE_DEVICES'] = "6"
# dataset_name_set = ['cub', 'car', 'aircraft']
# for dataset_name in dataset_name_set:
#     # compute_detr_res(dataset_name=dataset_name)
#     compute_salient_res(dataset_name=dataset_name, im_size=224)


# compute_detr_res(dataset_name='cub')
# compute_salient_res(dataset_name='cub', im_size=240)
