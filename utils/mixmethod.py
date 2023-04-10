import numpy as np
import torch
from PIL import Image
import utils


def mixup(input, target, conf, model=None):
    r = np.random.rand(1)
    lam_a = torch.ones(input.size(0)).cuda()
    bs = input.size(0)
    target_a = target
    target_b = target

    if r < conf.prob:
        rand_index = torch.randperm(bs).cuda()
        target_b = target[rand_index]
        lam = np.random.beta(conf.beta, conf.beta)
        lam_a = lam_a * lam
        input = input * lam + input[rand_index] * (1 - lam)

    lam_b = 1 - lam_a

    return input, target, target_b, lam_a.cuda(), lam_b.cuda()


def cutout(input, target, conf=None, model=None):
    r = np.random.rand(1)
    lam = torch.ones(input.size(0)).cuda()
    target_b = target.clone()
    lam_a = lam
    lam_b = 1 - lam

    if r < conf.prob:
        bs = input.size(0)
        lam = 0.75
        bbx1, bby1, bbx2, bby2 = utils.rand_bbox(input.size(), lam)
        input[:, :, bbx1:bbx2, bby1:bby2] = 0

    return input, target, target_b, lam_a.cuda(), lam_b.cuda()


def cutmix(input, target, conf, model=None):
    r = np.random.rand(1)
    lam_a = torch.ones(input.size(0)).cuda()
    target_b = target.clone()

    if r < conf.prob:
        bs = input.size(0)
        lam = np.random.beta(conf.beta, conf.beta)
        rand_index = torch.randperm(bs).cuda()
        target_b = target[rand_index]
        input_b = input[rand_index].clone()
        bbx1, bby1, bbx2, bby2 = utils.rand_bbox(input.size(), lam)
        input[:, :, bbx1:bbx2, bby1:bby2] = input_b[:, :, bbx1:bbx2, bby1:bby2]

        # adjust lambda to exactly match pixel ratio
        lam_a = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
        lam_a *= torch.ones(input.size(0))

    lam_b = 1 - lam_a

    return input, target, target_b, lam_a.cuda(), lam_b.cuda()


def get_object(img: np.array, detr_res: tuple):
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


def get_saliency_patch(img: np.array, saliency_res: tuple):
    img_shape = img.shape
    H = img_shape[0]
    W = img_shape[1]

    cut_rat = np.sqrt(1. - np.random.beta(1.0, 1.0))
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = int(saliency_res[0] * W)
    cy = int(saliency_res[1] * H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return img[bby1: bby2, bbx1: bbx2, :]


def detrmix(img: np.array, patch: np.array):
    img_shape = img.shape  # [height, width, channel]
    H = img_shape[0]
    W = img_shape[1]

    cut_rat = np.sqrt(1. - np.random.beta(1.0, 1.0))
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    if bbx2 > bbx1 and bby2 > bby1:
        img[bby1: bby2, bbx1: bbx2, :] = np.array(Image.fromarray(patch).resize((bbx2 - bbx1, bby2 - bby1)))

    return img, [bbx1, bby1, bbx2, bby2]


def saliencymix(img: np.array, patch: np.array, saliency_res: tuple):
    img_shape = img.shape  # [height, width, channel]
    patch_shape = patch.shape
    H = img_shape[0]
    W = img_shape[1]

    cut_rat = np.sqrt(1. - np.random.beta(1.0, 1.0))
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = int(saliency_res[0] * W)
    cy = int(saliency_res[1] * H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    if bbx2 > bbx1 and bby2 > bby1 and patch_shape[0] > 0 and patch_shape[1] > 0:
        img[bby1: bby2, bbx1: bbx2, :] = np.array(Image.fromarray(patch).resize((bbx2 - bbx1, bby2 - bby1)))
    lam_b = (bbx2 - bbx1) * (bby2 - bby1) / W / H

    return img, lam_b
