import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import utils


def get_cam_weight(mix_input, target_a, target_b, conf, model):
    imgsize = (conf.cropsize, conf.cropsize)
    size = mix_input.size()
    with torch.no_grad():
        output, fms, _ = model(mix_input)

        if conf.netname in {'inception'}:
            clsw = model.module.fc
        else:
            clsw = model.module.classifier
        weight = clsw.weight.data
        bias = clsw.bias.data
        weight = weight.view(weight.size(0), weight.size(1), 1, 1)

        fms = F.relu(fms)
        out = F.conv2d(fms, weight, bias=bias)

        sample_idx = list(range(size[0]))
        outmaps_a = out[sample_idx, target_a.long(), :, :]
        outmaps_b = out[sample_idx, target_b.long(), :, :]
        if imgsize is not None:
            outmaps_a = outmaps_a.view(outmaps_a.size(0), 1, outmaps_a.size(1), outmaps_a.size(2))
            outmaps_a = F.interpolate(outmaps_a, imgsize, mode='bilinear', align_corners=False)
            outmaps_b = outmaps_b.view(outmaps_b.size(0), 1, outmaps_b.size(1), outmaps_b.size(2))
            outmaps_b = F.interpolate(outmaps_b, imgsize, mode='bilinear', align_corners=False)
        outmaps_a = outmaps_a.squeeze()
        outmaps_b = outmaps_b.squeeze()

        outmaps_a = outmaps_a - (outmaps_a.min(dim=1, keepdim=True).values).min(dim=2, keepdim=True).values
        outmaps_a = outmaps_a / outmaps_a.sum(dim=[1, 2], keepdim=True)

        outmaps_b = outmaps_b - (outmaps_b.min(dim=1, keepdim=True).values).min(dim=2, keepdim=True).values
        outmaps_b = outmaps_b / outmaps_b.sum(dim=[1, 2], keepdim=True)

        lam_a = outmaps_a.sum(2).sum(1)
        lam_b = outmaps_b.sum(2).sum(1)

        lam_a = lam_a / (lam_a + lam_b + 1e-8)
        lam_b = 1.0 - lam_a

    return lam_a.cuda(), lam_b.cuda()


def get_spm(input, target, conf, model):
    imgsize = (conf.cropsize, conf.cropsize)
    bs = input.size(0)
    with torch.no_grad():
        output, fms, _ = model(input)
        if 'inception' in conf.netname:
            clsw = model.module.fc
        else:
            clsw = model.module.classifier
        weight = clsw.weight.data
        bias = clsw.bias.data
        weight = weight.view(weight.size(0), weight.size(1), 1, 1)
        fms = F.relu(fms)
        poolfea = F.adaptive_avg_pool2d(fms, (1, 1)).squeeze()
        clslogit = F.softmax(clsw.forward(poolfea))
        logitlist = []
        for i in range(bs):
            logitlist.append(clslogit[i, target[i].long()])
        clslogit = torch.stack(logitlist)

        out = F.conv2d(fms, weight, bias=bias)

        outmaps = []
        for i in range(bs):
            evimap = out[i, target[i].long()]
            outmaps.append(evimap)

        outmaps = torch.stack(outmaps)
        if imgsize is not None:
            outmaps = outmaps.view(outmaps.size(0), 1, outmaps.size(1), outmaps.size(2))
            outmaps = F.interpolate(outmaps, imgsize, mode='bilinear', align_corners=False)

        outmaps = outmaps.squeeze()

        for i in range(bs):
            outmaps[i] -= outmaps[i].min()
            outmaps[i] /= outmaps[i].sum()

    return outmaps, clslogit


def snapmix(input, target, conf, model=None):
    r = np.random.rand(1)
    lam_a = torch.ones(input.size(0))
    lam_b = 1 - lam_a
    target_b = target.clone()

    if r < conf.prob:
        wfmaps, _ = get_spm(input, target, conf, model)
        bs = input.size(0)
        lam = np.random.beta(conf.beta, conf.beta)
        lam1 = np.random.beta(conf.beta, conf.beta)
        rand_index = torch.randperm(bs).cuda()
        wfmaps_b = wfmaps[rand_index, :, :]
        target_b = target[rand_index]

        same_label = target == target_b
        bbx1, bby1, bbx2, bby2 = utils.rand_bbox(input.size(), lam)
        bbx1_1, bby1_1, bbx2_1, bby2_1 = utils.rand_bbox(input.size(), lam1)

        area = (bby2 - bby1) * (bbx2 - bbx1)
        area1 = (bby2_1 - bby1_1) * (bbx2_1 - bbx1_1)

        if area1 > 0 and area > 0:
            ncont = input[rand_index, :, bbx1_1:bbx2_1, bby1_1:bby2_1].clone()
            ncont = F.interpolate(ncont, size=(bbx2 - bbx1, bby2 - bby1), mode='bilinear', align_corners=True)
            input[:, :, bbx1:bbx2, bby1:bby2] = ncont
            lam_a = 1 - wfmaps[:, bbx1:bbx2, bby1:bby2].sum(2).sum(1) / (wfmaps.sum(2).sum(1) + 1e-8)
            lam_b = wfmaps_b[:, bbx1_1:bbx2_1, bby1_1:bby2_1].sum(2).sum(1) / (wfmaps_b.sum(2).sum(1) + 1e-8)
            tmp = lam_a.clone()
            lam_a[same_label] += lam_b[same_label]
            lam_b[same_label] += tmp[same_label]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            lam_a[torch.isnan(lam_a)] = lam
            lam_b[torch.isnan(lam_b)] = 1 - lam

    return input, target, target_b, lam_a.cuda(), lam_b.cuda()


def as_cutmix(input, target, conf, model=None):
    r = np.random.rand(1)
    lam_a = torch.ones(input.size(0))
    lam_b = 1 - lam_a
    target_b = target.clone()

    if r < conf.prob:
        bs = input.size(0)
        lam = np.random.beta(conf.beta, conf.beta)
        rand_index = torch.randperm(bs).cuda()
        target_b = target[rand_index]

        bbx1, bby1, bbx2, bby2 = utils.rand_bbox(input.size(), lam)
        bbx1_1, bby1_1, bbx2_1, bby2_1 = utils.rand_bbox(input.size(), lam)

        if (bby2_1 - bby1_1) * (bbx2_1 - bbx1_1) > 4 and (bby2 - bby1) * (bbx2 - bbx1) > 4:
            ncont = input[rand_index, :, bbx1_1:bbx2_1, bby1_1:bby2_1].clone()
            ncont = F.interpolate(ncont, size=(bbx2 - bbx1, bby2 - bby1), mode='bilinear', align_corners=True)
            input[:, :, bbx1:bbx2, bby1:bby2] = ncont
            # adjust lambda to exactly match pixel ratio
            lam_a = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            lam_a *= torch.ones(input.size(0))
    lam_b = 1 - lam_a

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


def saliencymix(input, target, conf, model=None):
    target, cx, cy = target
    target = target.cuda()
    size = input.size()
    r = np.random.rand(1)
    lam_a = torch.ones(size[0]).cuda()
    target_b = target.clone()

    if r < conf.prob:
        bs = size[0]
        rand_index = torch.randperm(bs).cuda()
        target_b = target[rand_index]
        input_b = input[rand_index].clone()
        bbx1, bby1, bbx2, bby2 = saliency_bbox(size, cx[rand_index[0]], cy[rand_index[0]])
        input[:, :, bbx1:bbx2, bby1:bby2] = input_b[:, :, bbx1:bbx2, bby1:bby2]
        lam_a = (1 - ((bbx2 - bbx1) * (bby2 - bby1) / (size[-1] * size[-2]))) * torch.ones(size[0])
    lam_b = 1 - lam_a

    return input, target, target_b, lam_a.cuda(), lam_b.cuda()


def detrmix_v2(input, target, conf, model=None):
    target, cx, cy, w, h, box_p = target
    target = target.cuda()
    size = input.size()
    r = np.random.rand(1)
    lam_b = torch.zeros(size[0]).cuda()
    target_b = target.clone()

    if r < conf.prob:
        bs = size[0]
        rand_index = torch.randperm(bs).cuda()
        target_b = target[rand_index]
        input_b = input[rand_index].clone()

        bbx1, bby1, bbx2, bby2 = detr_bbox_v2(size, cx, cy, w, h, box_p)
        # input [batch_size, channels, height, width]
        width_indexes = torch.arange(size[3]).unsqueeze(dim=0).tile(dims=[size[0], 1])
        height_indexes = torch.arange(size[2]).unsqueeze(dim=0).tile(dims=[size[0], 1])
        width_cond = (width_indexes >= bbx1.unsqueeze(dim=-1)) & (width_indexes < bbx2.unsqueeze(dim=-1))
        height_cond = (height_indexes >= bby1.unsqueeze(dim=-1)) & (height_indexes < bby2.unsqueeze(dim=-1))
        cond = (width_cond.unsqueeze(dim=1) & height_cond.unsqueeze(dim=-1))  # [bs, h, w]
        input_cond = cond.unsqueeze(dim=1).tile(dims=[1, size[1], 1, 1])

        input[input_cond[rand_index]] = input_b[input_cond[rand_index]]

        lam_b = ((bbx2 - bbx1) * (bby2 - bby1) / size[-1] / size[-2])[rand_index] * torch.ones(size[0])

    lam_a = 1 - lam_b

    return input, target, target_b, lam_a.cuda(), lam_b.cuda()


def pdetrmix(input, target, conf, model=None):
    """

    pdetrmix: detr box + detr label weight

    """
    [target, cx, cy, w, h, box_p] = target
    target = target.cuda()
    size = input.size()
    r = np.random.rand(1)
    lam_a = torch.ones(size[0])
    lam_b = 1 - lam_a
    target_b = target.clone()

    if r < conf.prob:
        bs = size[0]
        rand_index = torch.randperm(bs).cuda()
        target_b = target[rand_index]
        input_b = input[rand_index].clone()

        same_label = target == target_b

        bbx1, bby1, bbx2, bby2, cx, cy, w, h, flag = pdetr_bbox(size, cx, cy, box_p, 1.0)

        width_indexes = torch.arange(size[2]).unsqueeze(dim=0).tile(dims=[size[0], 1])
        height_indexes = torch.arange(size[3]).unsqueeze(dim=0).tile(dims=[size[0], 1])
        width_cond = (width_indexes >= bbx1.unsqueeze(dim=-1)) & (width_indexes < bbx2.unsqueeze(dim=-1))
        height_cond = (height_indexes >= bby1.unsqueeze(dim=-1)) & (height_indexes < bby2.unsqueeze(dim=-1))
        cond = (width_cond.unsqueeze(dim=-1) & height_cond.unsqueeze(dim=1))  # [bs, w, h]
        input_cond = cond.unsqueeze(dim=1).tile(dims=[1, size[1], 1, 1])

        input[input_cond[rand_index]] = input_b[input_cond[rand_index]]

        overlap_x = w - (cx - cx[rand_index]).abs()
        overlap_y = h - (cy - cy[rand_index]).abs()
        lam_cond = (overlap_x > 0) & (overlap_y > 0)

        box_p_a = torch.where(
            lam_cond,
            overlap_x * overlap_y / w / h * box_p,
            0
        )
        lam_a = box_p - box_p_a
        lam_b = box_p[rand_index]

        tmp = lam_a.clone()
        lam_a[same_label] += lam_b[same_label]
        lam_b[same_label] += tmp[same_label]

        lam = (((bbx2 - bbx1) * (bby2 - bby1) / size[-1] / size[-2])[rand_index]).double()

        lam_a[flag] = 1 - lam[flag]
        lam_b[flag] = lam[flag]

    return input, target, target_b, lam_a.cuda(), lam_b.cuda()


def normpdetrmix(input, target, conf, model=None):
    target, cx, cy, w, h, box_p = target
    target = target.cuda()
    size = input.size()
    r = np.random.rand(1)
    lam_a = torch.ones(size[0])
    lam_b = 1 - lam_a
    target_b = target.clone()

    if r < conf.prob:
        bs = size[0]
        rand_index = torch.randperm(bs).cuda()
        target_b = target[rand_index]
        input_b = input[rand_index].clone()

        same_label = target == target_b

        bbx1, bby1, bbx2, bby2, cx, cy, w, h, flag = pdetr_bbox(size, cx, cy, box_p, 1.0)

        width_indexes = torch.arange(size[2]).unsqueeze(dim=0).tile(dims=[size[0], 1])
        height_indexes = torch.arange(size[3]).unsqueeze(dim=0).tile(dims=[size[0], 1])
        width_cond = (width_indexes >= bbx1.unsqueeze(dim=-1)) & (width_indexes < bbx2.unsqueeze(dim=-1))
        height_cond = (height_indexes >= bby1.unsqueeze(dim=-1)) & (height_indexes < bby2.unsqueeze(dim=-1))
        cond = (width_cond.unsqueeze(dim=-1) & height_cond.unsqueeze(dim=1))  # [bs, w, h]
        input_cond = cond.unsqueeze(dim=1).tile(dims=[1, size[1], 1, 1])

        input[input_cond[rand_index]] = input_b[input_cond[rand_index]]

        overlap_x = w - (cx - cx[rand_index]).abs()
        overlap_y = h - (cy - cy[rand_index]).abs()
        lam_cond = (overlap_x > 0) & (overlap_y > 0)

        box_p_a = torch.where(
            lam_cond,
            overlap_x * overlap_y / w / h * box_p,
            0
        )
        lam_a = (box_p - box_p_a).abs()
        lam_b = (box_p[rand_index]).abs()

        lam_a = lam_a / (lam_a + lam_b + 1e-8)
        lam_b = 1 - lam_a

        tmp = lam_a.clone()
        lam_a[same_label] += lam_b[same_label]
        lam_b[same_label] += tmp[same_label]

        lam = (((bbx2 - bbx1) * (bby2 - bby1) / size[-1] / size[-2])[rand_index]).double()

        lam_a[flag] = 1 - lam[flag]
        lam_b[flag] = lam[flag]

    return input, target, target_b, lam_a.cuda(), lam_b.cuda()


def modelpdetrmix(input, target, conf, model=None):
    target, cx, cy, w, h, box_p = target
    target = target.cuda()
    size = input.size()
    r = np.random.rand(1)
    lam_a = torch.ones(size[0])
    lam_b = 1 - lam_a
    target_b = target.clone()

    if r < conf.prob:
        bs = size[0]
        rand_index = torch.randperm(bs).cuda()
        target_b = target[rand_index].clone()
        input_b = input[rand_index].clone()

        same_label = target == target_b

        bbx1, bby1, bbx2, bby2 = detr_bbox(size, cx, cy, box_p)

        width_indexes = torch.arange(size[2]).unsqueeze(dim=0).tile(dims=[size[0], 1])
        height_indexes = torch.arange(size[3]).unsqueeze(dim=0).tile(dims=[size[0], 1])
        width_cond = (width_indexes >= bbx1.unsqueeze(dim=-1)) & (width_indexes < bbx2.unsqueeze(dim=-1))
        height_cond = (height_indexes >= bby1.unsqueeze(dim=-1)) & (height_indexes < bby2.unsqueeze(dim=-1))
        cond = (width_cond.unsqueeze(dim=-1) & height_cond.unsqueeze(dim=1))  # [bs, w, h]
        input_cond = cond.unsqueeze(dim=1).tile(dims=[1, size[1], 1, 1])

        input[input_cond[rand_index]] = input_b[input_cond[rand_index]]

        with torch.no_grad():
            output, _ = model(input)
            prob = output.detach().softmax(dim=-1).cpu()
            sampleidx = list(np.arange(size[0]))
            lam_a = prob[sampleidx, target.long()]
            lam_b = prob[sampleidx, target_b.long()]

        tmp = lam_a.clone()
        lam_a[same_label] += lam_b[same_label]
        lam_b[same_label] += tmp[same_label]

    return input, target, target_b, lam_a.cuda(), lam_b.cuda()


def areamodelpdetrmix(epoch_rate, input, target, conf, model=None):
    target, cx, cy, w, h, box_p = target
    target = target.cuda()
    size = input.size()
    r = np.random.rand(1)
    lam_a = torch.ones(size[0])
    lam_b = 1 - lam_a
    target_b = target.clone()

    if r < conf.prob:
        bs = size[0]
        rand_index = torch.randperm(bs).cuda()
        target_b = target[rand_index]
        input_b = input[rand_index].clone()

        same_label = target == target_b

        bbx1, bby1, bbx2, bby2 = detr_bbox(size, cx, cy, box_p)

        width_indexes = torch.arange(size[2]).unsqueeze(dim=0).tile(dims=[size[0], 1])
        height_indexes = torch.arange(size[3]).unsqueeze(dim=0).tile(dims=[size[0], 1])
        width_cond = (width_indexes >= bbx1.unsqueeze(dim=-1)) & (width_indexes < bbx2.unsqueeze(dim=-1))
        height_cond = (height_indexes >= bby1.unsqueeze(dim=-1)) & (height_indexes < bby2.unsqueeze(dim=-1))
        cond = (width_cond.unsqueeze(dim=-1) & height_cond.unsqueeze(dim=1))  # [bs, w, h]
        input_cond = cond.unsqueeze(dim=1).tile(dims=[1, size[1], 1, 1])

        input[input_cond[rand_index]] = input_b[input_cond[rand_index]]

        output, _ = model(input)
        prob = output.detach().softmax(dim=-1).cpu()

        sampleidx = list(np.arange(size[0]))
        p1 = prob[sampleidx, target.long()]
        p2 = prob[sampleidx, target_b.long()]

        area2 = ((bbx2 - bbx1) * (bby2 - bby1) / size[-1] / size[-2])[rand_index]
        area1 = 1 - area2

        lam_a = (1 - epoch_rate) * area1 + epoch_rate * p1
        lam_b = (1 - epoch_rate) * area2 + epoch_rate * p2

        tmp = lam_a.clone()
        lam_a[same_label] += lam_b[same_label]
        lam_b[same_label] += tmp[same_label]

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


def camdetrmix(input, target, conf, model=None):
    target, cx, cy, w, h, box_p = target
    target = target.cuda()

    size = input.size()

    lam_a = torch.ones(size[0])
    lam_b = 1 - lam_a
    target_b = target.clone()

    r = np.random.rand(1)
    if r < conf.prob:
        wfmaps, _ = get_spm(input, target, conf, model)

        rand_index = torch.randperm(size[0]).cuda()
        wfmaps_b = wfmaps[rand_index, :, :].clone()
        input_b = input[rand_index].clone()
        target_b = target[rand_index].clone()

        same_label = target == target_b

        bbx1, bby1, bbx2, bby2 = detr_bbox(size, cx, cy, box_p)

        width_indexes = torch.arange(size[2]).unsqueeze(dim=0).tile(dims=[size[0], 1])
        height_indexes = torch.arange(size[3]).unsqueeze(dim=0).tile(dims=[size[0], 1])
        width_cond = (width_indexes >= bbx1.unsqueeze(dim=-1)) & (width_indexes < bbx2.unsqueeze(dim=-1))
        height_cond = (height_indexes >= bby1.unsqueeze(dim=-1)) & (height_indexes < bby2.unsqueeze(dim=-1))
        cond = (width_cond.unsqueeze(dim=-1) & height_cond.unsqueeze(dim=1))  # [bs, w, h]
        input_cond = cond.unsqueeze(dim=1).tile(dims=[1, size[1], 1, 1])

        input[input_cond[rand_index]] = input_b[input_cond[rand_index]]

        lam_cond = cond[rand_index].cuda()
        lam_a = 1 - torch.where(lam_cond, wfmaps, 0.0).sum(2).sum(1) / (wfmaps.sum(2).sum(1) + 1e-8)
        lam_b = torch.where(lam_cond, wfmaps_b, 0.0).sum(2).sum(1) / (wfmaps_b.sum(2).sum(1) + 1e-8)

        tmp = lam_a.clone()
        lam_a[same_label] += lam_b[same_label]
        lam_b[same_label] += tmp[same_label]

        lam = ((bbx2 - bbx1) * (bby2 - bby1) / size[-1] / size[-2])[rand_index].cuda()
        isnan = torch.isnan(lam_a)
        lam_a[isnan] = (1 - lam)[isnan]
        lam_b[isnan] = lam[isnan]

    return input, target, target_b, lam_a.cuda(), lam_b.cuda()


def pdetrmix_cam(input, target, conf, model=None):
    target, cx, cy, w, h, box_p = target
    target = target.cuda()
    size = input.size()
    r = np.random.rand(1)
    lam_a = torch.ones(size[0])
    lam_b = 1 - lam_a
    target_b = target.clone()

    if r < conf.prob:
        wfmaps, _ = get_spm(input, target, conf, model)
        bs = size[0]
        rand_index = torch.randperm(bs).cuda()
        wfmaps_b = wfmaps[rand_index, :, :]
        input_b = input[rand_index].clone()
        target_b = target[rand_index]

        same_label = target == target_b

        bbx1, bby1, bbx2, bby2, cx, cy, w, h, flag = pdetr_bbox(size, cx, cy, box_p, 1.0)

        width_indexes = torch.arange(size[2]).unsqueeze(dim=0).tile(dims=[size[0], 1])
        height_indexes = torch.arange(size[3]).unsqueeze(dim=0).tile(dims=[size[0], 1])
        width_cond = (width_indexes >= bbx1.unsqueeze(dim=-1)) & (width_indexes < bbx2.unsqueeze(dim=-1))
        height_cond = (height_indexes >= bby1.unsqueeze(dim=-1)) & (height_indexes < bby2.unsqueeze(dim=-1))
        cond = (width_cond.unsqueeze(dim=-1) & height_cond.unsqueeze(dim=1))  # [bs, w, h]
        input_cond = cond.unsqueeze(dim=1).tile(dims=[1, size[1], 1, 1])

        input[input_cond[rand_index]] = input_b[input_cond[rand_index]]

        lam_cond = cond[rand_index].cuda()
        lam_a = 1 - torch.where(lam_cond, wfmaps, 0.0).sum(2).sum(1) / (wfmaps.sum(2).sum(1) + 1e-8)
        lam_b = torch.where(lam_cond, wfmaps_b, 0.0).sum(2).sum(1) / (wfmaps_b.sum(2).sum(1) + 1e-8)

        tmp = lam_a.clone()
        lam_a[same_label] += lam_b[same_label]
        lam_b[same_label] += tmp[same_label]

        lam = (1 - ((bbx2 - bbx1) * (bby2 - bby1) / (size[-1] * size[-2]))).cuda()
        lam_a[torch.isnan(lam_a)] = lam[torch.isnan(lam_a)]
        lam_b[torch.isnan(lam_b)] = (1 - lam)[torch.isnan(lam_a)]

    return input, target, target_b, lam_a.cuda(), lam_b.cuda()


def detr_bbox(size, cx, cy, box_p):
    W = size[2]
    H = size[3]

    cut_rat = np.sqrt(1. - np.random.beta(1.0, 1.0))
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    x = torch.where(box_p >= 0, (cx * W).int(), np.random.randint(W))
    y = torch.where(box_p >= 0, (cy * H).int(), np.random.randint(H))

    bbx1 = np.clip(x - cut_w // 2, 0, W)
    bby1 = np.clip(y - cut_h // 2, 0, H)
    bbx2 = np.clip(x + cut_w // 2, 0, W)
    bby2 = np.clip(y + cut_h // 2, 0, H)

    return [bbx1, bby1, bbx2, bby2]


def detr_bbox_v2(size, cx, cy, w, h, box_p):
    """
        size: [batch_size, channels, height, width]
    """
    W = size[3]
    H = size[2]
    cond = box_p >= 0

    box_w = torch.where(cond, (W * w).int(), W)
    box_h = torch.where(cond, (H * h).int(), H)
    cx = torch.where(cond, (W * cx).int(), np.random.randint(W))
    cy = torch.where(cond, (H * cy).int(), np.random.randint(H))

    w_min = cx - box_w // 2
    h_min = cy - box_h // 2

    w_max = cx + box_w // 2
    h_max = cy + box_h // 2

    bbx1 = (w_min + box_w * torch.rand(1)).int()
    bby1 = (h_min + box_h * torch.rand(1)).int()
    bbx2 = (bbx1 + (w_max - bbx1) * torch.rand(1)).int()
    bby2 = (bby1 + (h_max - bby1) * torch.rand(1)).int()

    bbx1 = np.clip(bbx1, 0, W)
    bby1 = np.clip(bby1, 0, H)
    bbx2 = np.clip(bbx2, 0, W)
    bby2 = np.clip(bby2, 0, H)

    return [bbx1, bby1, bbx2, bby2]


def detr_bbox_v3(size, cx, cy, w, h, box_p):
    """
        size: [batch_size, channels, height, width]
    """
    W = size[3]
    H = size[2]
    cond = box_p >= 0

    box_w = torch.where(cond, (W * w).int(), W)
    box_h = torch.where(cond, (H * h).int(), H)
    cx = torch.where(cond, (W * cx).int(), np.random.randint(W))
    cy = torch.where(cond, (H * cy).int(), np.random.randint(H))

    w_min = cx - box_w // 2
    h_min = cy - box_h // 2

    w_max = cx + box_w // 2
    h_max = cy + box_h // 2

    bbx1 = (w_min + box_w * torch.rand(1)).int()
    bby1 = (h_min + box_h * torch.rand(1)).int()
    bbx2 = (bbx1 + (w_max - bbx1) * torch.rand(1)).int()
    bby2 = (bby1 + (h_max - bby1) * torch.rand(1)).int()

    bbx1 = np.clip(bbx1, 0, W)
    bby1 = np.clip(bby1, 0, H)
    bbx2 = np.clip(bbx2, 0, W)
    bby2 = np.clip(bby2, 0, H)

    return [bbx1, bby1, bbx2, bby2]


def pdetr_bbox(size, cx, cy, box_p, alpha):
    W = size[2]
    H = size[3]

    w_cut_rat = np.random.beta(alpha, alpha)
    h_cut_rat = np.random.beta(alpha, alpha)
    w_cut_rat = w_cut_rat if w_cut_rat >= 1.0 / W else 1.0 / W
    h_cut_rat = h_cut_rat if h_cut_rat >= 1.0 / H else 1.0 / H

    cond = box_p >= 0.7
    cx = torch.where(cond, W * cx, np.random.randint(W))
    cy = torch.where(cond, H * cy, np.random.randint(H))
    w = int(W * w_cut_rat)
    h = int(H * h_cut_rat)

    bbx1 = (cx - cx * w_cut_rat).floor().int()
    bbx2 = bbx1 + w
    bby1 = (cy - cy * h_cut_rat).floor().int()
    bby2 = bby1 + h
    return [bbx1, bby1, bbx2, bby2, cx, cy, w, h, box_p < 0.7]


def saliency_bbox(size, cx, cy):
    W = size[2]
    H = size[3]

    cut_rat = np.sqrt(1. - np.random.beta(1.0, 1.0))
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    x = (cx * W).int()
    y = (cy * H).int()

    bbx1 = np.clip(x - cut_w // 2, 0, W)
    bby1 = np.clip(y - cut_h // 2, 0, H)
    bbx2 = np.clip(x + cut_w // 2, 0, W)
    bby2 = np.clip(y + cut_h // 2, 0, H)

    return [bbx1, bby1, bbx2, bby2]


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


def get_part_object(img: np.array, detr_res: tuple):
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

    if bbx2 - bbx1 > 2:
        subbbx1 = np.random.randint(bbx1, bbx2 - 1)
        subbbx2 = np.random.randint(subbbx1 + 1, bbx2)
    else:
        subbbx1 = bbx1
        subbbx2 = bbx2

    if bby2 - bby1 > 2:
        subbby1 = np.random.randint(bby1, bby2 - 1)
        subbby2 = np.random.randint(subbby1 + 1, bby2)
    else:
        subbby1 = bby1
        subbby2 = bby2

    return img[subbby1: subbby2, subbbx1: subbbx2, :]


def get_detr_target_label_weight(target_img_size, target_detr_res: tuple, bbox: list):
    H = target_img_size[0]
    W = target_img_size[1]

    cx = int(target_detr_res[0][0] * W)
    cy = int(target_detr_res[1][0] * H)
    w = int(target_detr_res[2][0] * W)
    h = int(target_detr_res[3][0] * H)
    box_p = target_detr_res[4][0]

    tbbx1 = np.clip(cx - w // 2, 0, W)
    tbby1 = np.clip(cy - h // 2, 0, H)
    tbbx2 = np.clip(cx + w // 2, 0, W)
    tbby2 = np.clip(cy + h // 2, 0, H)

    bbx1, bby1, bbx2, bby2 = bbox

    if tbbx2 > bbx1 and bbx1 >= tbbx1:
        overlap_w = min(bbx2, tbbx2) - bbx1
    elif tbbx2 >= bbx2 and bbx2 > tbbx1:
        overlap_w = bbx2 - max(bbx1, tbbx1)
    elif bbx1 <= tbbx1 and tbbx2 <= bbx2:
        overlap_w = tbbx2 - tbbx1
    else:
        overlap_w = 0

    if tbby2 > bby1 and bby1 >= tbby1:
        overlap_h = min(bby2, tbby2) - bby1
    elif tbby2 >= bby2 and bby2 > tbby1:
        overlap_h = bby2 - max(bby1, tbby1)
    elif bby1 <= tbby1 and tbby2 <= bby2:
        overlap_h = tbby2 - tbby1
    else:
        overlap_h = 0

    source_area = (bbx2 - bbx1) * (bby2 - bby1)
    target_area = (tbbx2 - tbbx1) * (tbby2 - tbby1)
    overlap_area = overlap_w * overlap_h
    target_valid_area = target_area - overlap_area
    background_area = W * H - source_area - target_valid_area

    target_valid_area_wt = target_valid_area / (source_area + target_valid_area + 1e-8)
    target_background_area_wt = background_area / H / W
    target_valid_box_p = target_valid_area / target_area * box_p
    return target_valid_area_wt, target_background_area_wt, target_valid_box_p


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


def saliencymix_v2(img: np.array, patch: np.array, saliency_res: tuple):
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

# box_p = torch.tensor([2,3,4,-1])
# box_w = np.where(box_p >= 0, 6, 7)
# print(box_w)
#

# print()
