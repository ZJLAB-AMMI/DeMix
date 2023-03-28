import torch

from utils import *
from utils.mixmethod import get_cam_weight


def train(epoch_rate, train_loader, model, criterion, optimizer, conf, wmodel=None):
    losses = AverageMeter()

    model.train()

    mixmethod = None
    if 'mixmethod' in conf:
        if 'baseline' not in conf.mixmethod:
            mixmethod = conf.mixmethod
            if wmodel is None:
                wmodel = model

    for idx, (input, target) in enumerate(train_loader):
        input = input.cuda()
        if not isinstance(target, (tuple, list)):
            target = target.cuda()

        if 'baseline' not in conf.mixmethod:
            if conf.mixmethod == 'areamodelpdetrmix':
                input, target_a, target_b, lam_a, lam_b = eval(mixmethod)(epoch_rate, input, target, conf, wmodel)
            elif conf.mixmethod in {'detrmix', 'pdetrmix', 'sum1pdetrmix', 'saliencymix'}:
                target_a, target_b, lam_a, lam_b = target
                if conf.use_cam_weight:
                    lam_a, lam_b = get_cam_weight(input, target_a, target_b, conf, wmodel)

                target_a = target_a.cuda()
                target_b = target_b.cuda()
                lam_a = lam_a.cuda()
                lam_b = lam_b.cuda()

                # imgsize = (conf.cropsize, conf.cropsize)
                # bs = input.size(0)
                # with torch.no_grad():
                #     print(input.size())
                #     output, fms = wmodel(input)
                #     print(output.shape)
                #     print(fms.shape)
                #
                #     if conf.netname in {'inception'}:
                #         clsw = model.module.fc
                #     else:
                #         clsw = model.module.classifier
                #
                #     weight = clsw.weight.data
                #     bias = clsw.bias.data
                #     print(weight.shape)
                #     print(bias.shape)
                #
                #     weight = weight.view(weight.size(0), weight.size(1), 1, 1)
                #     print(weight.shape)
                #
                #     fms = F.relu(fms)
                #
                #     # poolfea = F.adaptive_avg_pool2d(fms, (1, 1)).squeeze()
                #     # clslogit = F.softmax(clsw.forward(poolfea))
                #     # logitlist = []
                #     # for i in range(bs):
                #     #     logitlist.append(clslogit[i, target[i]])
                #     # clslogit = torch.stack(logitlist)
                #     #
                #     out = F.conv2d(fms, weight, bias=bias)
                #     print(out.shape)
                #
                #
                #     outmaps = []
                #     for i in range(bs):
                #         evimap = out[i, target_a[i]]
                #         outmaps.append(evimap)
                #     outmaps = torch.stack(outmaps)
                #     print(outmaps)
                #     print(outmaps.shape)
                #
                #     x = out[list(range(bs)), target_a.long(), :, :]
                #     print(x)
                #     print(x.shape)
                #
                #     if imgsize is not None:
                #         outmaps = outmaps.view(outmaps.size(0), 1, outmaps.size(1), outmaps.size(2))
                #         print(outmaps.shape)
                #         outmaps = F.interpolate(outmaps, imgsize, mode='bilinear', align_corners=False)
                #         print(outmaps.shape)
                #
                #
                #     outmaps = outmaps.squeeze()
                #     print(outmaps.shape)
                #
                #     x_min = outmaps.min(dim=1, keepdim=True).values
                #     x_min = x_min.min(dim=2, keepdim=True).values
                #     print(x_min)
                #     print(x_min.shape)
                #
                #     x = outmaps - x_min
                #     x = x / x.sum(dim=[1, 2], keepdim=True)
                #     print(x)
                #     print(x.shape)
                #
                #     for i in range(bs):
                #         outmaps[i] -= outmaps[i].min()
                #         outmaps[i] /= outmaps[i].sum()
                #
                #     print(outmaps)
            else:
                input, target_a, target_b, lam_a, lam_b = eval(mixmethod)(input, target, conf, wmodel)

            output, _, moutput = model(input)

            loss_a = criterion(output, target_a)
            loss_b = criterion(output, target_b)
            loss = torch.mean(loss_a * lam_a + loss_b * lam_b)

            if 'inception' in conf.netname:
                loss1_a = criterion(moutput, target_a)
                loss1_b = criterion(moutput, target_b)
                loss1 = torch.mean(loss1_a * lam_a + loss1_b * lam_b)
                loss += 0.4 * loss1
        else:
            output, _, moutput = model(input)
            loss = torch.mean(criterion(output, target))

            if 'inception' in conf.netname:
                loss += 0.4 * torch.mean(criterion(moutput, target))

        losses.add(loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if conf.debug:
            break
    return {
        "loss": losses.value()
    }
