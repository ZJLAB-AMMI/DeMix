import torch

from utils import *
from utils.mixmethod import get_cam_weight


def train(train_loader, model, criterion, optimizer, conf, wmodel=None):
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
            if conf.mixmethod in {'detrmix', 'saliencymix'}:
                target_a, target_b, lam_a, lam_b = target
                if conf.use_cam_weight:
                    lam_a, lam_b = get_cam_weight(input, target_a, target_b, conf, wmodel)
                target_a = target_a.cuda()
                target_b = target_b.cuda()
                lam_a = lam_a.cuda()
                lam_b = lam_b.cuda()
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
