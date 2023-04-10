from utils import *


def validate(val_loader, model, criterion, conf):
    losses = AverageMeter()
    acces = AverageAccMeter()

    model.eval()

    with torch.no_grad():
        for idx, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            if 'inception' in conf.netname:
                output = model(input)
            else:
                output, _, _ = model(input)

            loss = torch.mean(criterion(output, target))

            losses.add(loss.item(), input.size(0))
            acces.add(output.data, target)
            del loss, output

    return {
        "loss": losses.value(),
        "accuracy": acces.value()
    }

