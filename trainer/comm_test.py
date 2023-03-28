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


# def eval(loader, model, criterion):
#     model.eval()
#
#     mean_loss = torch.zeros(1)
#     total_corrects = torch.zeros(1)
#     total_samples = torch.zeros(1)
#
#     acc = AverageAccMeter()
#
#     with torch.no_grad():
#         for i, (input, target) in enumerate(loader):
#             input = input.cuda(non_blocking=True)
#             target = target.cuda(non_blocking=True)
#             mean_loss = mean_loss.cuda(non_blocking=True)
#             total_corrects = total_corrects.cuda(non_blocking=True)
#             total_samples = total_samples.cuda(non_blocking=True)
#
#             output, _, moutput = model(input)
#             loss = torch.mean(criterion(output, target))
#             acc.add(output.data, target)
#
#             mean_loss = (i * mean_loss + loss.detach()) / (i + 1)
#
#             pred = output.data.argmax(1, keepdim=True)
#             correct = pred.eq(target.data.view_as(pred)).sum()
#             total_corrects += correct
#             total_samples += torch.tensor(input.size(0))
#
#     return {
#         "loss": mean_loss.item(),
#         "accuracy": (total_corrects / total_samples).item() * 100.0
#     }
