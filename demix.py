import logging
import os
import time
import warnings

import tabulate
import torch
import torch.nn as nn

import networks
from utils import get_config, set_env, set_logger, set_outdir
from utils import get_dataloader
from utils import get_train_setting, load_checkpoint, get_proc, save_checkpoint, get_criterion


def main(conf, method_str):
    warnings.filterwarnings("ignore")
    best_acc = 0.
    epoch_start = 0

    output_path = os.path.join(conf.output_root, conf.dataset.upper(), conf.netname + '_{}'.format(method_str))
    os.makedirs(output_path, exist_ok=True)

    checkpoint_dict = None
    if conf.resume or (not conf.is_train):
        checkpoint_dict = load_checkpoint(output_path, -1, is_best=True)

    # dataloader
    train_loader, val_loader = get_dataloader(conf)

    # model
    model = networks.get_model(conf)
    model = nn.DataParallel(model).cuda()

    if conf.weightfile is not None:
        wmodel = networks.get_model(conf)
        wmodel = nn.DataParallel(wmodel).cuda()
        checkpoint_dict = load_checkpoint(wmodel, conf.weightfile)
        if 'best_acc' in checkpoint_dict:
            print('best score: {}'.format(best_acc))
    else:
        wmodel = model

    if checkpoint_dict is not None:
        model.load_state_dict(checkpoint_dict['state_dict'])

    # training and evaluate process for each epoch
    train, validate = get_proc(conf)

    detach_epoch = conf.epochs + 1
    if 'detach_epoch' in conf:
        detach_epoch = conf.detach_epoch

    criterion = get_criterion(conf)
    optimizer, scheduler = get_train_setting(model, conf)
    if checkpoint_dict is not None:
        epoch_start = checkpoint_dict['epoch']
        print('Resuming training process from epoch {}...'.format(epoch_start))
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
        scheduler.load_state_dict(checkpoint_dict['scheduler'])
    for epoch in range(epoch_start, conf.epochs):
        time_ep = time.time()
        lr = optimizer.param_groups[0]['lr']

        if epoch == detach_epoch:
            model.module.set_detach(False)

        train_res = train(train_loader, model, criterion, optimizer, conf, wmodel)
        scheduler.step()

        test_res = {"loss": None, "accuracy": None}
        if epoch % conf.eval_freq == conf.eval_freq - 1 or epoch == conf.epochs - 1:
            with torch.no_grad():
                test_res = validate(val_loader, model, criterion, conf)
                cur_acc = test_res["accuracy"]
                is_best = cur_acc > best_acc
                best_acc = max(cur_acc, best_acc)
                if conf.save_model and is_best:
                    save_checkpoint(
                        output_path,
                        epoch + 1,
                        is_best=True,
                        state_dict=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(),
                        best_acc=best_acc
                    )

        time_ep = time.time() - time_ep
        columns = ["mixmethod", "epoch", "learning_rate", "train_loss", "test_loss",
                   "test_acc",
                   "cost_time"]
        values = [conf.mixmethod, epoch + 1, lr, train_res["loss"],
                  test_res["loss"],
                  test_res["accuracy"],
                  time_ep]
        table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
        if epoch % 50 == 0:
            table = table.split("\n")
            table = "\n".join([table[1]] + table)
        else:
            table = table.split("\n")[2]
        if epoch % conf.eval_freq == conf.eval_freq - 1 or epoch == conf.epochs - 1:
            logging.info(table)
        else:
            print(table)

    return 0


if __name__ == '__main__':
    # config
    gpu_ids = '0'
    netname = 'resnet18'
    dataset = 'cub'
    mixmethod = 'detrmix'
    pretrained = 1

    conf = get_config(dataset=dataset, netname=netname, mixmethod=mixmethod, gpu_ids=gpu_ids, pretrained=pretrained)

    method_str = conf.mixmethod
    method_str += '_pretrained{}_seed{}'.format(conf.pretrained, conf.seed)

    # set env
    set_env(conf)

    # generate outdir name
    set_outdir(conf)

    # Set the logger
    set_logger(conf, file_name=method_str)

    # main
    main(conf, method_str)
