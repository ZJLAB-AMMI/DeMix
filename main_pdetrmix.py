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
    if conf.is_train:
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

            train_res = train(epoch / conf.epochs, train_loader, model, criterion, optimizer, conf, wmodel)
            scheduler.step()
            test_res = {"loss": None, "accuracy": None}
            if epoch % conf.eval_freq == conf.eval_freq - 1 or epoch == conf.epochs - 1:
                with torch.no_grad():
                    test_res = validate(val_loader, model, criterion)
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
    else:
        with torch.no_grad():
            test_res = validate(val_loader, model, criterion)
            print(test_res)

    return 0


if __name__ == '__main__':
    # get configs and set envs
    # dataset = 'cub'
    # netname = 'resnet50'
    # mixmethod = 'pdetrmix'
    gpu_ids = '4,5'
    debug = False
    # conf.is_train = False
    # conf.save_model = False
    # conf.batch_size = 32
    # conf.resume = True
    # conf.eval_freq = 1

    """
    screen -r first_run
        data_augmentation_set = [False, True]
        dataset_set = ['cifar100', 'cifar10']
        mixmethod_set = ['baseline', 'mixup', 'cutmix']
    screen -r imagenet 
        conf.netname = 'resnet50'
        conf.epochs = 300
        conf.lrgamma = 0.1
        conf.lrstep = [75, 150, 225]
        conf.weight_decay = 1e-4
        data_augmentation_set = [False]
        dataset_set = ['tiny_imagenet']
        mixmethod_set = ['baseline', 'mixup', 'cutmix', 'saliencymix']
    screen -r resnet101
        dataset_set = ['cub']
        mixmethod_set = ['cutmix', 'mixup', 'baseline']
        netname = 'resnet101'
    screen -r snapmix
        dataset_set = ['cub']
        mixmethod_set = ['snapmix', 'pdetrmix', 'detrmix', 'sum1pdetrmix', 'saliencymix', 'cutmix', 'mixup', 'baseline']
        netname = 'resnet50'
    """
    # conf.netname = 'resnet50'
    # conf.epochs = 300
    # conf.lrgamma = 0.1
    # conf.lrstep = [75, 150, 225]
    # conf.weight_decay = 1e-4

    netname_set = ['resnet18']
    dataset_set = ['cub']
    mixmethod_set = ['pdetrmix']
    background_wt_set = [0.8, 0.9, 1.0]
    for netname in netname_set:
        for dataset in dataset_set:
            for mixmethod in mixmethod_set:
                for background_wt in background_wt_set:
                    conf = get_config(dataset=dataset, netname=netname, mixmethod=mixmethod, gpu_ids=gpu_ids, debug=debug, background_wt=background_wt)
                    conf.batch_size = 16
                    conf.workers = 16
                    conf.weight_decay = 1e-4
                    set_env(conf)
                    # generate outdir name
                    set_outdir(conf)
                    # Set the logger
                    method_str = conf.mixmethod
                    method_str += '_bs{}_wd{}_background_wt{}'.format(conf.batch_size, conf.weight_decay, conf.background_wt)
                    method_str += '_{}'.format(conf.seed)
                    set_logger(conf, file_name=method_str)
                    main(conf, method_str)
