import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from easydict import EasyDict as edict

parser = argparse.ArgumentParser(description='PyTorch Training')


def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# ------------------------------
def parser2dict():
    config, unparsed = parser.parse_known_args()
    cfg = edict(config.__dict__)
    return edict(cfg)


# ------------------------------
def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        # if k not in b:
        #    raise KeyError('{} is not a valid config key'.format(k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            # if k not in b:
            b[k] = v


def print_conf(opt):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        # default = self.parser.get_default(k)
        # if v != default:
        #     comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    return message


def cfg_from_file(cfg):
    """Load a config from file filename and merge it into the default options.
    """

    filename = cfg.config
    # args from yaml file
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.safe_load(f))

    _merge_a_into_b(yaml_cfg, cfg)

    return cfg


def get_config(dataset=None, netname=None, mixmethod=None, gpu_ids=None, debug=False, area_wt=0.0, background_wt=0.0, pretrained=1):
    # args from argparser
    cfg = parser2dict()
    cfg = cfg_from_file(cfg)
    if dataset is not None:
        cfg.dataset = dataset
    if netname is not None:
        cfg.netname = netname
    if mixmethod is not None:
        cfg.mixmethod = mixmethod
    if gpu_ids is not None:
        cfg.gpu_ids = gpu_ids
    if area_wt > 0:
        cfg.area_wt = area_wt
    if background_wt > 0:
        cfg.background_wt = background_wt
    cfg.pretrained = pretrained

    if 'mixmethod' in cfg:
        cfg['mixmethod'] = cfg['mixmethod'].split(',')
        if len(cfg['mixmethod']) == 1:
            cfg['mixmethod'] = cfg['mixmethod'][0]

    if cfg.dataset in {'cub', 'car', 'aircraft'}:
        cfg['prams_group'] = ['ftlayer', 'freshlayer']
        cfg['lr_group'] = [0.001, 0.01]
        cfg['lrstep'] = [80, 150, 180]
        cfg['lrgamma'] = 0.1
        cfg['lr'] = 0.01
        cfg['batch_size'] = 16
        cfg['workers'] = 16
        cfg['cropsize'] = 448

    if not cfg.pretrained:
        cfg['lr_group'] = [0.01, 0.01]
        cfg['epochs'] = 300

    if cfg['epochs'] == 300:
        cfg['lrstep'] = [150, 225, 270]

    if cfg['epochs'] == 100:
        cfg['lrstep'] = [40, 70]

    if cfg.dataset in ['nabirds', 'cub']:
        cfg['warp'] = False

    if debug:
        cfg['debug'] = True
        cfg['save_model'] = False
        cfg['epochs'] = 1
        cfg['batch_size'] = 16
        cfg['eval_freq'] = 1
    else:
        cfg['debug'] = False

    return cfg


def set_env(cfg):
    # set seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)  # cpu vars
    torch.manual_seed(cfg.seed)  # cpu  vars
    torch.cuda.manual_seed(cfg.seed)  # cpu  vars
    torch.cuda.manual_seed_all(cfg.seed)  # gpu vars
    if 'cudnn' in cfg:
        torch.backends.cudnn.benchmark = cfg.cudnn
    else:
        torch.backends.cudnn.benchmark = False

    cudnn.deterministic = True
    os.environ["NUMEXPR_MAX_THREADS"] = '16'
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids


# ----------------------------------------------------------------------------------------
# base
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=16, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--weightfile', default=None, type=str, metavar='PATH', help='path to model (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--seed', default=0, type=int, help='seeding for all random operation')
parser.add_argument('--config', default='config/comm.yml', type=str, help='config files')

# train
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='path', help='path to latest checkpoint (default: none)')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--eval_freq', default=5, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate')

parser.add_argument('--pretrained', default=1, type=float, help='loss weights')

# others

parser.add_argument('--mixmethod', default='snapmix', type=str, help='config files')
parser.add_argument('--netname', default='WideResNet28x10', type=str, help='config files')
parser.add_argument('--dropout_rate', type=float, default=0.3, help='')
parser.add_argument('--prob', type=float, default=1.0, help='')
parser.add_argument('--area_wt', type=float, default=0.0, help='')
parser.add_argument('--background_wt', type=float, default=0.0, help='')
parser.add_argument('--beta', type=float, default=1.0, help='')
parser.add_argument('--sample_rate', type=float, default=1.0, help='')
parser.add_argument('--cf_min', type=float, default=0.7, help='')
parser.add_argument('--dataset', default='cifar100', type=str, help='dataset')
parser.add_argument('--cropsize', default=448, type=int, metavar='N', help='cropsize')
parser.add_argument('--midlevel', dest='midlevel', action='store_true', help='midlevel')
parser.add_argument('--train_proc', default='comm', type=str, help='dataset')
parser.add_argument('--start_eval', default=-1, type=int, metavar='N', help='network depth')

parser.add_argument('--data_augmentation', dest='data_augmentation', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument("--output_root", type=str, default="/AMMI_DATA_01/WLP/test/model", required=False,
                    help=" ")

parser.add_argument("--result_root", type=str, default="/AMMI_DATA_01/WLP/test/result", required=False,
                    help=" ")

parser.add_argument('--single_box', dest='single_box', action='store_true', default=False,
                    help='evaluate model on validation set')

parser.add_argument('--save_model', dest='save_model', action='store_true', default=True,
                    help='evaluate model on validation set')

parser.add_argument('--is_train', dest='is_train', action='store_true', default=True,
                    help='evaluate model on validation set')

parser.add_argument('--use_cam_weight', dest='use_cam_weight', action='store_true', default=False,
                    help='evaluate model on validation set')
