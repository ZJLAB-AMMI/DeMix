import os
import os.path as path
from datetime import datetime

import torch


def save_checkpoint(dir, epoch, name="checkpoint", is_best=False, **kwargs):
    state = {"epoch": epoch}
    state.update(kwargs)
    if is_best:
        filepath = os.path.join(dir, "%s-best.pt" % name)
    else:
        filepath = os.path.join(dir, "%s-%d.pt" % (name, epoch))
    torch.save(state, filepath)


def load_checkpoint(dir, epoch, name="checkpoint", is_best=False, **kwargs):
    if is_best:
        filepath = os.path.join(dir, "%s-best.pt" % name)
    else:
        filepath = os.path.join(dir, "%s-%d.pt" % (name, epoch))
    return torch.load(filepath, **kwargs)


def set_outdir(conf):
    if 'timedir' in conf:
        timestr = datetime.now().strftime('%d-%m-%Y_%I_%M-%S_%p')
        outdir = os.path.join(conf.result_root, conf.mixmethod, conf.net_type + '_' + conf.dataset, timestr)
    else:
        outdir = os.path.join(conf.result_root, conf.mixmethod, conf.netname + '_' + conf.dataset)

        prefix = 'bs_' + str(conf.batch_size) + 'seed_' + str(conf.seed)

        if conf.weightfile:
            prefix = 'ft_' + prefix

        if not conf.pretrained:
            prefix = 'scratch_' + prefix

        if 'midlevel' in conf:
            if conf.midlevel:
                prefix += 'mid_'

        if 'mixmethod' in conf:
            if isinstance(conf.mixmethod, list):
                prefix += '_'.join(conf.mixmethod)
            else:
                prefix += conf.mixmethod + '_'
        if 'prob' in conf:
            prefix += '_p' + str(conf.prob)
        if 'beta' in conf:
            prefix += '_b' + str(conf.beta)

        outdir = os.path.join(outdir, prefix)

    ensure_dir(outdir)
    conf['outdir'] = outdir

    return conf


# check if dir exist, if not create new folder
def ensure_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print('{} is created'.format(dir_name))


def ensure_file(file_path):
    newpath = file_path
    if os.path.exists(file_path):
        timestr = datetime.now().strftime('%d-%m-%Y_%I_%M-%S_%p_')
        newpath = path.join(path.dirname(file_path), timestr + path.basename(file_path))
    return newpath
