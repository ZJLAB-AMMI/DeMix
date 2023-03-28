import warnings

from utils import get_config, set_env
from utils import get_dataloader


def main(conf):
    warnings.filterwarnings("ignore")
    train_loader, val_loader = get_dataloader(conf)


if __name__ == '__main__':
    gpu_ids = '7'
    netname_set = ['resnet50']
    dataset_set = ['cub']
    mixmethod_set = ['detrmix']
    for netname in netname_set:
        for dataset in dataset_set:
            for mixmethod in mixmethod_set:
                conf = get_config(dataset=dataset, netname=netname, mixmethod=mixmethod, gpu_ids=gpu_ids)
                conf.batch_size = 16
                conf.workers = 16
                conf.weight_decay = 1e-4
                set_env(conf)
                main(conf)
