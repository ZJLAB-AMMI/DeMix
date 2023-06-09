import imp
import os


def get_model(conf):
    if 'resnet' in conf.netname:
        net_type = 'resnet_ft'
    elif 'densenet' in conf.netname:
        net_type = 'densenet_ft'
    elif 'inception' in conf.netname:
        net_type = 'inception_ft'
    else:
        print('{} type not support'.format(conf.netname))

    src_file = os.path.join('networks', net_type + '.py')
    netimp = imp.load_source('networks', src_file)
    net = netimp.get_net(conf)
    return net


def count_params(net):
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
