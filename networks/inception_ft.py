from networks.inception import inception_v3


def get_net(conf):
    return inception_v3(pretrained=conf.pretrained)
