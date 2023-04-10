import torchvision.transforms as transforms

resizedict = {'224': 256, '448': 512, '112': 128}


def get_aircraft_transform(conf=None):
    return get_cub_transform(conf)


def get_car_transform(conf=None):
    return get_cub_transform(conf)


def get_cub_transform(conf=None):
    resize = 512
    cropsize = 448

    if conf and 'cropsize' in conf:
        cropsize = conf.cropsize
        resize = resizedict[str(cropsize)]

    if 'warp' in conf and conf.warp:
        resize = (resize, resize)
        cropsize = (cropsize, cropsize)

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    transform_train = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.RandomCrop(cropsize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    )

    transform_test = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(cropsize),
        transforms.ToTensor(),
        normalize
    ])

    return transform_train, transform_test
