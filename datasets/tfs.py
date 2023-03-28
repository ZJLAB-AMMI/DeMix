import torchvision.transforms as transforms

resizedict = {'224': 256, '448': 512, '112': 128}


def get_aircraft_transform(conf=None):
    return get_cub_transform(conf)


def get_car_transform(conf=None):
    return get_cub_transform(conf)


def get_nabirds_transform(conf=None):
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

    if conf.debug:
        transform_train = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.RandomCrop(cropsize),
                transforms.ToTensor(),
            ]
        )
    else:
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


def get_cifar_transform(conf=None):
    image_normalizer = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    transform_train = None
    if conf.is_train:
        transform_train = transforms.Compose([])

        transform_train.transforms.append(transforms.RandomCrop(conf.cropsize, padding=4))

        if conf.data_augmentation:
            transform_train.transforms.append(transforms.RandomHorizontalFlip())

        transform_train.transforms.append(transforms.ToTensor())

        transform_train.transforms.append(image_normalizer)

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            image_normalizer
        ]
    )

    return transform_train, transform_test


def get_imagenet_transform(conf=None):
    image_normalizer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    transform_train = None
    if conf.is_train:
        transform_train = transforms.Compose([])

        transform_train.transforms.append(transforms.RandomResizedCrop(224))

        if conf.data_augmentation:
            transform_train.transforms.append(transforms.RandomHorizontalFlip())

        transform_train.transforms.append(transforms.ToTensor())

        transform_train.transforms.append(image_normalizer)

    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            image_normalizer
        ]
    )

    return transform_train, transform_test
