# Use the Detection Transformer as a Data Augmenter

PyTorch implementation of DeMix | [paper](https://arxiv.org/pdf/2304.04554.pdf)

## Method Overview

![DeMix](./imgs/overview.jpg)

## Setup

### Install Package Dependencies

```
pip install -r requirements.txt
```

### Datasets

***download the fine-grained datasets:***

[CUB-200-2011](https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1)

Stanford-Cars
[devkit](https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz),
[train](https://ai.stanford.edu/~jkrause/car196/cars_train.tgz),
[test](https://ai.stanford.edu/~jkrause/car196/cars_test.tgz),
[test_annos_withlabels](https://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat)

[FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz)

### DETR object detection

1, import the function:```from datasets.dataset_process import compute_detr_res```

2, run the function: ```compute_detr_res(dataset_name='cub', datadir='cub data dir')  # ['cub', 'car', 'aircraft']```

## Training
```
python demix.py
    --dataset='cub' # ['cub', 'car', 'aircraft']
    --netname='resnet18' # ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'inception_v3', 'densenet121']
    --mixmethod='detrmix' # ['detrmix', 'saliencymix', 'mixup', 'cutmix']
    --pretrained=1 # if training from scratch, set pretrained=0
```

## Citation
If you find this code useful, please kindly cite  

@article{wang2023use,

  title={Use the Detection Transformer as a Data Augmenter},
  
  author={Wang, Luping and Liu, Bin},
  
  journal={arXiv preprint arXiv:2304.04554},
  
  year={2023}
  
}

## Acknowledgment

This code is based on the [SnapMix](https://github.com/Shaoli-Huang/SnapMix.git).

## Contact

If you have any questions or suggestions, please feel free to contact wangluping/liubin@zhejianglab.com.
