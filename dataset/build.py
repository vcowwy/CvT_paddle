from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

from timm_paddle.data.loader import create_loader

import paddle
import paddle.vision.datasets as datasets

from .transformas import build_transforms
from .samplers import RASampler
from .t2p import DataLoader, DistributedSampler


def build_dataset(cfg, is_train):
    dataset = None
    if 'imagenet' in cfg.DATASET.DATASET:#True
        dataset = _build_imagenet_dataset(cfg, is_train)
    else:
        raise ValueError('Unkown dataset: {}'.format(cfg.DATASET.DATASET))
    return dataset


def _build_image_folder_dataset(cfg, is_train):
    transforms = build_transforms(cfg, is_train)

    dataset_name = cfg.DATASET.TRAIN_SET if is_train else cfg.DATASET.TEST_SET
    dataset = datasets.ImageFolder(
        os.path.join(cfg.DATASET.ROOT, dataset_name), transforms
    )
    logging.info(
        '=> load samples: {}, is_train: {}'
            .format(len(dataset), is_train)
    )
    return dataset


def _build_imagenet_dataset(cfg, is_train):
    transforms = build_transforms(cfg, is_train)

    dataset_name = cfg.DATASET.TRAIN_SET if is_train else cfg.DATASET.TEST_SET
    dataset = datasets.ImageFolder(
        os.path.join(cfg.DATASET.ROOT, dataset_name), transforms
    )

    return dataset


# cfg.AUG.TIMM_AUG.USE_LOADER是True，如果is_train是True就要看看timm.data.create_loader
def build_dataloader(cfg, is_train=True, distributed=False):
    if is_train:
        batch_size_per_gpu = cfg.TRAIN.BATCH_SIZE_PER_GPU
        shuffle = True
    else:
        batch_size_per_gpu = cfg.TEST.BATCH_SIZE_PER_GPU
        shuffle = False

    dataset = build_dataset(cfg, is_train)

    if distributed:
        if is_train and cfg.DATASET.SAMPLER == 'repeated_aug':
            logging.info('=> use repeated aug sampler')
            sampler = RASampler(dataset, shuffle=shuffle)
        else:
            sampler = DistributedSampler(
                dataset, shuffle=shuffle
            )
        shuffle = False
    else:
        sampler = None

    if cfg.AUG.TIMM_AUG.USE_LOADER and is_train:
        logging.info('=> use timm loader for training')
        timm_cfg = cfg.AUG.TIMM_AUG
        data_loader = create_loader(
            dataset,
            input_size=cfg.TRAIN.IMAGE_SIZE[0],#224
            batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,#256
            is_training=True,
            use_prefetcher=True,
            no_aug=False,
            re_prob=timm_cfg.RE_PROB,#0.25
            re_mode=timm_cfg.RE_MODE,#pixel
            re_count=timm_cfg.RE_COUNT,#1
            re_split=timm_cfg.RE_SPLIT,#false
            scale=cfg.AUG.SCALE,#0.08,1.0
            ratio=cfg.AUG.RATIO,# 3.0 / 4.0, 4.0 / 3.0
            hflip=timm_cfg.HFLIP,#0.5
            vflip=timm_cfg.VFLIP,#0.0
            color_jitter=timm_cfg.COLOR_JITTER,#0.4
            auto_augment=timm_cfg.AUTO_AUGMENT,#rand-m9-mstd0.5-inc1
            num_aug_splits=0,
            interpolation=timm_cfg.INTERPOLATION,#bicubic
            mean=cfg.INPUT.MEAN,#[0.485, 0.456, 0.406]
            std=cfg.INPUT.STD,#[0.229, 0.224, 0.225]
            num_workers=cfg.WORKERS,#6
            distributed=distributed,
            collate_fn=None,
            pin_memory=cfg.PIN_MEMORY,#True
            use_multi_epochs_loader=True)
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size_per_gpu,
            shuffle=shuffle,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            sampler=sampler,
            drop_last=True if is_train else False)

    return data_loader
