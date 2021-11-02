from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import ImageFilter
import logging
import random

import paddle.vision.transforms as T


class GaussianBlur(object):
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_resolution(original_resolution):
    area = original_resolution[0] * original_resolution[1]
    return (160, 128) if area < 96 * 96 else (512, 480)


# cfg.AUG.TIMM_AUG默认值是False
# cfg.AUG.TIMM_AUG也没有提到
# cfg没提到FINETUNE，默认的FINETUNE.FINETUNE是False，FINETUNE.USE_TRAIN_AUG是False
# 因此RandomApply和RandomGrayscale要想办法
def build_transforms(cfg, is_train=True):

    normalize = T.Normalize(mean=cfg.INPUT.MEAN, std=cfg.INPUT.STD)

    transforms = None
    if is_train:
        if cfg.FINETUNE.FINETUNE and not cfg.FINETUNE.USE_TRAIN_AUG:

            crop = cfg.TRAIN.IMAGE_SIZE[0]
            precrop = crop + 32
            transforms = T.Compose([
                T.Resize((precrop, precrop),
                         interpolation=cfg.AUG.INTERPOLATION),
                T.RandomCrop((crop, crop)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize])
        else:
            aug = cfg.AUG
            scale = aug.SCALE
            ratio = aug.RATIO
            ts = [T.RandomResizedCrop(cfg.TRAIN.IMAGE_SIZE[0],
                                      scale=scale, ratio=ratio,
                                      interpolation=cfg.AUG.INTERPOLATION),
                  T.RandomHorizontalFlip()]

            cj = aug.COLOR_JITTER#0.4
            if cj[-1] > 0.0:
                ts.append([T.ColorJitter(*cj[:-1])])
                #ts.append(T.RandomApply([T.ColorJitter(*cj[:-1])], p=cj[-1]))

            ts.append(T.ToTensor())
            ts.append(normalize)

            transforms = T.Compose(ts)
    elif cfg.TEST.CENTER_CROP:
        transforms = T.Compose([
            T.Resize(int(cfg.TEST.IMAGE_SIZE[0] / 0.875),
                     interpolation=cfg.TEST.INTERPOLATION),
            T.CenterCrop(cfg.TEST.IMAGE_SIZE[0]),
            T.ToTensor(),
            normalize])
    else:
        transforms = T.Compose([
            T.Resize((cfg.TEST.IMAGE_SIZE[1],
                      cfg.TEST.IMAGE_SIZE[0]),
                     interpolation=cfg.TEST.INTERPOLATION),
            T.ToTensor(),
            normalize])
    return transforms
