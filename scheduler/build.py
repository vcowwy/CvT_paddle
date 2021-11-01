from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .learning_rate.scheduler_factory import create_scheduler


# cfg.TRAIN.LR_SCHEDULER.METHOD用的是timm cosin
def build_lr_scheduler(cfg, optimizer, begin_epoch):
    if 'METHOD' not in cfg.TRAIN.LR_SCHEDULER:
        raise ValueError('Please set TRAIN.LR_SCHEDULER.METHOD!')

    elif cfg.TRAIN.LR_SCHEDULER.METHOD == 'timm':
        args = cfg.TRAIN.LR_SCHEDULER.ARGS
        lr_scheduler, _ = create_scheduler(args, optimizer)
        lr_scheduler.step(begin_epoch)

    else:
        raise ValueError('Unknown lr scheduler: {}'.format(cfg.TRAIN.
            LR_SCHEDULER.METHOD))

    return lr_scheduler
