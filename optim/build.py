from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.optimizer as optim


def _is_depthwise(m):
    return (
            isinstance(m, paddle.nn.Conv2D)
            and m._parameters['groups'] == m._parameters['in_channels']
            and m._groups == m.out_channels
    )


def set_wd(cfg, model):
    without_decay_list = cfg.TRAIN.WITHOUT_WD_LIST
    without_decay_depthwise = []
    without_decay_norm = []
    for m in model.sublayers():
        if _is_depthwise(m) and 'dw' in without_decay_list:
            without_decay_depthwise.append(m.weight)
        elif isinstance(m, paddle.nn.BatchNorm2D) and 'bn' in without_decay_list:
            without_decay_norm.append(m.weight)
            without_decay_norm.append(m.bias)
        elif isinstance(m, paddle.nn.GroupNorm) and 'gn' in without_decay_list:
            without_decay_norm.append(m.weight)
            without_decay_norm.append(m.bias)
        elif isinstance(m, paddle.nn.LayerNorm) and 'ln' in without_decay_list:
            without_decay_norm.append(m.weight)
            without_decay_norm.append(m.bias)

    with_decay = []
    without_decay = []

    skip = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()

    skip_keys = {}
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keys = model.no_weight_decay_keywords()

    for n, p in model.named_parameters():
        ever_set = False

        if p.stop_gradient is True:
            continue

        skip_flag = False
        if n in skip:
            print('=> set {} wd to 0'.format(n))
            without_decay.append(p)
            skip_flag = True
        else:
            for i in skip:
                if i in n:
                    print('=> set {} wd to 0'.format(n))
                    without_decay.append(p)
                    skip_flag = True

        if skip_flag:
            continue

        for i in skip_keys:
            if i in n:
                print('=> set {} wd to 0'.format(n))

        if skip_flag:
            continue

        for pp in without_decay_depthwise:
            if p is pp:
                if cfg.VERBOSE:
                    print('=> set depthwise({}) wd to 0'.format(n))
                without_decay.append(p)
                ever_set = True
                break

        for pp in without_decay_norm:
            if p is pp:
                if cfg.VERBOSE:
                    print('=> set norm({}) wd to 0'.format(n))
                without_decay.append(p)
                ever_set = True
                break

        if not ever_set and 'bias' in without_decay_list and n.endswith('.bias'):
            if cfg.VERBOSE:
                print('=> set bias({}) wd to 0'.format(n))
            without_decay.append(p)

        elif not ever_set:
            with_decay.append(p)
    params = [{'params': with_decay},
              {'params': without_decay, 'weight_decay': 0.0}]
    return params


#cfg.TRAIN.OPTIMIZER用的是adamW
def build_optimizer(cfg, model):
    optimizer = None
    params = set_wd(cfg, model)
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.Momentum(
            parameters=params,
            learning_rate=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            use_nesterov=cfg.TRAIN.NESTEROV)
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            parameters=params,
            learning_rate=cfg.TRAIN.LR,
            weight_decay=cfg.TRAIN.WD)
    elif cfg.TRAIN.OPTIMIZER == 'adamW':
        optimizer = optim.AdamW(
            parameters=params,
            learning_rate=cfg.TRAIN.LR,
            weight_decay=cfg.TRAIN.WD,
        )
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = optim.RMSProp(
            parameters=params,
            # filter(lambda p: p.requires_grad, model.parameters()),
            learning_rate=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            rho=cfg.TRAIN.RMSPROP_ALPHA,
            centered=cfg.TRAIN.RMSPROP_CENTERED
        )

    return optimizer
