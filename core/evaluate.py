from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import torch


@paddle.no_grad()
def accuracy(output, target, topk=(1,)):
    if isinstance(output, list):
        output = output[-1]

    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.equal(target.reshape([1, -1]).expand_as(pred))

    res = []
    for k in topk:
        correct_k = paddle.to_tensor(correct[:k].reshape([-1]), dtype=paddle.float32).sum(0, keepdim=True)
        res.append(correct_k*(100.0 / batch_size))
    return res
