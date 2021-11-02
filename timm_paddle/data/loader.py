import paddle
import numpy as np

from .transforms_factory import create_transform
from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .random_erasing import RandomErasing
from .mixup import FastCollateMixup


def fast_collate(batch):
    assert isinstance(batch[0], tuple)
    batch_size = len(batch)
    if isinstance(batch[0][0], tuple):

        inner_tuple_size = len(batch[0][0])
        flattened_batch_size = batch_size * inner_tuple_size
        targets = paddle.zeros(flattened_batch_size, dtype=paddle.int64)
        tensor = paddle.zeros((flattened_batch_size, *batch[0][0][0].shape), dtype=paddle.uint8)
        for i in range(batch_size):
            assert len(batch[i][0]) == inner_tuple_size  # all input tensor tuples must be same length
            for j in range(inner_tuple_size):
                targets[i + j * batch_size] = batch[i][1]
                tensor[i + j * batch_size] += paddle.to_tensor(batch[i][0][j])
        return tensor, targets
    elif isinstance(batch[0][0], np.ndarray):
        targets = paddle.to_tensor([b[1] for b in batch], dtype=paddle.int64)
        assert len(targets) == batch_size
        tensor = paddle.zeros((batch_size, *batch[0][0].shape), dtype=paddle.uint8)
        for i in range(batch_size):
            tensor[i] += paddle.to_tensor(batch[i][0])
        return tensor, targets
    elif isinstance(batch[0][0], paddle.Tensor):
        targets = paddle.to_tensor([b[1] for b in batch], dtype=paddle.int64)
        assert len(targets) == batch_size
        tensor = paddle.zeros((batch_size, *batch[0][0].shape), dtype=paddle.uint8)
        for i in range(batch_size):
            tensor[i].copy_(batch[i][0])
        return tensor, targets
    else:
        assert False


class PrefetchLoader:

    def __init__(self,
                 loader,
                 mean=IMAGENET_DEFAULT_MEAN,
                 std=IMAGENET_DEFAULT_STD,
                 fp16=False,
                 re_prob=0.,
                 re_mode='const',
                 re_count=1,
                 re_num_splits=0):
        self.loader = loader
        self.mean = paddle.to_tensor([x * 255 for x in mean]).cuda().view(1, 3, 1, 1)
        self.std = paddle.to_tensor([x * 255 for x in std]).cuda().view(1, 3, 1, 1)
        self.fp16 = fp16
        if fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()
        if re_prob > 0.:
            self.random_erasing = RandomErasing(
                probability=re_prob, mode=re_mode, max_count=re_count, num_splits=re_num_splits)
        else:
            self.random_erasing = None

    def __iter__(self):
        #stream = paddle.cuda.Stream()
        first = True

        for next_input, next_target in self.loader:
            #with paddle.cuda.stream(stream):
            next_input = next_input.cuda(non_blocking=True)
            next_target = next_target.cuda(non_blocking=True)
            if self.fp16:
                next_input = next_input.half().sub_(self.mean).div_(self.std)
            else:
                next_input = next_input.float().sub_(self.mean).div_(self.std)
            if self.random_erasing is not None:
                next_input = self.random_erasing(next_input)

            if not first:
                yield input, target
            else:
                first = False

            #paddle.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset

    @property
    def mixup_enabled(self):
        if isinstance(self.loader.collate_fn, FastCollateMixup):
            return self.loader.collate_fn.mixup_enabled
        else:
            return False

    @mixup_enabled.setter
    def mixup_enabled(self, x):
        if isinstance(self.loader.collate_fn, FastCollateMixup):
            self.loader.collate_fn.mixup_enabled = x


def create_loader(
        dataset,
        input_size,
        batch_size,
        is_training=False,
        use_prefetcher=True,
        no_aug=False,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_split=False,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        num_aug_splits=0,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_workers=1,
        distributed=False,
        crop_pct=None,
        collate_fn=None,
        pin_memory=False,
        fp16=False,
        tf_preprocessing=False,
        use_multi_epochs_loader=False,
        persistent_workers=True,
):
    re_num_splits = 0

    dataset.transform = create_transform(
        input_size,
        is_training=is_training,#True
        use_prefetcher=use_prefetcher,#True
        no_aug=no_aug,#False
        scale=scale,#0.08, 1.0
        ratio=ratio,# 3.0 / 4.0, 4.0 / 3.0
        hflip=hflip,#0.5
        vflip=vflip,#0.0
        color_jitter=color_jitter,#0.4
        auto_augment=auto_augment,#rand-m9-mstd0.5-inc1
        interpolation=interpolation,#bicubic
        mean=mean,#[0.485, 0.456, 0.406]
        std=std,#[0.229, 0.224, 0.225]
        crop_pct=crop_pct,#None
        tf_preprocessing=tf_preprocessing,#False
        re_prob=re_prob,#0.25
        re_mode=re_mode,#pixel
        re_count=re_count,#1
        re_num_splits=re_num_splits,#0
        separate=num_aug_splits > 0,#0. false
    )

    sampler = None
    if distributed and not isinstance(dataset, paddle.io.IterableDataset):
        if is_training:
            sampler = paddle.io.DistributedBatchSampler(dataset, batch_size=batch_size)

    if collate_fn is None:
        collate_fn = fast_collate if use_prefetcher else None

    loader_class = paddle.io.DataLoader

    if use_multi_epochs_loader:
        loader_class = MultiEpochsDataLoader

    loader_args = dict(
        batch_size=batch_size,
        shuffle=not isinstance(dataset, paddle.io.IterableDataset) and sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training,
        persistent_workers=persistent_workers)
    try:
        loader = loader_class(dataset, **loader_args)
    except TypeError as e:
        loader_args.pop('persistent_workers')  # only in Pypaddle 1.7+
        loader = loader_class(dataset, **loader_args)
    if use_prefetcher:
        prefetch_re_prob = re_prob if is_training and not no_aug else 0.
        loader = PrefetchLoader(
            loader,
            mean=mean,
            std=std,
            fp16=fp16,
            re_prob=prefetch_re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits
        )

    return loader


class MultiEpochsDataLoader(paddle.io.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
