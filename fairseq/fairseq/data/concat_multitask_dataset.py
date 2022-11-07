# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import bisect

import numpy as np
from torch.utils.data.dataloader import default_collate

from . import FairseqDataset


class ConcatMultitaskDataset(FairseqDataset):
    @staticmethod
    def cumsum(sequence, sample_ratios):
        r, s = [], 0
        for e, ratio in zip(sequence, sample_ratios):
            curr_len = int(ratio * len(e))
            r.append(curr_len + s)
            s += curr_len
        return r

    def __init__(self, datasets, sample_ratios=1, separate_collater=False):
        super(ConcatMultitaskDataset, self).__init__()
        assert len(datasets) > 0, "datasets should not be an empty iterable"
        self.datasets = list(datasets)
        if isinstance(sample_ratios, int):
            sample_ratios = [sample_ratios] * len(self.datasets)
        self.sample_ratios = sample_ratios
        self.cumulative_sizes = self.cumsum(self.datasets, sample_ratios)
        self.real_sizes = [len(d) for d in self.datasets]
        self.separate_collater = separate_collater
        self.index2task = dict()

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self._get_dataset_and_sample_index(idx)
        return self.datasets[dataset_idx][sample_idx]

    def _get_dataset_and_sample_index(self, idx: int):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        sample_idx = sample_idx % self.real_sizes[dataset_idx]
        return dataset_idx, sample_idx

    def collater(self, samples, **extra_args):
        # For now only supports datasets with same underlying collater implementations
        if self.separate_collater and len(samples) > 0 and 'task_id' in samples[0]:
            # print('f[JJ] ConcatMultitaskDataset: samples', samples[0])
            data_idx = samples[0]['task_id']
            return self.datasets[data_idx].collater(samples, **extra_args)
        elif hasattr(self.datasets[0], 'collater'):
            # print(f'dataone collater', type(self.datasets[0]), type(samples))
            return self.datasets[0].collater(samples, **extra_args)
        else:
            # print(f'the other datasets.collater')
            return default_collate(samples, **extra_args)

    def size(self, idx: int):
        """
        Return an example's size as a float or tuple.
        """
        dataset_idx, sample_idx = self._get_dataset_and_sample_index(idx)
        return self.datasets[dataset_idx].size(sample_idx)

    def num_tokens(self, index: int):
        return np.max(self.size(index))

    def attr(self, attr: str, index: int):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, index)
        return getattr(self.datasets[dataset_idx], attr, None)

    @property
    def sizes(self):
        _dataset_sizes = []
        for ds, sr in zip(self.datasets, self.sample_ratios):
            if isinstance(ds.sizes, np.ndarray):
                # print('ds.sizes', type(ds.sizes), type(ds.sizes[0]), ds.sizes.shape, ds.sizes[0:10], sr)
                x = np.tile(ds.sizes, np.ceil(sr).astype(np.int32))
                slen = int(sr * len(ds))
                x = x[:slen]
                # ds = [1,2,3,4] rs = 1.5
                # x = [1,2,3,4,1,2,3,4]
                # slen = 1.5 * len(ds) = 6
                # x = [1,2,3,4,1,2]
                _dataset_sizes.append(x.astype(np.int32))
            else:
                # Only support underlying dataset with single size array.
                assert isinstance(ds.sizes, list)
                _dataset_sizes.append(np.tile(ds.sizes[0], sr))
                print(' the last dateset in the list', len(_dataset_sizes[-1]))
        for data_idx, dataset in enumerate(_dataset_sizes):
            for idx in range(len(dataset)):
                self.index2task[idx] = data_idx
        return np.concatenate(_dataset_sizes)

    @property
    def supports_prefetch(self):
        return all(d.supports_prefetch for d in self.datasets)

    def ordered_indices(self):
        """
        Returns indices sorted by length. So less padding is needed.
        """
        return np.argsort(self.sizes)

    def prefetch(self, indices):
        frm = 0
        for to, ds in zip(self.cumulative_sizes, self.datasets):
            real_size = len(ds)
            if getattr(ds, 'supports_prefetch', False):
                ds.prefetch([(i - frm) % real_size for i in indices if frm <= i < to])
            frm = to

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        for ds in self.datasets:
            if hasattr(ds, 'set_epoch'):
                ds.set_epoch(epoch)
