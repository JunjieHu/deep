# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
from . import BaseWrapperDataset
import numpy as np

class DropLastTokenDataset(BaseWrapperDataset):
    '''
    Replace the last token by another token specified by self.token.
    '''
    def __init__(self, dataset, token, max_len=128):
        super().__init__(dataset)
        self.token = token
        self.max_len = max_len
        # print(' *[JJ] DropLastTokenDataset.py: dataset.sizes', len(dataset.sizes), dataset.sizes)
        self._sizes = np.zeros(len(dataset.sizes))

        # check
        # for i, s in enumerate(dataset.sizes):
        #     if s > max_len:
        #         print(f' *[JJ] dataset.sizes[{i}] ={s}> {max_len}')
                
        for i, s in enumerate(dataset.sizes):
            self._sizes[i] = min(s, max_len)
        
        # # check
        # for i, s in enumerate(self._sizes):
        #     if s > max_len:
        #         print(f' *[JJ] self._sizes[{i}] ={s}> {max_len}')
        # print(' *[JJ] DropLastTokenDataset.py: self._sizes', len(dataset.sizes), max_len, list(self._sizes))

    def __getitem__(self, index):
        item = self.dataset[index]
        lidx = min(self.max_len, len(item))
        return torch.cat([item[:lidx-1], item.new_tensor([self.token])])

    def size(self, index):
        # n = self.dataset.size(index)
        # return min(self.max_len, n)
        return self._sizes[index]

    @property
    def sizes(self):
        return self._sizes
    