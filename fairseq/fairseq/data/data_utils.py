# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import contextlib
import itertools
import logging
import os
import warnings

from typing import Tuple, Optional

import numpy as np
import torch


logger = logging.getLogger(__name__)


def infer_language_pair(path):
    """Infer language pair from filename: <split>.<lang1>-<lang2>.(...).idx"""
    src, dst = None, None
    for filename in os.listdir(path):
        parts = filename.split('.')
        if len(parts) >= 3 and len(parts[1].split('-')) == 2:
            return parts[1].split('-')
    return src, dst


def get_pad_size(values, input_shapes, pad_to_length=None):
    """
    Returns the pad size.

    On GPUs, pad to the max sequence length of a given input
    On TPUs, that would cause a lot of compilations and slow training.
      Thus, we pad to the "next sequence length as specified in the
      `input_shapes` argument.
    We assume `input_shapes` is an array of the form:
      [[batchsize0, seqlen0], [batchsize1, seqlen1], ...]
    sorted from shortest to longest sequence lengths, and unique in batch_sizes 

    e.g. [[512, 32], [256, 64], [128, 128]]
    """
    if input_shapes is None:
        size = max(v.size(0) for v in values)
        size = size if pad_to_length is None else max(size, pad_to_length)
        return (len(values), size)   # [batch_size, sequence_length]
    for batch_size, padlen in input_shapes:
        if len(values) == batch_size:
            return (batch_size, padlen) # [batch_size, sequence_length]
    else:
        # raise IndexError(
        #     'Encountered values with invalid length {}, input shapes were {}'
        #     .format(len(values), input_shapes)
        # )
        logger.info(f' [JJ] mini-batch size ={len(values)} not matched the pre-defined shape={input_shapes}, create a dumpy mini-batch')
        for batch_size, padlen in input_shapes:
            if len(values) < batch_size:
                return (batch_size, padlen)

def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, 
    move_eos_to_beginning=False, pad_to_length=None, input_shapes=None):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    # tpu-comment: fix input shape
    (batch_size, size) = get_pad_size(values, input_shapes, pad_to_length)
    res = values[0].new(batch_size, size).fill_(pad_idx)

    # print('-[JJ] data_utils.py: batch_size, size, input_shapes, left_pad', batch_size, size, input_shapes, left_pad)
    # print('-[JJ] data_utils.py res', res.shape)
    def copy_tensor(src, dst):
        assert dst.numel() == src.numel(), f'dst={dst.numel()} != src={src.numel()}, shape={dst.shape}, {src.shape}, src={src}, dst={dst}'
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        # src = v
        src = v if len(v) <= size else v[:size]
        dst = res[i][size - len(src):] if left_pad else res[i][:len(src)]
        copy_tensor(src, dst)
    return res


def load_indexed_dataset(path, dictionary=None, dataset_impl=None, combine=False, default='cached'):
    """A helper function for loading indexed datasets.

    Args:
        path (str): path to indexed dataset (e.g., 'data-bin/train')
        dictionary (~fairseq.data.Dictionary): data dictionary
        dataset_impl (str, optional): which dataset implementation to use. If
            not provided, it will be inferred automatically. For legacy indexed
            data we use the 'cached' implementation by default.
        combine (bool, optional): automatically load and combine multiple
            datasets. For example, if *path* is 'data-bin/train', then we will
            combine 'data-bin/train', 'data-bin/train1', ... and return a
            single ConcatDataset instance.
    """
    from fairseq.data.concat_dataset import ConcatDataset
    import fairseq.data.indexed_dataset as indexed_dataset

    datasets = []
    for k in itertools.count():
        path_k = path + (str(k) if k > 0 else '')

        dataset_impl_k = dataset_impl
        if dataset_impl_k is None:
            dataset_impl_k = indexed_dataset.infer_dataset_impl(path_k)

        # [JJ]
        print('[JJ] load indexed_dataset', dataset_impl_k, k, path_k)
        dataset = indexed_dataset.make_dataset(
            path_k,
            impl=dataset_impl_k or default,
            fix_lua_indexing=True,
            dictionary=dictionary,
        )
        if dataset is not None:
            print('[JJ] finished loading data', len(dataset))
        if dataset is None:
            break
        logger.info('loaded {} examples from: {}'.format(len(dataset), path_k))
        datasets.append(dataset)
        if not combine:
            break
    if len(datasets) == 0:
        return None
    elif len(datasets) == 1:
        return datasets[0]
    else:
        return ConcatDataset(datasets)


def load_shard_indexed_dataset(path, dictionary=None, dataset_impl=None, combine=False, default='cached'):
    """A helper function for loading indexed datasets.

    Args:
        path (str): path to indexed dataset (e.g., 'data-bin/train')
        dictionary (~fairseq.data.Dictionary): data dictionary
        dataset_impl (str, optional): which dataset implementation to use. If
            not provided, it will be inferred automatically. For legacy indexed
            data we use the 'cached' implementation by default.
        combine (bool, optional): automatically load and combine multiple
            datasets. For example, if *path* is 'data-bin/train', then we will
            combine 'data-bin/train', 'data-bin/train1', ... and return a
            single ConcatDataset instance.
    """
    from fairseq.data.concat_dataset import ConcatDataset
    import fairseq.data.indexed_dataset as indexed_dataset

    # datasets = []
    # for k in itertools.count():
    # path_k = path #+ (str(k) if k > 0 else '')

    dataset_impl_k = dataset_impl
    if dataset_impl_k is None:
        dataset_impl_k = indexed_dataset.infer_dataset_impl(path)

    # [JJ]
    print('[JJ] load indexed_dataset', dataset_impl_k, path)
    dataset = indexed_dataset.make_dataset(
        path,
        impl=dataset_impl_k or default,
        fix_lua_indexing=True,
        dictionary=dictionary,
    )
    if dataset is not None:
        print('[JJ] finished loading data', len(dataset))
    logger.info('loaded {} examples from: {}'.format(len(dataset), path))
    return dataset
    # datasets.append(dataset)
    # if not combine:
    #     break
    # if len(datasets) == 0:
    #     return None
    # elif len(datasets) == 1:
    #     return datasets[0]
    # else:
    #     return ConcatDataset(datasets)


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def collect_filtered(function, iterable, filtered):
    """
    Similar to :func:`filter` but collects filtered elements in ``filtered``.

    Args:
        function (callable): function that returns ``False`` for elements that
            should be filtered
        iterable (iterable): iterable to filter
        filtered (list): list to store filtered elements
    """
    for el in iterable:
        if function(el):
            yield el
        else:
            filtered.append(el)


def _filter_by_size_dynamic(indices, size_fn, max_positions, raise_exception=False):
    def compare_leq(a, b):
        return a <= b if not isinstance(a, tuple) else max(a) <= b

    def check_size(idx):
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            return size_fn(idx) <= max_positions
        elif isinstance(max_positions, dict):
            idx_size = size_fn(idx)
            assert isinstance(idx_size, dict)
            intersect_keys = set(max_positions.keys()) & set(idx_size.keys())
            return all(
                all(a is None or b is None or a <= b
                    for a, b in zip(idx_size[key], max_positions[key]))
                for key in intersect_keys
            )
        else:
            # Hacky as heck, for the specific case of multilingual training with RoundRobin.
            if isinstance(size_fn(idx), dict) and isinstance(max_positions, tuple):
                return all(
                    a is None or b is None or compare_leq(a, b)
                    for a, b in zip(size_fn(idx).values(), max_positions)
                )
            # For MultiCorpusSampledDataset, will generalize it later
            if not isinstance(size_fn(idx), Iterable):
                #print('[JJ] check_size ', idx, size_fn(idx), max_positions)
                return all(size_fn(idx) <= b for b in max_positions)
            return all(
                a is None or b is None or a <= b
                for a, b in zip(size_fn(idx), max_positions)
            )
    # [JJ]
    print('[JJ] data_utils.py: _filter_by_size_dynamic, indices', len(indices))
    ignored = []
    itr = collect_filtered(check_size, indices, ignored)
    indices = np.fromiter(itr, dtype=np.int64, count=-1)
    print('[JJ] data_utils.py: after filter, keep ', len(indices), ', ignored', len(ignored))
    return indices, ignored


def filter_by_size(indices, dataset, max_positions, raise_exception=False):
    """
    [deprecated] Filter indices based on their size.
    Use `FairseqDataset::filter_indices_by_size` instead.

    Args:
        indices (List[int]): ordered list of dataset indices
        dataset (FairseqDataset): fairseq dataset instance
        max_positions (tuple): filter elements larger than this size.
            Comparisons are done component-wise.
        raise_exception (bool, optional): if ``True``, raise an exception if
            any eldata_utils.ements are filtered (default: False).
    """
    # warnings.warn(
    #     'data_utils.filter_by_size is deprecated. '
    #     'Use `FairseqDataset::filter_indices_by_size` instead.',
    #     stacklevel=2
    # )
    if isinstance(max_positions, float) or isinstance(max_positions, int):
        if hasattr(dataset, 'sizes') and isinstance(dataset.sizes, np.ndarray):
            ignored = indices[dataset.sizes[indices] > max_positions].tolist()
            indices = indices[dataset.sizes[indices] <= max_positions]
        elif hasattr(dataset, 'sizes') and isinstance(dataset.sizes, list) and len(dataset.sizes) == 1:
            ignored = indices[dataset.sizes[0][indices] > max_positions].tolist()
            indices = indices[dataset.sizes[0][indices] <= max_positions]
        else:
            indices, ignored = _filter_by_size_dynamic(indices, dataset.size, max_positions)
    else:
        indices, ignored = _filter_by_size_dynamic(indices, dataset.size, max_positions)

    if len(ignored) > 0 and raise_exception:
        raise Exception((
            'Size of sample #{} is invalid (={}) since max_positions={}, '
            'skip this example with --skip-invalid-size-inputs-valid-test'
        ).format(ignored[0], dataset.size(ignored[0]), max_positions))
    if len(ignored) > 0:
        logger.warning((
            '{} samples have invalid sizes and will be skipped, '
            'max_positions={}, first few sample ids={}'
        ).format(len(ignored), max_positions, ignored[:10]))
    return indices


def batch_by_size(
    indices, num_tokens_fn, max_tokens=None, max_sentences=None,
    required_batch_size_multiple=1, fixed_shapes=None, sort_by_length=False
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be less than N or a multiple of N (default: 1).
        fixed_shapes (List[Tuple[int, int]], optional): if given, batches will
            only be created with the given shapes. *max_sentences* and
            *required_batch_size_multiple* will be ignored (default: None).
    """
    try:
        from fairseq.data.data_utils_fast import (
            batch_by_size_fast, batch_fixed_shapes_fast, batch_by_size_sort_by_length_fast
        )
    except ImportError:
        raise ImportError(
            'Please build Cython components with: `pip install --editable .` '
            'or `python setup.py build_ext --inplace`'
        )

    max_tokens = max_tokens if max_tokens is not None else -1
    max_sentences = max_sentences if max_sentences is not None else -1
    bsz_mult = required_batch_size_multiple

    if not isinstance(indices, np.ndarray):
        indices = np.fromiter(indices, dtype=np.int64, count=-1)

    if fixed_shapes is None:
        # return batch_by_size_fast(
        #     indices, num_tokens_fn, max_tokens, max_sentences, bsz_mult,
        # )
        if sort_by_length:
            print('[JJ] data_utils.py: batch_by_size_sort_by_length_fast')
            batches = batch_by_size_sort_by_length_fast(
                indices, num_tokens_fn, max_tokens, max_sentences, bsz_mult,
            )
        else:
            print('[JJ] data_utils.py: batch_by_size_fast')
            batches = batch_by_size_fast(
                indices, num_tokens_fn, max_tokens, max_sentences, bsz_mult,
            )

        print('[JJ] data_utils.py: done batching, num_batch', len(batches), 'num_samples=', len(indices))
        # # Debug 
        # cnt = 0
        # from collections import Counter
        # cnter = Counter()
        # sent_cnter = Counter()
        # for batch in batches:
        #     # total_tokens = sum(num_to)
        #     total_tokens = sum(num_tokens_fn(idx) for idx in batch)
        #     total_sents = len(batch)
        #     if num_tokens_fn(batch[0]) > 300:
        #         cnter['dae'] += total_tokens
        #         sent_cnter['dae'] += total_sents
        #     else:
        #         cnter['mt'] += total_tokens
        #         sent_cnter['mt'] += total_sents
        # print('token cnter', cnter)
        # print('sent_cnter', sent_cnter)
        # exit(0)

        #     cnter[total_tokens] += 1
        #     if total_tokens <= 0:
        #         print(f'[JJ] data_utils.py: batch is empty', batch, total_tokens)
        #         cnt += 1
        # print(f'[JJ] token cnter')
        # for w, c in cnter.most_common():
        #     print(f'  {w}\t{c}')
        # print(f'[JJ] {cnt}/ {len(batches)} empty batches')
        # print(f'[JJ] data_utils.py: batch_by_size_fast: sort={sort_by_length} len(indices)={len(indices)}, len(batches)={len(batches)}, bsz_mult={bsz_mult}, max_tokens={max_tokens}, max_sentences={max_sentences}')
        # for b in batches:
        #     print(f'sum_tokens', sum(num_tokens_fn(bi) for bi in b))
        return batches
    else:
        fixed_shapes = np.array(fixed_shapes, dtype=np.int64)
        sort_order = np.lexsort([
            fixed_shapes[:, 1].argsort(),  # length
            fixed_shapes[:, 0].argsort(),  # bsz
        ])
        fixed_shapes_sorted = fixed_shapes[sort_order]
        print('shape ', fixed_shapes_sorted, num_tokens_fn, len(indices))
        return batch_fixed_shapes_fast(indices, num_tokens_fn, fixed_shapes_sorted)


def batch_by_size_tpu(
    indices, num_tokens_fn, input_shapes
):
    """
    tpu-comment: varying input shapes cause compilations and slow TPU training.
     There is a trade-off between
     * allow varying input shapes and lose time to compilations
     * fix input shapes by padding and lose time by wasting flops

    It is generally up to experimentation to determine the optimal input_shapes
    parameter that results in the best performance.
    """
    batches = [[] for _ in input_shapes]
    for idx in indices:
        sample_len = num_tokens_fn(idx)
        for j, (batch_size, padlen) in enumerate(input_shapes):
            if padlen < sample_len:
                continue
            batches[j].append(idx)
            if len(batches[j]) == batch_size:
                yield batches[j]
                batches[j] = []
            break

def post_process(sentence: str, symbol: str):
    if symbol == "sentencepiece":
        print('[JJ] before processing', sentence)
        sentence = sentence.replace(" ", "").replace("\u2581", " ").strip()
        print('[JJ] after processing', sentence)
    elif symbol == 'wordpiece':
        sentence = sentence.replace(" ", "").replace("_", " ").strip()
    elif symbol == 'letter':
        sentence = sentence.replace(" ", "").replace("|", " ").strip()
    elif symbol == "_EOW":
        sentence = sentence.replace(" ", "").replace("_EOW", " ").strip()
    elif symbol is not None and symbol != 'none':
        sentence = (sentence + " ").replace(symbol, "").rstrip()
    return sentence

def compute_mask_indices(
        shape: Tuple[int, int],
        padding_mask: Optional[torch.Tensor],
        mask_prob: float,
        mask_length: int,
        mask_type: str = "static",
        mask_other: float = 0.0,
        min_masks: int = 0,
        no_overlap: bool = False,
        min_space: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = np.random.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = np.random.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = np.random.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []
            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e-length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start-min_space+1))
                if e - span_start - keep_length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter((e - s if e-s >= length+min_space else 0 for s, e in parts), np.int)
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len  = sz - num_mask - 1

            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True

    return mask
