# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import math

from . import data_utils, FairseqDataset


def collate(
    samples,
    pad_idx,
    eos_idx,
    vocab,
    left_pad_source=False,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    input_shapes=None,
):
    assert input_feeding
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=None,  # use eos_idx of each sample instead of vocab.eos()
            left_pad=left_pad,
            move_eos_to_beginning=move_eos_to_beginning,
            pad_to_length=pad_to_length,
            input_shapes=input_shapes,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    # print('[JJ] context_denoising_dataset.py: before merge', [len(s['source']) for s in samples])
    src_tokens = merge(
        'source', left_pad=left_pad_source,
        pad_to_length=pad_to_length['source'] if pad_to_length is not None else None,
    )
    # print('[JJ] context_denoising_dataset.py: after merge', src_tokens.shape)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)
    # print('[JJ] denoising_dataset.py: after index_select', src_tokens.shape)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge(
            'target', left_pad=left_pad_target,
            pad_to_length=pad_to_length['target'] if pad_to_length is not None else None,
        )
        # print(' * [JJ] finish merging target')
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length['target'] if pad_to_length is not None else None,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    mask_index = [s['mask_index'] for s in samples]
    context = [s['context'] for s in samples]
#    print('[JJ] context_denoising_dataset.py: collate batch:', ntokens, src_tokens.shape, target.shape)
#    # print('[JJ] context_denoising_dataset.py: collate src_tokens:', src_tokens)
#
#    print('[JJ] context_denoising_dataset.py: collate')
#    for jj in range(2):
#        src = src_tokens[jj].tolist()
#        tgt = target[jj].tolist()
#        print(f'src[{jj}]=', (''.join([vocab[s] for s in src]).replace('▁', ' ')))
#        print(f'src[{jj}] toks=', [(vocab[s], t) for t,s in enumerate(src)])
#        print(f'tgt[{jj}]=', (''.join([vocab[s] for s in tgt]).replace('▁', ' ')))
#        print(f'tgt[{jj}] toks=', [(vocab[s], t) for t,s in enumerate(tgt)])
#        print(f'mask[{jj}]=', mask_index[jj].tolist())
#        print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # exit(0)


    batch = {
        'id': id,
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
        'nsentences': samples[0]['source'].size(0),
        'sort_order': sort_order,
        'mask_index': mask_index,
        'context': context,
    }
    # print('[JJ] denoising_dataset.py: net_input src_lengths', src_lengths)
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    return batch


class ContextDenoisingTranslationDataset(FairseqDataset):
    """
    A wrapper around TokenBlockDataset for BART dataset.

    Args:
        dataset (TokenBlockDataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (~fairseq.data.Dictionary): vocabulary
        mask_idx (int): dictionary index used for masked token
        mask_whole_words: only mask whole words. This should be a byte mask
            over vocab indices, indicating whether it is the beginning of a
            word. We will extend any mask to encompass the whole word.
        shuffle (bool, optional): shuffle the elements before batching.
          Default: ``True``
        seed: Seed for random number generator for reproducibility.
        args: argparse arguments.
    """

    def __init__(
        self,
        src_dataset,
        src_sizes,
        tgt_dataset,
        tgt_sizes,
        mask_index_dataset,
        mask_index_sizes,
        task_type,
        vocab,
        mask_idx,
        mask_whole_words,
        shuffle,
        seed,
        args,
        ctxt_dataset=None,
        ctxt_sizes=None,
        eos=None,
        item_transform_func=None,
        input_shapes=None,
    ):
        self.src = src_dataset
        self.src_sizes = src_sizes
        self.sizes = src_sizes
        self.tgt = tgt_dataset
        self.tgt_sizes = tgt_sizes
        self.mask_index = mask_index_dataset
        self.mask_index_sizes = mask_index_sizes
        self.task_type = task_type
        self.ctxt = ctxt_dataset
        self.ctxt_sizes = ctxt_sizes
        self.args = args

        self.vocab = vocab
        self.shuffle = shuffle
        self.seed = seed
        self.mask_idx = mask_idx
        self.mask_whole_word = mask_whole_words
        self.mask_ratio = args.mask
        self.random_ratio = args.mask_random
        self.insert_ratio = args.insert
        self.rotate_ratio = args.rotate
        self.permute_sentence_ratio = args.permute_sentences
        self.eos = (eos if eos is not None else vocab.eos())
        self.item_transform_func = item_transform_func
        # tpu-comment: pass input_shapes argument
        self.input_shapes = input_shapes

        if args.bpe != 'gpt2':
            self.full_stop_index = self.vocab.eos()
        else:
            assert args.bpe == 'gpt2'
            self.full_stop_index = self.vocab.index('13')

        self.replace_length = args.replace_length
        if self.replace_length not in [-1, 0, 1]:
            raise ValueError(f'invalid arg: replace_length={self.replace_length}')
        if args.mask_length not in ['subword', 'word', 'span-poisson']:
            raise ValueError(f'invalid arg: mask-length={args.mask_length}')
        if args.mask_length == 'subword' and args.replace_length not in [0, 1]:
            raise ValueError(f'if using subwords, use replace-length=1 or 0')

        self.mask_span_distribution = None
        if args.mask_length == 'span-poisson':
            _lambda = args.poisson_lambda

            lambda_to_the_k = 1
            e_to_the_minus_lambda = math.exp(-_lambda)
            k_factorial = 1
            ps = []
            for k in range(0, 128):
                ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
                lambda_to_the_k *= _lambda
                k_factorial *= (k + 1)
                if ps[-1] < 0.0000001:
                    break
            ps = torch.FloatTensor(ps)
            self.mask_span_distribution = torch.distributions.Categorical(ps)

        # print(' =[JJ] mask_length', args.mask_length, self.mask_span_distribution)
        self.epoch = 0

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            source = self.src[index]
            # target = self.tgt[index] if self.tgt is not None else source.clone()
            # target = source.clone()
            target = self.tgt[index]
            # target = source.clone()
            mask_index = self.mask_index[index]
            ctxt = None
            # ctxt = self.ctxt[index]

            # print('\n\n==========================')
            # print('index', index)
            # print('source', source)
            # print('target', target)
            # print(f'src[{index}]=', [(self.vocab[ss],tt) for (tt,ss) in enumerate(source.tolist())])
            # print(f'tgt[{index}]=', [(self.vocab[ss],tt) for (tt,ss) in enumerate(target.tolist())])
            # mask0 = mask_index.tolist()
            # print(f'mask[{index}]=', mask0)
            # print('mask_index', mask_index)
            # exit(0)

            assert source[-1] == self.eos, f'source[-1]={source[-1]}, self.eos={self.eos}'

            if self.permute_sentence_ratio > 0.0:
                source = self.permute_sentences(source, self.permute_sentence_ratio)
                # print('  *[JJ] permute_sentences', source.shape)

            if self.mask_ratio > 0:
                source = self.add_whole_word_mask(source, self.mask_ratio, mask_index)
                # print('  *[JJ] add_whole_word_mask', source.shape)

            if self.insert_ratio > 0:
                source = self.add_insertion_noise(source, self.insert_ratio)
                # print('  *[JJ] add_insertion_noise', source.shape)

            if self.rotate_ratio > 0.0 and np.random.random() < self.rotate_ratio:
                source = self.add_rolling_noise(source)
                # print('  *[JJ] add_rolling_noise', source.shape)
        # there can additional changes to make:
        if self.item_transform_func is not None:
            source, target = self.item_transform_func(source, target)

        assert (source >= 0).all()
        assert (source[1:-1] >= 1).all()
        assert (source <= len(self.vocab)).all()
        assert source[0] == self.vocab.bos()
        assert source[-1] == self.eos
        # print('[JJ] denoising_dataset.py', source.shape, target.shape)
        return {
            'id': index,
            'source': source,
            'target': target,
            'mask_index': mask_index,
            'context': ctxt,
        }

    def __len__(self):
        return len(self.src)

    def permute_sentences(self, source, p=1.0):
        full_stops = (source == self.full_stop_index)
        # Pretend it ends with a full stop so last span is a sentence
        full_stops[-2] = 1

        # Tokens that are full stops, where the previous token is not
        sentence_ends = (full_stops[1:] * ~full_stops[:-1]).nonzero(as_tuple=False) + 2
        result = source.clone()

        num_sentences = sentence_ends.size(0)
        num_to_permute = math.ceil((num_sentences * 2 * p) / 2.0)
        substitutions = torch.randperm(num_sentences)[:num_to_permute]
        ordering = torch.arange(0, num_sentences)
        ordering[substitutions] = substitutions[torch.randperm(num_to_permute)]

        # Ignore <bos> at start
        index = 1
        for i in ordering:
            sentence = source[(sentence_ends[i - 1] if i > 0 else 1):sentence_ends[i]]
            result[index:index + sentence.size(0)] = sentence
            index += sentence.size(0)
        return result

    def word_starts(self, source):
        if self.mask_whole_word is not None:
            is_word_start = self.mask_whole_word.gather(0, source)
        else:
            is_word_start = torch.ones(source.size())
        is_word_start[0] = 0
        is_word_start[-1] = 0
        return is_word_start

    def add_whole_word_mask(self, source, p, mask_index):
        is_word_start = self.word_starts(source)
        if self.args.add_entity_mask:
           is_word_start.index_fill_(0, mask_index, 0)  # mask those entity words
           p = max(0, p - len(mask_index) / len(source))
        num_to_mask = int(math.ceil(is_word_start.float().sum() * p))
        # [JJ]
        # num_to_mask = max(0, num_to_mask - mask_index.size(0))

        if num_to_mask < 0 :
            # print(f'-[JJ] {num_to_mask}')
            return source

        num_inserts = 0
        if num_to_mask == 0:
            return source
        # print(' =[JJ] num_to_mask', num_to_mask)

        if self.mask_span_distribution is not None:
            lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))

            # Make sure we have enough to mask
            cum_length = torch.cumsum(lengths, 0)
            while cum_length[-1] < num_to_mask:
                lengths = torch.cat([lengths, self.mask_span_distribution.sample(sample_shape=(num_to_mask,))], dim=0)
                cum_length = torch.cumsum(lengths, 0)

            # Trim to masking budget
            i = 0
            while cum_length[i] < num_to_mask:
                i += 1
            lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
            num_to_mask = i + 1
            lengths = lengths[:num_to_mask]

            # Handle 0-length mask (inserts) separately
            lengths = lengths[lengths > 0]
            num_inserts = num_to_mask - lengths.size(0)
            num_to_mask -= num_inserts
            if num_to_mask == 0:
                return self.add_insertion_noise(source, num_inserts / source.size(0))

            assert (lengths > 0).all()
        else:
            lengths = torch.ones((num_to_mask,)).long()
        assert is_word_start[-1] == 0
        word_starts = is_word_start.nonzero(as_tuple=False)
        indices = word_starts[torch.randperm(word_starts.size(0))[:num_to_mask]].squeeze(1)
        mask_random = torch.FloatTensor(num_to_mask).uniform_() < self.random_ratio

        source_length = source.size(0)
        assert source_length - 1 not in indices
        to_keep = torch.ones(source_length, dtype=torch.bool)
        is_word_start[-1] = 255 # acts as a long length, so spans don't go over the end of doc
        if self.replace_length == 0:
            to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            source[indices] = self.mask_idx
            source[indices[mask_random]] = torch.randint(1, len(self.vocab), size=(mask_random.sum(),))

        if self.mask_span_distribution is not None:
            assert len(lengths.size()) == 1
            assert lengths.size() == indices.size()
            lengths -= 1
            while indices.size(0) > 0:
                assert lengths.size() == indices.size()
                lengths -= is_word_start[indices + 1].long()
                uncompleted = lengths >= 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                lengths = lengths[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx
                    source[indices[mask_random]] = torch.randint(1, len(self.vocab), size=(mask_random.sum(),))
        else:
            # A bit faster when all lengths are 1
            while indices.size(0) > 0:
                uncompleted = is_word_start[indices + 1] == 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx
                    source[indices[mask_random]] = torch.randint(1, len(self.vocab), size=(mask_random.sum(),))

                assert source_length - 1 not in indices

        source = source[to_keep]

        if num_inserts > 0:
            source = self.add_insertion_noise(source, num_inserts / source.size(0))

        return source

    def add_permuted_noise(self, tokens, p):
        num_words = len(tokens)
        num_to_permute = math.ceil(((num_words * 2) * p) / 2.0)
        substitutions = torch.randperm(num_words - 2)[:num_to_permute] + 1
        tokens[substitutions] = tokens[substitutions[torch.randperm(num_to_permute)]]
        return tokens

    def add_rolling_noise(self, tokens):
        offset = np.random.randint(1, max(1, tokens.size(-1) - 1) + 1)
        tokens = torch.cat(
            (tokens[0:1], tokens[offset:-1], tokens[1:offset], tokens[-1:]),
            dim=0,
        )
        return tokens

    def add_insertion_noise(self, tokens, p):
        if p == 0.0:
            return tokens

        num_tokens = len(tokens)
        n = int(math.ceil(num_tokens * p))

        noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
        noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
        noise_mask[noise_indices] = 1
        result = torch.LongTensor(n + len(tokens)).fill_(-1)

        num_random = int(math.ceil(n * self.random_ratio))
        result[noise_indices[num_random:]] = self.mask_idx
        result[noise_indices[:num_random]] = torch.randint(low=1, high=len(self.vocab), size=(num_random,))

        result[~noise_mask] = tokens

        assert (result >= 0).all()
        return result

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch of data
        """
        return collate(
            samples, self.vocab.pad(), self.eos, self.vocab,
            pad_to_length=pad_to_length, input_shapes=self.input_shapes)

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        # sort by target length first, then source length
        if self.tgt_sizes is not None:
            indices = indices[
                np.argsort(self.tgt_sizes[indices], kind='mergesort')
            ]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    def prefetch(self, indices):
        self.src.prefetch(indices)
        self.tgt.prefetch(indices)

    @property
    def supports_prefetch(self):
        return (
            hasattr(self.src, 'supports_prefetch')
            and self.src.supports_prefetch
            and hasattr(self.tgt, 'supports_prefetch')
            and self.tgt.supports_prefetch
        )

    def get_batch_shapes(self):
        return self.input_shapes
