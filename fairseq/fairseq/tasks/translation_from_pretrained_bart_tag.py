# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np

from fairseq.data import LanguagePairDataset
from fairseq import utils
from fairseq.data import (
    data_utils,
    AppendTokenDataset,
    PrependTokenDataset,
    TruncateDataset,
    StripTokenDataset,
    LanguagePairDataset,
    ConcatMultitaskDataset,
    DenoisingDataset,
    SortDataset,
    DropLastTokenDataset,
    ContextDenoisingDataset,
    OffsetTokensDataset,
    encoders,
)


from fairseq.data.encoders.utils import get_whole_word_mask
from .translation import load_langpair_dataset, TranslationTask
from . import register_task
import logging
logger = logging.getLogger(__name__)



@register_task('translation_from_pretrained_bart_tag')
class TranslationFromPretrainedBARTTagTask(TranslationTask):
    """
    Translate from source language to target language with a model initialized with a multilingual pretrain.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        # parser.add_argument('--use_gpu', action='store_true', default=False)
        parser.add_argument('--use-lang-tokens', action='store_true', default=False)
        parser.add_argument('--langs', required=True, metavar='LANG',
                            help='comma-separated list of monolingual language, '
                                 'for example, "en,de,fr". These should match the '
                                 'langs from pretraining (and be in the same order). '
                                 'You should always add all pretraining language idx '
                                 'during finetuning.')
        parser.add_argument('--prepend-bos', action='store_true',
                            help='prepend bos token to each sentence, which matches '
                                 'mBART pretraining')


        # Arguments for multi-task learning
        parser.add_argument('--sort_by_length', default=False, action='store_true')
        parser.add_argument('--tasks', required=True, metavar='TASK',
                            help='comma-separated list of monolingual language, '
                                 'for example, "en,de,fr". These should match the '
                                 'langs from pretraining (and be in the same order). '
                                 'You should always add all pretraining language idx '
                                 'during finetuning.')
        parser.add_argument('--multitask', default=False, action='store_true')
        parser.add_argument('--sample_by_tokens', default=False, action='store_true')

        # Arguments for DAE
        parser.add_argument('--add-entity-mask', default=False, action='store_true')
        parser.add_argument('--mono_data_path', default=None, type=str)
        parser.add_argument('--num_shards', default=10, type=int)
        parser.add_argument('--no-whole-word-mask-langs', type=str, default='', metavar='N',
                            help='languages without spacing between words dont support whole word masking')
        parser.add_argument('--tokens-per-sample', default=512, type=int,
                            help='max number of total tokens over all segments'
                                 ' per sample for dataset')
        parser.add_argument(
            '--sample-break-mode', default="complete_doc", type=str,
            help='mode for breaking sentence',
        )
        parser.add_argument(
            '--mask', default=0.0, type=float,
            help='fraction of words/subwords that will be masked',
        )
        parser.add_argument(
            '--mask-random', default=0.0, type=float,
            help='instead of using [MASK], use random token this often'
        )
        parser.add_argument(
            '--insert', default=0.0, type=float,
            help='insert this percentage of additional random tokens',
        )
        parser.add_argument(
            '--permute', default=0.0, type=float,
            help='take this proportion of subwords and permute them',
        )
        parser.add_argument(
            '--rotate', default=0.5, type=float,
            help='rotate this proportion of inputs',
        )
        parser.add_argument(
            '--poisson-lambda', default=3.0, type=float,
            help='randomly shuffle sentences for this proportion of inputs'
        )
        parser.add_argument(
            '--permute-sentences', default=0.0, type=float,
            help='shuffle this proportion of sentences in all inputs'
        )
        parser.add_argument(
            '--mask-length', default="subword", type=str,
            choices=['subword', 'word', 'span-poisson'],
            help='mask length to choose'
        )
        parser.add_argument(
            '--replace-length', default=-1, type=int,
            help='when masking N tokens, replace with 0, 1, or N tokens (use -1 for N)'
        )

        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.seed = self.args.seed
        if args.use_lang_tokens:
            self.langs = args.langs.split(',')
            self.tasks = args.tasks.split(',')
            print('self.tasks', self.tasks)
            for d in [src_dict, tgt_dict]:
                for l in self.langs:
                    d.add_symbol('[{}]'.format(l))
                for t in self.tasks:
                    d.add_symbol('[{}]'.format(t))
                    print('adding tasks', t)
        for d in [src_dict, tgt_dict]:
            d.add_symbol('<mask>')
        self.mask_idx = src_dict.add_symbol('<mask>')
        print('num of vocab', len(self.src_dict))

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        tasks = self.args.tasks.split(',')
        task2id = {t:i for i, t in enumerate(tasks)}
        id2dataset = {}
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang
        mt_dataset = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=getattr(self.args, 'max_source_positions', 1024),
            max_target_positions=getattr(self.args, 'max_target_positions', 1024),
            load_alignments=self.args.load_alignments,
            prepend_bos=getattr(self.args, 'prepend_bos', False),
            append_source_id=True,
            prepend_task_tag=True,
            task_tag='mt',
            task_id=task2id['mt'],
            source_id=tgt,
            )
        id2dataset[task2id['mt']] = mt_dataset

        if split == 'train':
            eidx = epoch % self.args.num_shards
            prefix = f'{self.args.mono_data_path}/train{eidx}'
            if 'dae' in task2id:
                dae_dataset = self.load_dae_dataset(
                    prefix, split, src, self.src_dict, tgt, self.tgt_dict,
                    combine=combine, dataset_impl=self.args.dataset_impl,
                    upsample_primary=self.args.upsample_primary,
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                    max_source_positions=getattr(self.args, 'max_source_positions', 1024),
                    max_target_positions=getattr(self.args, 'max_target_positions', 1024),
                    load_alignments=self.args.load_alignments,
                    prepend_bos=getattr(self.args, 'prepend_bos', False),
                    append_lang_tag=True,
                    prepend_task_tag=True,
                    task_id=task2id['dae'],
                )
                id2dataset[task2id['dae']] = dae_dataset
                dae_id = task2id['dae']

            elif 'bart' in task2id:
                dae_dataset = self.load_bart_dataset(
                    prefix, split,
                    tgt, self.tgt_dict,
                    dataset_impl=self.args.dataset_impl,
                    upsample_primary=self.args.upsample_primary,
                    left_pad_source=self.args.left_pad_source,
                    max_source_positions=getattr(self.args, 'max_source_positions', 1024),
                    prepend_bos=getattr(self.args, 'prepend_bos', False),
                    append_lang_tag=True,
                    shuffle=True,
                    prepend_task_tag=True,
                    task_id=task2id['bart'],
                )
                id2dataset[task2id['bart']] = dae_dataset
                dae_id = task2id['bart']

            sample_ratios = [1] * len(task2id)
            dataset = [id2dataset[idx] for idx in range(len(task2id))]
            if self.args.sample_by_tokens:
                num_dae = dae_dataset.num_total_tokens() # 10000
                num_mt = mt_dataset.num_total_tokens()   # 1000
                sample_ratios[dae_id] = num_mt / num_dae  # [1, 1/10]  
                # print(f' DAE avg tokens={num_dae / len(dae_dataset)}')
                # print(f' MT avg tokens={num_mt / len(mt_dataset)}')
                # from collections import Counter
                # ttag_cnter = Counter()
                # for i in range(len(dae_dataset)):
                #     src = dae_dataset.src[i]
                #     ttag = dae_dataset.vocab[src.tolist()[0]]
                #     ttag_cnter[ttag] += 1
                # print(f'ttag_cnter', ttag_cnter)

                # ttag_cnter = Counter()
                # for i in range(len(mt_dataset)):
                #     src = mt_dataset.src[i]
                #     ttag = mt_dataset.src_dict[src.tolist()[0]]
                #     ttag_cnter[ttag] += 1
                # print(f'ttag_cnter', ttag_cnter)
                # exit(0)

                # N dae sentences
                # M mt sentences

                # M mt sentences: num_mt
                # sampled sentences for DAE:  avg_mt = num_mt / M , avg_dae = num_dae / N
                # M * avg_mt = N' * avg_dae
                # N' = num_mt / avg_dae = num_mt / num_dae * N

                # mini-bach: 
                # Single-task training:1024 * 32 * 2 tokens
                # multitask-training:  1024 * 64 * 2 tokens, train the same amounts of tokens for MT & DAE

            else:
                sample_ratios[dae_id] = len(mt_dataset) / len(dae_dataset)
            print(f'[JJ] translation_from_pretrained_bart_tag.py sample_ratios', sample_ratios)
            dataset = ConcatMultitaskDataset(dataset, sample_ratios, separate_collater=False)
            print('mt_datset', len(mt_dataset), 'dae_dataset', len(dae_dataset), 'total dataset', len(dataset))
            with data_utils.numpy_seed(self.args.seed + epoch):
                shuffle = np.random.permutation(len(dataset))

            # Shuffle the dataset
            self.datasets[split] = SortDataset(
                dataset,
                sort_order=[
                    shuffle,
                    dataset.sizes,
                ],
            )
            print(f' [JJ] translation_from_pretrained_bart_tag.py: {split} sample_ratios', sample_ratios, len(mt_dataset), len(dae_dataset), len(self.datasets[split]))

        else:
            self.datasets[split] = mt_dataset
            print(f'[JJ] translation_from_pretrained_bart_tag.py: {split}', len(self.datasets[split]))


    def load_dae_dataset(self,
        prefix, split,
        src, src_dict,
        tgt, tgt_dict,
        combine, dataset_impl, upsample_primary,
        left_pad_source, left_pad_target, max_source_positions,
        max_target_positions, prepend_bos=False, load_alignments=False,
        truncate_source=False, append_lang_tag=False,
        num_buckets=0,
        shuffle=True,
        prepend_task_tag=True,
        task_id=0,
    ):


        print(f'reading from {prefix}.src')
        src_dataset = data_utils.load_indexed_dataset(f'{prefix}.src', src_dict, dataset_impl)
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )

        print(f'reading from {prefix}.tgt')
        tgt_dataset = data_utils.load_indexed_dataset(f'{prefix}.tgt', tgt_dict, dataset_impl)

        
        print(f'reading mask from {prefix}.mask')
        mask_index_dataset = data_utils.load_shard_indexed_dataset(f'{prefix}.mask', combine=combine)

        logger.info('[JJ] {} {}-{} {} examples'.format(
            prefix, src, tgt, len(src_dataset)
        ))


        # Source side: [dae] + tokens + </s> + [lang tag]
        # Target side: tokens + </s> + [lang tag]
        # Prepend the task token to the begining
        if prepend_task_tag:
            src_dataset = PrependTokenDataset(src_dataset, src_dict.index('[dae]'))
            mask_index_dataset = OffsetTokensDataset(mask_index_dataset, offset=1)

        # Replace the last token by the language token
        eos = None
        if append_lang_tag:
            # src_dataset = AppendTokenDataset(src_dataset, src_dict.index('[{}]'.format(src)))
            print(f'add language tag {tgt} to source sentences')
            eos = src_dict.index('[{}]'.format(tgt))
            src_dataset = DropLastTokenDataset(src_dataset, eos, max_source_positions)
            if tgt_dataset is not None:
                print(f'add language tag {tgt} to target sentences')
                # tgt_dataset = AppendTokenDataset(tgt_dataset, tgt_dict.index('[{}]'.format(tgt)))
                tgt_dataset = DropLastTokenDataset(tgt_dataset, eos, max_target_positions)
            # eos = tgt_dict.index('[{}]'.format(tgt))
        tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None

        # Mask
        mask_whole_words = get_whole_word_mask(self.args, tgt_dict)
        language_without_segmentations = self.args.no_whole_word_mask_langs.split(',')
        lang_mask_whole_words = mask_whole_words if tgt not in language_without_segmentations else None

        return ContextDenoisingDataset(
                src_dataset,
                src_dataset.sizes,
                tgt_dataset,
                tgt_dataset.sizes,
                mask_index_dataset,
                mask_index_dataset.sizes,
                tgt_dict,
                self.mask_idx,
                lang_mask_whole_words,
                shuffle=shuffle,
                seed=self.seed,
                args=self.args,
                eos=None if not append_lang_tag else tgt_dict.index('[{}]'.format(tgt)),
                input_shapes=getattr(self.args, 'input_shapes', None),
                task_id=task_id,
            )


    def load_bart_dataset(self,
        prefix, split,
        language, dictionary,
        dataset_impl, upsample_primary,
        left_pad_source, max_source_positions,
        prepend_bos=False, 
        truncate_source=False, append_lang_tag=False,
        shuffle=True,
        prepend_task_tag=True,
        task_id=0,
    ):


        print(f'reading from {prefix}')
        dataset = data_utils.load_indexed_dataset(f'{prefix}', dictionary, dataset_impl)
        if truncate_source:
            dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(dataset, dictionary.eos()),
                    max_source_positions - 1,
                ),
                dictionary.eos(),
            )

        logger.info('[JJ] {} {} examples'.format(
            prefix, len(dataset)
        ))

        # Source side: [dae] + tokens + </s> + [lang tag]
        # Target side: tokens + </s> + [lang tag]
        # Prepend the task token to the begining
        if prepend_task_tag:
            dataset = PrependTokenDataset(dataset, dictionary.index('[bart]'))
            
        # Replace the last token by the language token
        eos = None
        if append_lang_tag:
            print(f'add language tag {language} to source sentences')
            eos = dictionary.index('[{}]'.format(language))
            dataset = DropLastTokenDataset(dataset, eos, max_source_positions)
            
        # Mask
        mask_whole_words = get_whole_word_mask(self.args, dictionary)
        language_without_segmentations = self.args.no_whole_word_mask_langs.split(',')
        lang_mask_whole_words = mask_whole_words if language not in language_without_segmentations else None

        return DenoisingDataset(
                dataset,
                dataset.sizes,
                dictionary,
                self.mask_idx,
                lang_mask_whole_words,
                shuffle=shuffle,
                seed=self.seed,
                args=self.args,
                eos=None if not append_lang_tag else dictionary.index('[{}]'.format(language)),
                input_shapes=getattr(self.args, 'input_shapes', None),
            )


    def build_generator(self, models, args):
        if getattr(args, 'score_reference', False):
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(
                self.target_dictionary,
                eos=self.tgt_dict.index('[{}]'.format(self.args.target_lang)) if args.use_lang_tokens else self.tgt_dict.index('</s>') 
            )
        else:
            from fairseq.sequence_generator import SequenceGenerator
            end_token = self.tgt_dict.index('[{}]'.format(self.args.target_lang)) if args.use_lang_tokens else self.tgt_dict.index('</s>')
            return SequenceGenerator(
                models,
                self.target_dictionary,
                beam_size=getattr(args, 'beam', 5),
                max_len_a=getattr(args, 'max_len_a', 0),
                max_len_b=getattr(args, 'max_len_b', 200),
                min_len=getattr(args, 'min_len', 1),
                normalize_scores=(not getattr(args, 'unnormalized', False)),
                len_penalty=getattr(args, 'lenpen', 1),
                unk_penalty=getattr(args, 'unkpen', 0),
                temperature=getattr(args, 'temperature', 1.),
                match_source_len=getattr(args, 'match_source_len', False),
                no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
                eos=end_token
            )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        src_lang_id = self.source_dictionary.index('[{}]'.format(self.args.source_lang))
        source_tokens = []
        for s_t in src_tokens:
            s_t = torch.cat([s_t, s_t.new(1).fill_(src_lang_id)])
            source_tokens.append(s_t)
        dataset = LanguagePairDataset(source_tokens, src_lengths, self.source_dictionary,
                                      tgt_dict=self.target_dictionary,
                                      constraints=constraints)
        return dataset
