# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np

from fairseq.data import (
    data_utils,
    Dictionary,
    AppendTokenDataset,
    ConcatDataset,
    DenoisingDataset,
    PrependTokenDataset,
    ResamplingDataset,
    SortDataset,
    TokenBlockDataset,
    DropLastTokenDataset,
    OffsetTokensDataset,
    ContextDenoisingDataset,
)
from .denoising import DenoisingTask
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.tasks import register_task
from fairseq.tasks.translation import load_langpair_dataset
import time

logger = logging.getLogger(__name__)




@register_task('multilingual_context_shard_denoising_translation')
class MultilingualContextShardDenoisingTranslationTask(DenoisingTask):

    @staticmethod
    def add_args(parser):
        DenoisingTask.add_args(parser)
        parser.add_argument('--multilang-sampling-alpha', type=float, default=1.0,
                            help='smoothing alpha for sample ratios across multiple datasets')
        parser.add_argument('--add-lang-token', default=False, action='store_true')
        parser.add_argument('--add-entity-mask', default=False, action='store_true')
        parser.add_argument('--langs', type=str, help="language ids we are considering", default=None)
        parser.add_argument('--no-whole-word-mask-langs', type=str, default='', metavar='N',
                            help='languages without spacing between words dont support whole word masking')
        parser.add_argument('--num-shards', type=int, default=1, required=True,
                            help="specify how many shards for data in each language")

        # parameters for translation
        parser.add_argument('--parallel_data_path', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner; \
                            however, valid and test data are always in the first directory to \
                            avoid the need for repeating them in all directories')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--load-alignments', action='store_true',
                            help='load the binarized alignments')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--truncate-source', action='store_true', default=False,
                            help='truncate source to max-source-positions')
        parser.add_argument('--num-batch-buckets', default=0, type=int, metavar='N',
                            help='if >0, then bucket source and target lengths into N '
                                 'buckets and pad accordingly; this is useful on TPUs '
                                 'to minimize the number of compilations')

        # parameters for multi-tasking
        parser.add_argument('--add_task_token_to_src', default=False, action='store_true')
        parser.add_argument('--task_tokens', default='dae,mt', type=str)
        parser.add_argument('--add_task_token_to_tgt', default=False, action='store_true')
        parser.add_argument('--dae_sample_ratio', default=1, type=float)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task.
        """
        paths = args.data.split(':')
        assert len(paths) > 0
        dictionary = Dictionary.load(os.path.join(paths[0], 'dict.txt'))

        data_path = paths[0]
        if args.langs is None:
            languages = sorted([
                name for name in os.listdir(data_path)
                if os.path.isdir(os.path.join(data_path, name))
            ])
        else:
            languages = args.langs.split(',')

        if args.add_lang_token:
            for lang in languages:
                dictionary.add_symbol('[{}]'.format(lang))

        if args.add_task_token_to_src and args.task_tokens is not None:
            for tok in args.task_tokens.split(','):
                dictionary.add_symbol('[{}]'.format(tok))


        logger.info("dictionary: {} types".format(len(dictionary)))
        if not hasattr(args, 'shuffle_instance'):
            args.shuffle_instance = False
        return cls(args, dictionary)

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)
        self.dictionary = dictionary
        self.seed = args.seed

        # add mask token
        self.mask_idx = self.dictionary.add_symbol('<mask>')
        self.langs = args.langs
        self.args = args
        print('[JJ] multilingual_context_shard_denoising_translation.py: __init__')

    def _get_sample_prob(self, dataset_lens):
        """
        Get smoothed sampling porbability by languages. This helps low resource
        languages by upsampling them.
        """
        prob = dataset_lens / dataset_lens.sum()
        smoothed_prob = prob ** self.args.multilang_sampling_alpha
        smoothed_prob = smoothed_prob / smoothed_prob.sum()
        return smoothed_prob

    def _get_path(self):
        paths = path

    def has_sharded_data(self, split):
        return True

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):

        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        # load denoising dataset for DAE training
        denoising_dataset = self.load_denoising_dataset(split, epoch, combine, **kwargs)

        # load langpair dataset for MT training
        src, tgt = self.args.source_lang, self.args.target_lang
        langpair_dataset = load_langpair_dataset(
            self.args.parallel_data_path, split, 
            src, self.dictionary, tgt, self.dictionary,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
            truncate_source=self.args.truncate_source,
            num_buckets=self.args.num_batch_buckets,
            shuffle=False,
        )

        

        # Combine two datasets
        # Upsample or downsample denoising dataset to match the size of the parallel dataset
        sample_ratios = [1, int(len(langpair_dataset) / len(denoising_dataset) * self.args.dae_sample_ratio)]
        dataset = ConcatDataset([langpair_dataset, denoising_dataset], sample_ratios)
        
        if split == 'test':
            shuffle = np.array(range(len(dataset)))
        else:
            with data_utils.numpy_seed(self.args.seed + epoch):
                shuffle = np.random.permutation(len(dataset))

        self.datasets[split] = SortDataset(
            dataset,
            sort_order=[
                shuffle,
                dataset.sizes,
            ],
        )

    def load_denoising_dataset(self, split, epoch=1, combine=False, **kwargs):

        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        num_shards = self.args.num_shards


        # paths = self.args.data.split(':')
        # assert len(paths) > 0
        # data_path = paths[(epoch - 1) % len(paths)]
        # split_path = os.path.join(data_path, split)
        data_path = self.args.data

        if self.langs is None:
            languages = sorted([
                name for name in os.listdir(data_path)
                if os.path.isdir(os.path.join(data_path, name))
            ])
        else:
            languages = self.langs.split(',')
            for name in languages:
                p = os.path.join(data_path, name)
                assert os.path.exists(p), "data not found: {}".format(p)

        logger.info("Training on {0} languages: {1}".format(len(languages), languages))
        logger.info("Language to id mapping: ", {
                lang: id for id, lang in enumerate(languages)
            }
        )

        mask_whole_words = get_whole_word_mask(self.args, self.dictionary)
        language_without_segmentations = self.args.no_whole_word_mask_langs.split(',')
        lang_datasets = []
        for language in languages:
            if split == self.args.train_subset:
                shard_idx = (epoch - 1) % self.args.num_shards
                split_path = os.path.join(data_path, language, '{}{}'.format(split, shard_idx))
            else:
                split_path = os.path.join(data_path, language, split)
            logger.info(f"Load data from {split_path}")

            src_dataset = data_utils.load_shard_indexed_dataset(
                f'{split_path}.src',
                self.source_dictionary,
                self.args.dataset_impl,
                combine=combine,
            )

            tgt_dataset = data_utils.load_shard_indexed_dataset(
                f'{split_path}.tgt',
                self.source_dictionary,
                self.args.dataset_impl,
                combine=combine,
            )

            # ctxt_dataset = data_utils.load_shard_indexed_dataset(
            #     f'{split_path}.ctx',
            #     self.source_dictionary,
            #     self.args.dataset_impl,
            #     combine=combine,
            # )

            mask_index_dataset = data_utils.load_shard_indexed_dataset(f'{split_path}.mask', combine=combine)


            # print('[JJ] multilingual_shard_denoising.py: after load_shard_indexed_dataset', language)
            # print('[JJ] ', [dataset[xx].shape for xx in range(10)])
            if src_dataset is None or tgt_dataset is None or mask_index_dataset is None:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))

            # Source side: <s> + tokens + </s> + <lang_tag>
            # - prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
            src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary.bos())
            offset = 1
            # - prepend a task-token ([dae] for denoising auto-encoding objective)
            if self.args.add_task_token_to_src:
                src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary.index('[dae]'))
                offset = 2
            mask_index_dataset = OffsetTokensDataset(mask_index_dataset, offset=offset)

            if self.args.add_lang_token:
                end_token = self.source_dictionary.index('[{}]'.format(language))
                src_dataset = DropLastTokenDataset(src_dataset, end_token, self.args.max_source_positions)
            else:
                end_token = self.source_dictionary.index('</s>')
            
            # Target side: 
            # - prepend a task-token ([dae] for denoising auto-encoding objective)
            if self.args.add_task_token_to_tgt:
                tgt_dataset = PrependTokenDataset(tgt_dataset, self.source_dictionary.index('[dae]'))
            # - replace the last token by "end_token"
            tgt_dataset = DropLastTokenDataset(tgt_dataset, end_token, self.args.max_target_positions)

            # print('[JJ] multilingual_shard_denoising.py: after PrependTokenDataset', language)
            # print('[JJ] ', [dataset[xx].shape for xx in range(10)])

            lang_mask_whole_words = mask_whole_words if language not in language_without_segmentations else None
            # print('[JJ] lang_mask_whole_words', lang_mask_whole_words)
            noising_time = time.time()
            lang_dataset = ContextDenoisingDataset(
                src_dataset,
                src_dataset.sizes,
                tgt_dataset,
                tgt_dataset.sizes,
                mask_index_dataset,
                mask_index_dataset.sizes,
                self.dictionary,
                self.mask_idx,
                lang_mask_whole_words,
                shuffle=self.args.shuffle_instance,
                seed=self.seed,
                args=self.args,
                eos=None if not self.args.add_lang_token else self.source_dictionary.index('[{}]'.format(language)),
                input_shapes=getattr(self.args, 'input_shapes', None)
            )
                # ctxt_dataset=ctxt_dataset,
                # ctxt_sizes=ctxt_dataset.sizes,
            lang_datasets.append(lang_dataset)
            logger.info(f'  =[JJ] time to create noisy training data {time.time() - noising_time} s for language={language}')
            # print('[JJ] multilingual_shard_denoising.py: after DenoisingDataset', language)
            # print('[JJ] ', [lang_dataset[xx]['source'].shape for xx in range(10)], lang_dataset[0].keys() )

        dataset_lengths = np.array(
            [len(d) for d in lang_datasets],
            dtype=float,
        )
        logger.info(
            'loaded total {} blocks for all languages'.format(
                int(dataset_lengths.sum()),
            )
        )
        if split == self.args.train_subset:
            # For train subset, additionally up or down sample languages.
            sample_probs = self._get_sample_prob(dataset_lengths)
            logger.info(
                "Sample probability by language: {}".format({
                    lang: "{0:.4f}".format(sample_probs[id])
                    for id, lang in enumerate(languages)
                })
            )
            size_ratio = (sample_probs * dataset_lengths.sum()) / dataset_lengths
            # logger.info(f' =[JJ] dataset_lengths = {dataset_lengths}, sum(dataset_lengths)={dataset_lengths.sum()}')
            logger.info(
                "Up/Down Sampling ratio by language: {}".format({
                    lang: "{}".format(size_ratio[id])
                    for id, lang in enumerate(languages)
                })
            )

            resampled_time = time.time()
            for dd in lang_datasets:
                logger.info(f' =[JJ] before resampled_lang_datasets: length = {len(dd)}')
            resampled_lang_datasets = [
                ResamplingDataset(
                    lang_datasets[i],
                    size_ratio=size_ratio[i],
                    seed=self.args.seed,
                    epoch=epoch,
                    replace=size_ratio[i] >= 1.0,
                )
                for i, d in enumerate(lang_datasets)
            ]
            for dd in resampled_lang_datasets:
                logger.info(f' =[JJ] after resampled_lang_datasets: length = {len(dd)}')
            logger.info(f' =[JJ] time to create resampled_lang_datasets {time.time() - resampled_time} s')
            # print('[JJ] multilingual_shard_denoising.py: after ResamplingDataset')
            # print('[JJ] ', resampled_lang_datasets[0][0]['source'].shape)
            # print('[JJ] ', resampled_lang_datasets[1][0]['source'].shape)
            dataset = ConcatDataset(
                resampled_lang_datasets,
            )
            # print('[JJ] multilingual_shard_denoising.py: after ResamplingDataset')
            # print('[JJ] ', [dataset[xx]['source'].shape for xx in range(10)])
        else:
            dataset = ConcatDataset(lang_datasets)
            lang_splits = [split]
            for lang_id, lang_dataset in enumerate(lang_datasets):
                split_name = split + '_' + languages[lang_id]
                lang_splits.append(split_name)
                self.datasets[split_name] = lang_dataset

            if split in self.args.valid_subset:
                self.args.valid_subset = self.args.valid_subset.replace(
                    split, ','.join(lang_splits)
                )

        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(dataset))
        
        return dataset