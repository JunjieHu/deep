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
)
from .denoising import DenoisingTask
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.tasks import register_task
import time

logger = logging.getLogger(__name__)


@register_task('multilingual_shard_denoising')
class MultilingualShardDenoisingTask(DenoisingTask):

    @staticmethod
    def add_args(parser):
        DenoisingTask.add_args(parser)
        parser.add_argument('--multilang-sampling-alpha', type=float, default=1.0,
                            help='smoothing alpha for sample ratios across multiple datasets')
        parser.add_argument('--add-lang-token', default=False, action='store_true')
        parser.add_argument('--langs', type=str, help="language ids we are considering", default=None)
        parser.add_argument('--no-whole-word-mask-langs', type=str, default='', metavar='N',
                            help='languages without spacing between words dont support whole word masking')
        parser.add_argument('--num-shards', type=int, default=1, required=True,
                            help="specify how many shards for data in each language")
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

        # add lang tokens
        paths = args.data.split(':')
        assert len(paths) > 0
        # dictionary = Dictionary.load(os.path.join(paths[0], 'dict.txt'))

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

            dataset = data_utils.load_shard_indexed_dataset(
                split_path,
                self.source_dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            # print('[JJ] multilingual_shard_denoising.py: after load_shard_indexed_dataset', language)
            # print('[JJ] ', [dataset[xx].shape for xx in range(10)])
            if dataset is None:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))

            # end_token = self.source_dictionary.index('[{}]'.format(language)) \
            #     if self.args.add_lang_token else self.source_dictionary.eos()

            # # create continuous blocks of tokens
            # dataset = TokenBlockDataset(
            #     dataset,
            #     dataset.sizes,
            #     self.args.tokens_per_sample - 2,  # one less for <s>
            #     pad=self.source_dictionary.pad(),
            #     eos=end_token,
            #     break_mode=self.args.sample_break_mode,
            # )
            # logger.info('loaded {} blocks from: {}'.format(len(dataset), split_path))

            # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
            dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())
            if self.args.add_lang_token:
                end_token = self.source_dictionary.index('[{}]'.format(language))
                dataset = DropLastTokenDataset(dataset, end_token)
                print('[JJ] add the lang tokens to the last tokens')
            # print('[JJ] multilingual_shard_denoising.py: after PrependTokenDataset', language)
            # print('[JJ] ', [dataset[xx].shape for xx in range(10)])

            lang_mask_whole_words = mask_whole_words if language not in language_without_segmentations else None
            # print('[JJ] lang_mask_whole_words', lang_mask_whole_words)
            noising_time = time.time()
            lang_dataset = DenoisingDataset(
                dataset,
                dataset.sizes,
                self.dictionary,
                self.mask_idx,
                lang_mask_whole_words,
                shuffle=self.args.shuffle_instance,
                seed=self.seed,
                args=self.args,
                eos=None if not self.args.add_lang_token else self.source_dictionary.index('[{}]'.format(language)),
                input_shapes=getattr(self.args, 'input_shapes', None),
            )
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

        self.datasets[split] = SortDataset(
            dataset,
            sort_order=[
                shuffle,
                dataset.sizes,
            ],
        )
