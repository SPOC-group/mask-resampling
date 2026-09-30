# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import logging
import os
from typing import Optional

from omegaconf import MISSING, II, OmegaConf

import numpy as np
from fairseq import utils
from fairseq.data import (
    Dictionary,
    FixedMaskDataset,
    FixedMaskMetadataDataset,
    IdDataset,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PrependTokenDataset,
    RightPadDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
)
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

from .language_modeling import SAMPLE_BREAK_MODE_CHOICES, SHORTEN_METHOD_CHOICES


logger = logging.getLogger(__name__)


@dataclass
class MaskedLMConfig(FairseqDataclass):
    data: str = field(
        default=MISSING,
        metadata={
            "help": "colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner"
        },
    )
    sample_break_mode: SAMPLE_BREAK_MODE_CHOICES = field(
        default="none",
        metadata={
            "help": 'If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.'
        },
    )
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    mask_prob: float = field(
        default=0.15,
        metadata={"help": "probability of replacing a token with mask"},
    )
    leave_unmasked_prob: float = field(
        default=0.1,
        metadata={"help": "probability that a masked token is unmasked"},
    )
    random_token_prob: float = field(
        default=0.1,
        metadata={"help": "probability of replacing a token with a random token"},
    )
    dup_factor: int = field(
        default=0,
        metadata={
            "help": "number of fixed masks per completed token sequence; "
            "0 preserves legacy dynamic masking"
        },
    )
    dynamic_masking: bool = field(
        default=False,
        metadata={
            "help": "resample MLM corruption for every training epoch while "
            "keeping completed base sequences fixed; validation stays static"
        },
    )
    mask_seed: Optional[int] = field(
        default=None,
        metadata={
            "help": "seed for finite static masks (defaults to --seed)"
        },
    )
    base_sequence_limit: int = field(
        default=0,
        metadata={
            "help": "for finite masking, use only this deterministic prefix "
            "of completed training sequences before duplication; 0 uses all"
        },
    )
    freq_weighted_replacement: bool = field(
        default=False,
        metadata={"help": "sample random replacement words based on word frequencies"},
    )
    mask_whole_words: bool = field(
        default=False,
        metadata={"help": "mask whole words; you may also want to set --bpe"},
    )
    mask_multiple_length: int = field(
        default=1,
        metadata={"help": "repeat the mask indices multiple times"},
    )
    mask_stdev: float = field(
        default=0.0,
        metadata={"help": "stdev of the mask length"},
    )
    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed --tokens-per-sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)'
        },
    )
    seed: int = II("common.seed")

    include_target_tokens: bool = field(
        default=False,
        metadata={
            "help": "include target tokens in model input. this is used for data2vec"
        },
    )


@register_task("masked_lm", dataclass=MaskedLMConfig)
class MaskedLMTask(FairseqTask):

    cfg: MaskedLMConfig

    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    def __init__(self, cfg: MaskedLMConfig, dictionary):
        super().__init__(cfg)
        self.dictionary = dictionary

        # add mask token
        self.mask_idx = dictionary.add_symbol("<mask>")

    @classmethod
    def setup_task(cls, cfg: MaskedLMConfig, **kwargs):
        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(cfg, dictionary)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path_index = (epoch - 1) % len(paths)
        data_path = paths[data_path_index]
        split_path = os.path.join(data_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path,
            self.source_dictionary,
            combine=combine,
        )
        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )

        dataset = maybe_shorten_dataset(
            dataset,
            split,
            self.cfg.shorten_data_split_list,
            self.cfg.shorten_method,
            self.cfg.tokens_per_sample,
            self.cfg.seed,
        )

        # create continuous blocks of tokens
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.cfg.tokens_per_sample - 1,  # one less for <s>
            pad=self.source_dictionary.pad(),
            eos=self.source_dictionary.eos(),
            break_mode=self.cfg.sample_break_mode,
        )
        logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())

        fixed_mask_dataset = None
        if self.cfg.dup_factor < 0:
            raise ValueError("--dup-factor must be 0 (legacy) or a positive integer")
        if self.cfg.dynamic_masking and self.cfg.dup_factor != 0:
            raise ValueError("--dynamic-masking requires --dup-factor 0")
        if self.cfg.base_sequence_limit < 0:
            raise ValueError("--base-sequence-limit must be non-negative")
        controlled_masking = self.cfg.dup_factor > 0 or self.cfg.dynamic_masking
        dynamic_training = self.cfg.dynamic_masking and split.startswith("train")
        if controlled_masking:
            # Validation remains a single fixed view across every K so changing
            # either K or the training corruption mode cannot change the
            # evaluation distribution or validation cost.
            effective_dup_factor = (
                self.cfg.dup_factor
                if split.startswith("train") and not dynamic_training
                else 1
            )
            base_sequence_limit = (
                self.cfg.base_sequence_limit
                if split.startswith("train") and self.cfg.base_sequence_limit > 0
                else len(dataset)
            )
            if base_sequence_limit > len(dataset):
                raise ValueError(
                    "--base-sequence-limit={} exceeds the {} completed {} "
                    "sequences".format(
                        base_sequence_limit,
                        len(dataset),
                        split,
                    )
                )
            namespace = "{}:{}:{}".format(
                split,
                data_path_index,
                os.path.basename(os.path.normpath(data_path)),
            )
            fixed_mask_dataset = FixedMaskDataset(
                dataset,
                dup_factor=effective_dup_factor,
                namespace=namespace,
                base_sequence_limit=base_sequence_limit,
            )
            fixed_mask_dataset.verify_same_input_ids()
            if dynamic_training:
                logger.info(
                    "dynamic masking: {} fixed sequences; a new reproducible "
                    "mask is sampled for every epoch; input_ids sanity check "
                    "passed".format(fixed_mask_dataset.base_sequence_count)
                )
            else:
                logger.info(
                    "finite static masking: {} fixed sequences x {} masks = {} "
                    "examples; duplicate input_ids sanity check passed".format(
                        fixed_mask_dataset.base_sequence_count,
                        effective_dup_factor,
                        len(fixed_mask_dataset),
                    )
                )
            dataset = fixed_mask_dataset

        # create masked input and targets
        mask_whole_words = (
            get_whole_word_mask(self.args, self.source_dictionary)
            if self.cfg.mask_whole_words
            else None
        )

        src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
            dataset,
            self.source_dictionary,
            pad_idx=self.source_dictionary.pad(),
            mask_idx=self.mask_idx,
            seed=(
                self.cfg.mask_seed
                if self.cfg.mask_seed is not None
                else self.cfg.seed
            ),
            mask_prob=self.cfg.mask_prob,
            leave_unmasked_prob=self.cfg.leave_unmasked_prob,
            random_token_prob=self.cfg.random_token_prob,
            freq_weighted_replacement=self.cfg.freq_weighted_replacement,
            mask_whole_words=mask_whole_words,
            mask_multiple_length=self.cfg.mask_multiple_length,
            mask_stdev=self.cfg.mask_stdev,
            static_mask=fixed_mask_dataset is not None and not dynamic_training,
        )

        with data_utils.numpy_seed(self.cfg.seed):
            shuffle = np.random.permutation(len(src_dataset))

        target_dataset = RightPadDataset(
            tgt_dataset,
            pad_idx=self.source_dictionary.pad(),
        )

        input_dict = {
            "src_tokens": RightPadDataset(
                src_dataset,
                pad_idx=self.source_dictionary.pad(),
            ),
            "src_lengths": NumelDataset(src_dataset, reduce=False),
        }
        if self.cfg.include_target_tokens:
            input_dict["target_tokens"] = target_dataset

        dataset_definition = {
            "id": IdDataset(),
            "net_input": input_dict,
            "target": target_dataset,
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_dataset, reduce=True),
        }
        if fixed_mask_dataset is not None:
            dataset_definition.update(
                {
                    "example_id": FixedMaskMetadataDataset(
                        fixed_mask_dataset, "example_id"
                    ),
                    "mask_index": FixedMaskMetadataDataset(
                        fixed_mask_dataset, "mask_index"
                    ),
                }
            )

        self.datasets[split] = SortDataset(
            NestedDictionaryDataset(
                dataset_definition,
                sizes=[src_dataset.sizes],
            ),
            sort_order=[
                shuffle,
                src_dataset.sizes,
            ],
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, sort=True):
        src_dataset = RightPadDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                self.cfg.tokens_per_sample - 1,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            pad_idx=self.source_dictionary.pad(),
        )
        src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary.bos())
        src_dataset = NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": src_dataset,
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
            },
            sizes=src_lengths,
        )
        if sort:
            src_dataset = SortDataset(src_dataset, sort_order=[src_lengths])
        return src_dataset

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary
