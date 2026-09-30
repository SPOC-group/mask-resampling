# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch

from fairseq.data import (
    Dictionary,
    FixedMaskDataset,
    FixedMaskMetadataDataset,
    ListDataset,
    MaskTokensDataset,
)


class TestFixedMaskDataset(unittest.TestCase):
    def setUp(self):
        self.dictionary = Dictionary()
        token_ids = [
            self.dictionary.add_symbol("token_{}".format(index))
            for index in range(96)
        ]
        self.sequences = [
            torch.tensor(token_ids[:64]),
            torch.tensor(token_ids[16:80]),
            torch.tensor(token_ids[32:96]),
        ]
        self.base_dataset = ListDataset(
            self.sequences,
            sizes=np.asarray([len(item) for item in self.sequences]),
        )

    def _apply_mask(self, dataset, seed=17):
        mask_idx = self.dictionary.add_symbol("<mask>")
        return MaskTokensDataset.apply_mask(
            dataset,
            self.dictionary,
            pad_idx=self.dictionary.pad(),
            mask_idx=mask_idx,
            seed=seed,
            mask_prob=0.5,
            leave_unmasked_prob=0.1,
            random_token_prob=0.1,
            static_mask=True,
        )

    def test_duplication_is_after_fixed_sequence_construction(self):
        dataset = FixedMaskDataset(self.base_dataset, 5, "train:0:test-data")
        other_k_dataset = FixedMaskDataset(
            self.base_dataset, 2, "train:0:test-data"
        )
        self.assertEqual(len(dataset), len(self.base_dataset) * 5)
        dataset.verify_same_input_ids(range(len(self.base_dataset)))

        for example_index, expected in enumerate(self.sequences):
            copies = [dataset[example_index * 5 + k] for k in range(5)]
            self.assertTrue(all(torch.equal(copy, expected) for copy in copies))
            example_ids = {
                dataset.example_id(example_index * 5 + k) for k in range(5)
            }
            self.assertEqual(len(example_ids), 1)
            self.assertEqual(
                [dataset.mask_index(example_index * 5 + k) for k in range(5)],
                list(range(5)),
            )
            self.assertEqual(
                dataset.example_id(example_index * 5),
                other_k_dataset.example_id(example_index * 2),
            )

    def test_static_masks_are_reproducible_and_epoch_independent(self):
        first_dataset = FixedMaskDataset(
            self.base_dataset, 5, "train:0:test-data"
        )
        second_dataset = FixedMaskDataset(
            self.base_dataset, 5, "train:0:test-data"
        )
        first_source, first_target = self._apply_mask(first_dataset)
        second_source, second_target = self._apply_mask(second_dataset)
        second_source.set_epoch(99)
        second_target.set_epoch(99)

        for index in range(len(first_dataset)):
            self.assertTrue(torch.equal(first_source[index], second_source[index]))
            self.assertTrue(torch.equal(first_target[index], second_target[index]))

        # Independent k substreams should yield mask diversity for one sequence.
        masks = {
            tuple(first_target[k].ne(self.dictionary.pad()).tolist())
            for k in range(5)
        }
        self.assertGreater(len(masks), 1)

        # Increasing K only adds substreams; existing (example_id, k) masks stay
        # byte-identical.
        k2_dataset = FixedMaskDataset(self.base_dataset, 2, "train:0:test-data")
        k2_source, k2_target = self._apply_mask(k2_dataset)
        for example_index in range(len(self.base_dataset)):
            for mask_index in range(2):
                k5_index = example_index * 5 + mask_index
                k2_index = example_index * 2 + mask_index
                self.assertTrue(
                    torch.equal(first_source[k5_index], k2_source[k2_index])
                )
                self.assertTrue(
                    torch.equal(first_target[k5_index], k2_target[k2_index])
                )

    def test_base_sequence_limit_is_applied_before_duplication(self):
        dataset = FixedMaskDataset(
            self.base_dataset,
            10,
            "train:0:test-data",
            base_sequence_limit=2,
        )
        self.assertEqual(dataset.base_sequence_count, 2)
        self.assertEqual(len(dataset), 20)
        dataset.verify_same_input_ids([0, 1])
        for index in range(20):
            self.assertLess(dataset.source_index(index), 2)
        with self.assertRaises(IndexError):
            dataset.verify_same_input_ids([2])

        with self.assertRaises(ValueError):
            FixedMaskDataset(
                self.base_dataset,
                10,
                "train:0:test-data",
                base_sequence_limit=4,
            )

    def test_mask_seed_and_namespace_change_corruption(self):
        first = FixedMaskDataset(self.base_dataset, 2, "train:0:test-data")
        other_namespace = FixedMaskDataset(
            self.base_dataset, 2, "train:1:test-data"
        )
        first_source, _ = self._apply_mask(first, seed=17)
        other_seed_source, _ = self._apply_mask(first, seed=18)
        other_namespace_source, _ = self._apply_mask(other_namespace, seed=17)

        self.assertFalse(torch.equal(first_source[0], other_seed_source[0]))
        self.assertFalse(torch.equal(first_source[0], other_namespace_source[0]))

    def test_metadata_is_lazy_and_collates_to_int64(self):
        dataset = FixedMaskDataset(self.base_dataset, 2, "train:0:test-data")
        example_ids = FixedMaskMetadataDataset(dataset, "example_id")
        mask_indices = FixedMaskMetadataDataset(dataset, "mask_index")

        self.assertEqual(example_ids[0], example_ids[1])
        self.assertEqual([mask_indices[0], mask_indices[1]], [0, 1])
        self.assertEqual(mask_indices.collater([0, 1]).dtype, torch.int64)

    def test_legacy_dynamic_masking_still_changes_by_epoch(self):
        mask_idx = self.dictionary.add_symbol("<mask>")
        source = MaskTokensDataset(
            self.base_dataset,
            self.dictionary,
            pad_idx=self.dictionary.pad(),
            mask_idx=mask_idx,
            seed=17,
            mask_prob=0.5,
            leave_unmasked_prob=0.1,
            random_token_prob=0.1,
        )
        epoch_zero = source[0].clone()
        source.set_epoch(1)
        self.assertFalse(torch.equal(epoch_zero, source[0]))

    def test_dynamic_apply_mask_resamples_and_clears_epoch_cache(self):
        dataset = FixedMaskDataset(self.base_dataset, 1, "train:0:test-data")
        mask_idx = self.dictionary.add_symbol("<mask>")
        source, target = MaskTokensDataset.apply_mask(
            dataset,
            self.dictionary,
            pad_idx=self.dictionary.pad(),
            mask_idx=mask_idx,
            seed=17,
            mask_prob=0.5,
            leave_unmasked_prob=0.0,
            random_token_prob=0.0,
            static_mask=False,
        )

        source.set_epoch(1)
        target.set_epoch(1)
        epoch_one_source = source[0].clone()
        epoch_one_target = target[0].clone()
        self.assertTrue(
            torch.equal(
                epoch_one_source.eq(mask_idx),
                epoch_one_target.ne(self.dictionary.pad()),
            )
        )

        # These reads populate the outer LRU caches. set_epoch must invalidate
        # them so the same base example receives new corruption.
        source.set_epoch(2)
        target.set_epoch(2)
        self.assertFalse(torch.equal(epoch_one_source, source[0]))
        self.assertFalse(torch.equal(epoch_one_target, target[0]))

        # Epoch-keyed resampling is deterministic and resume-safe.
        source.set_epoch(1)
        target.set_epoch(1)
        self.assertTrue(torch.equal(epoch_one_source, source[0]))
        self.assertTrue(torch.equal(epoch_one_target, target[0]))


if __name__ == "__main__":
    unittest.main()
