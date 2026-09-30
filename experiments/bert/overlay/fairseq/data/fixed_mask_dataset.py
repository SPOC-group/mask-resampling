# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hashlib

import numpy as np
import torch

from . import BaseWrapperDataset, FairseqDataset


class FixedMaskDataset(BaseWrapperDataset):
    """Repeat fixed token sequences for finite, static MLM corruption.

    Item ``example_index * dup_factor + mask_index`` always reads the same
    underlying item ``example_index``.  Duplication therefore happens after
    all tokenization and sequence construction performed by the wrapped
    dataset.
    """

    def __init__(self, dataset, dup_factor, namespace, base_sequence_limit=None):
        super().__init__(dataset)
        if dup_factor < 1:
            raise ValueError("dup_factor must be at least 1")
        if len(dataset) >= 2**32:
            raise ValueError("FixedMaskDataset supports fewer than 2^32 examples")
        if base_sequence_limit is None:
            base_sequence_limit = len(dataset)
        if not 0 <= base_sequence_limit <= len(dataset) or (
            len(dataset) > 0 and base_sequence_limit == 0
        ):
            raise ValueError(
                "base_sequence_limit must select at least one sequence and "
                "not exceed len(dataset)"
            )

        self.dup_factor = dup_factor
        self.base_sequence_count = base_sequence_limit
        self.namespace = namespace
        namespace_digest = hashlib.blake2b(
            namespace.encode("utf-8"), digest_size=4, person=b"DinkyTrain"
        ).digest()
        # Reserve the sign bit so example IDs fit in a torch.int64 tensor.
        self.namespace_id = int.from_bytes(namespace_digest, "little") & 0x7FFFFFFF
        self._sizes = np.repeat(
            np.asarray(dataset.sizes)[:base_sequence_limit], dup_factor
        )

    def __len__(self):
        return self.base_sequence_count * self.dup_factor

    @property
    def sizes(self):
        return self._sizes

    def source_index(self, index):
        return index // self.dup_factor

    def mask_index(self, index):
        return index % self.dup_factor

    def example_id(self, index):
        return (self.namespace_id << 32) | self.source_index(index)

    def mask_seed_components(self, index):
        """Return the stable ``(example_id, k)`` mask RNG key."""
        return self.example_id(index), self.mask_index(index)

    def __getitem__(self, index):
        return self.dataset[self.source_index(index)]

    def num_tokens(self, index):
        return self.dataset.num_tokens(self.source_index(index))

    def size(self, index):
        return self.dataset.size(self.source_index(index))

    def attr(self, attr, index):
        if attr == "example_id":
            return self.example_id(index)
        if attr == "mask_index":
            return self.mask_index(index)
        if attr == "mask_seed_components":
            return self.mask_seed_components(index)
        return self.dataset.attr(attr, self.source_index(index))

    def prefetch(self, indices):
        self.dataset.prefetch({self.source_index(index) for index in indices})

    def set_epoch(self, epoch):
        # Finite masking requires a fixed X.  In particular, do not forward the
        # epoch to RandomCropDataset or any other sequence-building wrapper.
        self.epoch = epoch

    def verify_same_input_ids(self, example_indices=None):
        """Assert that every mask copy for selected examples has identical IDs."""
        if self.base_sequence_count == 0:
            return
        if example_indices is None:
            example_indices = sorted(
                {
                    0,
                    self.base_sequence_count // 2,
                    self.base_sequence_count - 1,
                }
            )

        for example_index in example_indices:
            if not 0 <= example_index < self.base_sequence_count:
                raise IndexError("example index is out of range")
            first = self[example_index * self.dup_factor]
            for mask_index in range(1, self.dup_factor):
                duplicate = self[example_index * self.dup_factor + mask_index]
                if not torch.equal(first, duplicate):
                    raise AssertionError(
                        "fixed-mask copies for example_id={} have different input_ids"
                        .format(self.example_id(example_index * self.dup_factor))
                    )


class FixedMaskMetadataDataset(FairseqDataset):
    """Expose stable example and mask-copy IDs without materializing arrays."""

    def __init__(self, dataset, field):
        super().__init__()
        if field not in {"example_id", "mask_index"}:
            raise ValueError("field must be example_id or mask_index")
        self.dataset = dataset
        self.field = field

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return getattr(self.dataset, self.field)(index)

    def collater(self, samples):
        return torch.tensor(samples, dtype=torch.long)
