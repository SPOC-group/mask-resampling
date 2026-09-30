"""Create the original masked-CNN splits and training-only normalization."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from cifar10_conv_mae_core import balanced_splits, channel_statistics, load_cifar10_train, save_json, validate_splits

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(5)))
    a = p.parse_args()
    images, labels = load_cifar10_train(a.data_root)
    for seed in a.seeds:
        split_seed = 2026 + 1009 * seed
        splits = balanced_splits(labels, split_seed, 4000, 500, 500)
        validate_splits(splits, labels, 4000, 500, 500)
        root = a.output_root / f'seed{seed}' / 'protocol'
        root.mkdir(parents=True, exist_ok=True)
        saved = root / 'split_indices.npz'
        if saved.exists():
            with np.load(saved) as old:
                if not all(np.array_equal(old[k], v) for k, v in splits.items()):
                    raise ValueError(f'Existing split differs: {root}')
        np.savez(saved, **splits)
        mean, std = channel_statistics(images, splits['pretrain'])
        save_json(root / 'normalization.json', {'mean': mean.tolist(), 'std': std.tolist(), 'source': 'pretraining images only'})
        save_json(root / 'protocol.json', {'seed': seed, 'split_seed': split_seed,
                  'pretraining_images': 40000, 'reconstruction_validation_images': 5000,
                  'probe_training_images': 40000, 'probe_validation_images': 10000,
                  'official_test_batch_opened': False})
    print(f'Prepared {len(a.seeds)} protocols')

if __name__ == '__main__':
    main()
