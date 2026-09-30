#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare an ImageFolder-style ImageNet-100 subset from ImageNet-1K.')
    parser.add_argument('--imagenet1k_root', required=True,
                        help='Root containing train/ and val/ ImageNet-1K folders.')
    parser.add_argument('--output_root', required=True,
                        help='Output root for the ImageNet-100 subset.')
    parser.add_argument('--class_list', required=True,
                        help='Text file with one WordNet synset id per line.')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'],
                        help='Splits to prepare, e.g. train val.')
    parser.add_argument('--copy', action='store_true',
                        help='Copy folders instead of creating symlinks.')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print actions without changing the filesystem.')
    return parser.parse_args()


def read_class_list(path):
    classes = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            classes.append(line)
    if len(classes) != len(set(classes)):
        raise ValueError('class list contains duplicate synsets')
    if len(classes) != 100:
        raise ValueError('expected exactly 100 classes, found {}'.format(len(classes)))
    return classes


def count_images(root):
    return sum(
        1 for path in root.rglob('*')
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def ensure_link_or_copy(src, dst, copy, dry_run):
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and Path(dst.readlink()) == src:
            return 'exists'
        if copy and dst.is_dir():
            return 'exists'
        raise FileExistsError('{} already exists and does not match {}'.format(dst, src))

    if dry_run:
        return 'would_copy' if copy else 'would_symlink'

    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copytree(src, dst)
        return 'copied'
    dst.symlink_to(src, target_is_directory=True)
    return 'symlinked'


def main():
    args = parse_args()
    imagenet_root = Path(args.imagenet1k_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser()
    classes = read_class_list(args.class_list)

    print('Selected classes: {}'.format(len(classes)))
    print('Mode: {}'.format('copy' if args.copy else 'symlink'))
    if args.dry_run:
        print('Dry run: no filesystem changes will be made')

    total_images = 0
    for split in args.splits:
        split_images = 0
        missing = []
        for synset in classes:
            src = imagenet_root / split / synset
            if not src.is_dir():
                missing.append(str(src))
                continue
            split_images += count_images(src)

        if missing:
            raise FileNotFoundError(
                'Missing {} class folders for split {}:\n{}'.format(
                    len(missing), split, '\n'.join(missing[:20])))

        print('{}: {} images'.format(split, split_images))
        total_images += split_images

        for synset in classes:
            src = (imagenet_root / split / synset).resolve()
            dst = output_root / split / synset
            action = ensure_link_or_copy(src, dst, args.copy, args.dry_run)
            print('{} {} -> {}'.format(action, src, dst))

    print('Total images found: {}'.format(total_images))
    print('Output root: {}'.format(output_root))


if __name__ == '__main__':
    main()
