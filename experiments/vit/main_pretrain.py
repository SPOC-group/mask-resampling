# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import hashlib
import json
import math
import numpy as np
import os
import random
import subprocess
import time
from pathlib import Path

import sitecustomize  # Compatibility for the original timm version.
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae

from engine_pretrain import train_one_epoch
from engine_finetune import evaluate
from timm.utils import accuracy


def optional_int(value):
    if value is None:
        return None
    if str(value).lower() == 'none':
        return None
    return int(value)


def validate_pretrain_args(args):
    if not 0 <= args.mask_ratio <= 1:
        raise ValueError('--mask_ratio must be between 0 and 1')
    if args.mask_mode == 'none':
        if args.mask_ratio != 0:
            raise ValueError("--mask_mode none requires --mask_ratio 0")
        if args.reconstruction_loss_scope != 'all':
            raise ValueError(
                "--mask_mode none requires --reconstruction_loss_scope all")
    elif args.mask_ratio == 0:
        raise ValueError(
            "--mask_ratio 0 requires --mask_mode none so patches are not shuffled")
    if args.epochs <= 0:
        raise ValueError('--epochs must be positive')
    if args.stop_after_epoch is not None:
        if not 0 <= args.stop_after_epoch < args.epochs:
            raise ValueError(
                '--stop_after_epoch must be between 0 and epochs - 1')
        if args.stop_after_epoch < args.start_epoch:
            raise ValueError(
                '--stop_after_epoch must not precede --start_epoch')
    if args.val_interval <= 0:
        raise ValueError('--val-interval must be positive')
    if not 0 <= args.val_start_epoch < args.epochs:
        raise ValueError('--val-start-epoch must be between 0 and epochs - 1')
    if args.save_every_epoch_from is not None:
        if not 0 <= args.save_every_epoch_from < args.epochs:
            raise ValueError(
                '--save-every-epoch-from must be between 0 and epochs - 1')
    if args.max_retained_checkpoints < 0:
        raise ValueError('--max-retained-checkpoints must be non-negative')
    if len(set(args.save_epochs)) != len(args.save_epochs):
        raise ValueError('--save-epochs must not contain duplicates')
    for epoch in args.save_epochs:
        if not 0 <= epoch < args.epochs:
            raise ValueError('--save-epochs values must be between 0 and epochs - 1')
        if args.stop_after_epoch is not None and epoch > args.stop_after_epoch:
            raise ValueError('--save-epochs values must not follow --stop-after-epoch')


def get_run_end_epoch(args):
    if args.stop_after_epoch is not None:
        return args.stop_after_epoch
    return args.epochs - 1


def should_validate_epoch(args, epoch):
    return (
        epoch >= args.val_start_epoch
        and (epoch - args.val_start_epoch) % args.val_interval == 0
    )


def should_save_checkpoint(args, epoch):
    dense_save = (
        args.save_every_epoch_from is not None
        and epoch >= args.save_every_epoch_from
    )
    selected_save = epoch in args.save_epochs
    return (
        epoch % 20 == 0
        or dense_save
        or selected_save
        or epoch == get_run_end_epoch(args)
    )


def prune_old_checkpoints(args):
    """Keep only the newest recovery checkpoints when a retention limit is set."""
    if (
            not args.output_dir
            or not misc.is_main_process()
            or args.max_retained_checkpoints == 0):
        return

    output_dir = Path(args.output_dir)
    checkpoints = sorted(
        output_dir.glob('checkpoint-*.pth'),
        key=lambda path: int(path.stem.rsplit('-', 1)[1]))
    for checkpoint_path in checkpoints[:-args.max_retained_checkpoints]:
        checkpoint_path.unlink()
        print('Deleted superseded recovery checkpoint: {}'.format(
            checkpoint_path.name))


class IndexedImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        return sample, target, index


def build_pretrain_transform(args, is_train):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    if is_train and args.aug_mode == 'standard':
        return transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])

    resize_size = int(args.input_size / (224 / 256))
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=3),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        normalize])


def get_git_commit():
    return 'not_recorded'


def get_environment_info(args):
    gpu_name = None
    if torch.cuda.is_available():
        device_index = getattr(args, 'gpu', 0)
        gpu_name = torch.cuda.get_device_name(device_index)
    return {
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(),
        'gpu_name': gpu_name,
        'world_size': misc.get_world_size(),
    }


def model_state_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(b'\0')
        digest.update(str(tensor.dtype).encode('ascii'))
        digest.update(b'\0')
        digest.update(str(tuple(tensor.shape)).encode('ascii'))
        digest.update(b'\0')
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_initial_model(args, model):
    if args.init_checkpoint:
        checkpoint_path = Path(args.init_checkpoint).expanduser().resolve()
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        checkpoint_model = checkpoint.get('model', checkpoint)
        model.load_state_dict(checkpoint_model, strict=True)

        expected_sha256 = checkpoint.get('model_sha256')
        actual_sha256 = model_state_sha256(model)
        if expected_sha256 and expected_sha256 != actual_sha256:
            raise RuntimeError(
                'Initial model checksum mismatch for {}: expected {}, got {}'.format(
                    checkpoint_path, expected_sha256, actual_sha256))
        args.init_checkpoint = str(checkpoint_path)
        args.initial_model_sha256 = actual_sha256
        print('Loaded model-only initialization: {}'.format(checkpoint_path))
    else:
        args.initial_model_sha256 = model_state_sha256(model)

    print('Initial model SHA256: {}'.format(args.initial_model_sha256))


def save_run_metadata(args):
    if not args.output_dir or not misc.is_main_process():
        return
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'args.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    metadata = {
        'git_commit': get_git_commit(),
        'environment': get_environment_info(args),
        'created_at': datetime.datetime.now().isoformat(),
    }
    with open(os.path.join(args.output_dir, 'environment.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def cleanup_intermediate_checkpoints(args):
    if not args.output_dir or not misc.is_main_process():
        return
    if args.keep_intermediate_checkpoints:
        print('Keeping intermediate checkpoints by request.')
        return

    output_dir = Path(args.output_dir)
    final_checkpoint = output_dir / 'checkpoint-{}.pth'.format(
        get_run_end_epoch(args))
    if not final_checkpoint.is_file():
        print('Skipping checkpoint cleanup because final checkpoint is missing: {}'.format(
            final_checkpoint))
        return

    deleted = []
    for checkpoint_path in sorted(output_dir.glob('checkpoint-*.pth')):
        if checkpoint_path == final_checkpoint:
            continue
        if checkpoint_path.is_file():
            checkpoint_path.unlink()
            deleted.append(checkpoint_path.name)

    if deleted:
        print('Deleted intermediate checkpoints after successful pretraining: {}'.format(
            ', '.join(deleted)))
    else:
        print('No intermediate checkpoints to delete after successful pretraining.')


def assert_finite_metrics(name, metrics, epoch):
    non_finite = {
        key: value for key, value in metrics.items()
        if isinstance(value, (int, float)) and not math.isfinite(value)
    }
    if non_finite:
        raise FloatingPointError(
            'Non-finite {} metrics at epoch {}: {}'.format(
                name, epoch, non_finite))


def assert_finite_training_state(model, loss_scaler, epoch):
    bad_tensors = []
    for name, tensor in model.state_dict().items():
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
            bad_tensors.append(name)
    if bad_tensors:
        raise FloatingPointError(
            'Model contains non-finite tensors after epoch {}: {}'.format(
                epoch, ', '.join(bad_tensors[:10])))

    scaler_state = loss_scaler.state_dict()
    scale = float(scaler_state.get('scale', 1.0))
    if not math.isfinite(scale) or scale <= 0:
        raise FloatingPointError(
            'AMP gradient scale is invalid after epoch {}: {}'.format(
                epoch, scale))


def parse_wandb_tags(tags):
    if not tags:
        return None
    return [tag.strip() for tag in tags.split(',') if tag.strip()]


def init_wandb(args):
    """External experiment tracking is disabled in the supplementary package."""
    return None


def prepare_static_mask_cache(args, num_patches, dataset_len):
    if args.mask_mode != 'static_per_image':
        return None

    num_samples = args.num_train_samples or dataset_len
    if num_samples < dataset_len:
        raise ValueError(
            '--num_train_samples ({}) must be >= the training dataset size ({})'.format(
                num_samples, dataset_len))

    if not args.static_mask_cache:
        return None

    cache_path = Path(args.static_mask_cache)
    if misc.is_main_process() and not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ids_shuffle = models_mae.generate_static_ids_shuffle(
            num_samples, num_patches, args.static_mask_seed)
        torch.save({
            'ids_shuffle': ids_shuffle,
            'num_samples': num_samples,
            'num_patches': num_patches,
            'static_mask_seed': args.static_mask_seed,
        }, cache_path)
        print('Saved static mask cache to {}'.format(cache_path))

    if misc.is_dist_avail_and_initialized():
        torch.distributed.barrier()

    cache = torch.load(cache_path, map_location='cpu')
    ids_shuffle = cache['ids_shuffle'] if isinstance(cache, dict) else cache
    if ids_shuffle.shape != (num_samples, num_patches):
        raise ValueError(
            'static mask cache shape {} does not match expected {}'.format(
                tuple(ids_shuffle.shape), (num_samples, num_patches)))
    if isinstance(cache, dict) and cache.get('static_mask_seed') != args.static_mask_seed:
        raise ValueError(
            'static mask cache seed {} does not match --static_mask_seed {}'.format(
                cache.get('static_mask_seed'), args.static_mask_seed))
    print('Loaded static mask cache from {}'.format(cache_path))
    return ids_shuffle


@torch.no_grad()
def save_feature_spectrum(data_loader, model, device, args, epoch):
    if not args.save_feature_spectrum or not args.output_dir:
        return

    model.eval()
    features = []
    for batch_idx, batch in enumerate(data_loader):
        images = batch[0].to(device, non_blocking=True)
        feature_mask_mode = 'none' if args.mask_mode == 'none' else 'dynamic'
        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            output = model(
                images,
                mask_ratio=args.mask_ratio,
                mask_mode=feature_mask_mode,
                reconstruction_loss_scope=args.reconstruction_loss_scope)
        features.append(output[3].detach().float().cpu())
        if batch_idx + 1 >= args.feature_spectrum_max_batches:
            break

    if not features or not misc.is_main_process():
        return

    features = torch.cat(features, dim=0)
    features = features - features.mean(dim=0, keepdim=True)
    _, singular_values, _ = torch.svd(features)
    singular_values = singular_values.numpy()
    probs = singular_values / (singular_values.sum() + 1e-12)
    effective_rank = float(np.exp(-(probs * np.log(probs + 1e-12)).sum()))

    spectrum_dir = Path(args.output_dir) / 'feature_spectrum'
    spectrum_dir.mkdir(parents=True, exist_ok=True)
    np.save(spectrum_dir / 'singular_values_epoch_{:04d}.npy'.format(epoch), singular_values)
    with open(spectrum_dir / 'summary_epoch_{:04d}.json'.format(epoch), 'w', encoding='utf-8') as f:
        json.dump({'effective_rank': effective_rank, 'num_features': int(features.shape[0])}, f, indent=2)

    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(singular_values)
        plt.xlabel('component')
        plt.ylabel('singular value')
        plt.tight_layout()
        plt.savefig(spectrum_dir / 'singular_values_epoch_{:04d}.png'.format(epoch))
        plt.close()
    except Exception as exc:
        print('Skipping feature spectrum plot: {}'.format(exc))


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    # parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--stop-after-epoch', '--stop_after_epoch', default=None,
                        type=optional_int,
                        help='Optional zero-based final epoch; --epochs still controls the LR schedule.')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--mask_mode', default='dynamic',
                        choices=['dynamic', 'static_per_image', 'global_static', 'none'],
                        help='Mask sampling mode. Default preserves original dynamic MAE masking.')
    parser.add_argument('--reconstruction_loss_scope', default='masked',
                        choices=['masked', 'all'],
                        help='Compute reconstruction MSE on masked patches or on all patches.')
    parser.add_argument('--num_train_samples', default=None, type=optional_int,
                        help='Number of training samples for static mask cache; defaults to dataset length.')
    parser.add_argument('--static_mask_seed', default=0, type=int,
                        help='Seed used for static_per_image and global_static mask permutations.')
    parser.add_argument('--static_mask_cache', default=None, type=str,
                        help='Optional path to save/load static per-image patch permutations.')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    # parser.set_defaults(norm_pix_loss=False)
    parser.set_defaults(norm_pix_loss=True)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    # parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet100', type=str,
                        help='dataset path')
    parser.add_argument('--aug_mode', default='standard', choices=['standard', 'no_geom'],
                        help='standard uses original pretrain crops/flips; no_geom uses deterministic resize/center crop.')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--wandb_project', default='',
                        help='W&B project name. Empty disables W&B logging.')
    parser.add_argument('--wandb_entity', default='',
                        help='Optional W&B entity/team.')
    parser.add_argument('--wandb_run_name', default='',
                        help='Optional W&B run name; defaults to output directory name.')
    parser.add_argument('--wandb_group', default='',
                        help='Optional W&B group name.')
    parser.add_argument('--wandb_tags', default='',
                        help='Optional comma-separated W&B tags.')
    parser.add_argument('--wandb_id', default='',
                        help='Optional stable W&B run id for resuming a run.')
    parser.add_argument('--wandb_mode', default='disabled',
                        choices=['online', 'offline', 'disabled'],
                        help='W&B mode. Use offline to log locally and sync later.')
    parser.add_argument('--wandb_log_freq', default=20, type=int,
                        help='Log train batches to W&B every N optimizer steps.')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--init_checkpoint', default='',
                        help='load model weights only before creating a fresh optimizer')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-every-epoch-from', '--save_every_epoch_from',
                        default=None, type=optional_int,
                        help='Also save every epoch from this zero-based epoch onward.')
    parser.add_argument('--save-epochs', '--save_epochs', default=(), nargs='+',
                        type=int,
                        help='Also save these specific zero-based epochs.')
    parser.add_argument('--keep-intermediate-checkpoints',
                        '--keep_intermediate_checkpoints', action='store_true',
                        help='Do not delete intermediate checkpoints after a successful run.')
    parser.add_argument('--max-retained-checkpoints',
                        '--max_retained_checkpoints', default=0, type=int,
                        help='Keep only the newest N checkpoints while training; 0 keeps all until successful cleanup.')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', '--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')


    # new
    parser.add_argument('--lamb', type=float, default=0)
    parser.add_argument('--reg', type=str, default='none', choices=['none', 'entropy', 'spectral'])
    parser.add_argument('--val-interval', default=10, type=int)
    parser.add_argument('--val-start-epoch', '--val_start_epoch', default=0,
                        type=int,
                        help='Zero-based epoch at which periodic validation begins.')
    parser.add_argument('--save_feature_spectrum', action='store_true',
                        help='Optionally save validation feature singular values/effective rank.')
    parser.add_argument('--feature_spectrum_max_batches', default=8, type=int,
                        help='Validation batches to use when --save_feature_spectrum is enabled.')

    return parser


def main(args):
    validate_pretrain_args(args)
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = True

    transform_train = build_pretrain_transform(args, is_train=True)
    transform_val = build_pretrain_transform(args, is_train=False)

    dataset_train = IndexedImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_val)
    print(dataset_train)
    print(dataset_val)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))

        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_val = %s" % str(sampler_val))

    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    load_initial_model(args, model)
    static_ids_shuffle = prepare_static_mask_cache(
        args, model.patch_embed.num_patches, len(dataset_train))
    model.set_static_mask_cache(static_ids_shuffle)

    if False:
        checkpoint = torch.load('checkpoint-199.pth', map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()

        # interpolate position embedding
        #interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)


    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    save_run_metadata(args)
    wandb_run = init_wandb(args)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    run_end_epoch = get_run_end_epoch(args)
    print('Training schedule spans {} epochs; this process will stop after epoch {}.'.format(
        args.epochs, run_end_epoch))
    start_time = time.time()
    max_accuracy = 0.0
    test_stats = {'acc1': 0.0, 'acc5': 0.0, 'loss': 0.0}
    eval_model_kwargs = {}
    if args.mask_mode == 'none':
        eval_model_kwargs = {
            'mask_ratio': 0,
            'mask_mode': 'none',
            'reconstruction_loss_scope': 'all',
        }
    last_validation_epoch = None
    for epoch in range(args.start_epoch, run_end_epoch + 1):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            wandb_run=wandb_run,
            args=args
        )
        checkpoint_saved = False
        if args.output_dir and should_save_checkpoint(args, epoch):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
            prune_old_checkpoints(args)
            checkpoint_saved = True

        validation_ran = should_validate_epoch(args, epoch)
        if validation_ran:
            test_stats = evaluate(
                data_loader_val, model, device,
                model_kwargs=eval_model_kwargs)
            assert_finite_metrics('validation', test_stats, epoch)
            last_validation_epoch = epoch
            save_feature_spectrum(data_loader_val, model, device, args, epoch)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'mask_mode': args.mask_mode,
                        'aug_mode': args.aug_mode,
                        'mask_ratio': args.mask_ratio,
                        'reconstruction_loss_scope': args.reconstruction_loss_scope,
                        'validation_ran': validation_ran,
                        'static_mask_seed': args.static_mask_seed,}

        if wandb_run is not None:
            wandb_stats = {f'train/epoch_{k}': v for k, v in train_stats.items()}
            if validation_ran:
                wandb_stats.update({f'val/{k}': v for k, v in test_stats.items()})
            wandb_stats.update({
                'epoch': epoch,
                'val/max_acc1': max_accuracy,
                'checkpoint/saved': int(checkpoint_saved),
            })
            wandb_run.log(wandb_stats, step=(epoch + 1) * 1000)
            wandb_run.summary['max_acc1'] = max_accuracy

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    assert_finite_metrics('training', train_stats, run_end_epoch)
    assert_finite_training_state(model_without_ddp, loss_scaler, run_end_epoch)
    if last_validation_epoch is None:
        print('Warning: no validation was run during this process.')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if wandb_run is not None:
        wandb_run.summary['training_time'] = total_time_str
        wandb_run.finish()
    cleanup_intermediate_checkpoints(args)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
