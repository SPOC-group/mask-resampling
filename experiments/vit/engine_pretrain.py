# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable
from math import ceil
import torch
import warnings

import util.misc as misc
import util.lr_sched as lr_sched

import torch.distributed as dist

from loss_func import uniformity_loss


  
def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    wandb_run=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    total = []
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if len(batch) == 3:
            samples, targets, sample_indices = batch
        else:
            samples, targets = batch
            sample_indices = None

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if sample_indices is not None:
            sample_indices = sample_indices.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss_mae, _, mask, cls_feats, outputs = model(
                samples, mask_ratio=args.mask_ratio,
                mask_mode=args.mask_mode, sample_indices=sample_indices,
                static_mask_seed=args.static_mask_seed,
                reconstruction_loss_scope=args.reconstruction_loss_scope)

            if args.reg == 'none':
                loss_reg = torch.zeros_like(loss_mae)
            else:
                loss_reg = uniformity_loss(cls_feats)
            
            loss_ce = torch.nn.functional.cross_entropy(outputs, targets)

        loss = loss_mae + args.lamb * loss_reg + loss_ce

        loss_mae_value = loss_mae.item()
        loss_reg_value = loss_reg.item()
        loss_ce_value = loss_ce.item()
        loss_value = loss.item()
        loss_values = {
            'total': loss_value,
            'mae': loss_mae_value,
            'regularization': loss_reg_value,
            'classification': loss_ce_value,
        }
        non_finite = {
            name: value for name, value in loss_values.items()
            if not math.isfinite(value)
        }
        if non_finite:
            raise FloatingPointError(
                'Non-finite loss at epoch {}, iteration {}: {}'.format(
                    epoch, data_iter_step, non_finite))
        train_acc = (outputs.argmax(dim=1) == targets).float().mean()
        mask_fraction = mask.float().mean()
        static_mask_consistent = None
        if args.mask_mode == 'static_per_image' and sample_indices is not None and data_iter_step == 0:
            module = model.module if hasattr(model, 'module') else model
            len_keep = int(mask.shape[1] * (1 - args.mask_ratio))
            ids_keep_a = module.get_static_ids_shuffle(
                sample_indices[:1], mask.shape[1], samples.device,
                args.static_mask_seed)[:, :len_keep]
            ids_keep_b = module.get_static_ids_shuffle(
                sample_indices[:1], mask.shape[1], samples.device,
                args.static_mask_seed)[:, :len_keep]
            static_mask_consistent = float(torch.equal(ids_keep_a, ids_keep_b))


        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_mae=loss_mae_value)
        metric_logger.update(loss_reg=loss_reg_value)
        metric_logger.update(loss_ce=loss_ce_value)
        metric_logger.update(train_acc=train_acc)
        metric_logger.update(mask_fraction=mask_fraction)
        metric_logger.update(static_mask_consistent=static_mask_consistent)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_mae_value_reduce = misc.all_reduce_mean(loss_mae_value)
        loss_reg_value_reduce = misc.all_reduce_mean(loss_reg_value)
        loss_ce_value_reduce = misc.all_reduce_mean(loss_ce_value)
        train_acc_reduce = misc.all_reduce_mean(train_acc)
        mask_fraction_reduce = misc.all_reduce_mean(mask_fraction)

        if (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if log_writer is not None:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('train_loss_mae', loss_mae_value_reduce, epoch_1000x)
                log_writer.add_scalar('train_loss_reg', loss_reg_value_reduce, epoch_1000x)
                log_writer.add_scalar('train_loss_ce', loss_ce_value_reduce, epoch_1000x)
                log_writer.add_scalar('train_acc', train_acc_reduce, epoch_1000x)
                log_writer.add_scalar('mask_fraction', mask_fraction_reduce, epoch_1000x)
                log_writer.add_text('mask_mode', args.mask_mode, epoch_1000x)
                log_writer.add_text(
                    'reconstruction_loss_scope',
                    args.reconstruction_loss_scope,
                    epoch_1000x)
                log_writer.add_text('aug_mode', args.aug_mode, epoch_1000x)
                log_writer.add_scalar('mask_ratio', args.mask_ratio, epoch_1000x)
                log_writer.add_scalar('static_mask_seed', args.static_mask_seed, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)

            if wandb_run is not None:
                log_freq = max(1, getattr(args, 'wandb_log_freq', 20))
                optimizer_step = data_iter_step // accum_iter
                if data_iter_step == 0 or (optimizer_step + 1) % log_freq == 0:
                    wandb_run.log({
                        'train/loss': loss_value_reduce,
                        'train/loss_mae': loss_mae_value_reduce,
                        'train/loss_reg': loss_reg_value_reduce,
                        'train/loss_ce': loss_ce_value_reduce,
                        'train/acc': train_acc_reduce,
                        'train/mask_fraction': mask_fraction_reduce,
                        'train/lr': lr,
                        'epoch': epoch,
                    }, step=epoch_1000x)
        # break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
