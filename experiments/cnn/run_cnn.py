"""Independent CNN experiment tasks using the original trainers and frozen probe."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RHOS = (0.05, 0.1, 0.171875, 0.25, 0.375, 0.5, 0.625, 0.75, 0.828125, 0.9, 0.95)
KS = (1, 2, 4, 8, 16, 32)
EPOCHS = (1, 2, 4, 8, 16, 25, 50, 100, 200, 300, 400, 600, 800)
PRESETS = ('original', 'augmentation', 'fixed_lr', 'fixed_budget', 'early_diagnostics', 'full_reconstruction')

def tasks(preset):
    rows = []
    def add(seed, k, rho, aug='none'):
        rows.append(dict(seed=seed, k=k, rho=rho, augmentation=aug))
    for seed in range(5):
        if preset == 'original':
            for k in KS:
                for rho in RHOS[1:-1]:
                    add(seed, k, rho)
            for rho in RHOS:
                add(seed, 'dynamic', rho)
            add(seed, 'dynamic', 0.625, 'crop_flip')
        elif preset == 'augmentation':
            for aug in ('none', 'crop_flip'):
                for k in (1, 'dynamic'):
                    for rho in RHOS:
                        add(seed, k, rho, aug)
        elif preset in ('fixed_lr', 'fixed_budget', 'early_diagnostics'):
            for k in (*KS, 'dynamic'):
                add(seed, k, 0.625)
        elif preset == 'full_reconstruction':
            add(seed, 'unmasked', 0.0)
        else:
            raise ValueError(preset)
    return rows

def directory(root, row):
    label = f"K{row['k']}" if isinstance(row['k'], int) else row['k']
    token = f"{row['rho']:.12g}".replace('.', 'p')
    return root / 'runs' / row['augmentation'] / f"seed{row['seed']}" / label / f'rho{token}'

def worker_count(preset, row):
    return 0 if preset in ('original', 'full_reconstruction') and row['augmentation'] == 'none' else 4

def training_command(a, row, run, protocol):
    workers = worker_count(a.preset, row) if a.num_workers is None else a.num_workers
    common = ['--data-root', str(a.data_root), '--output-dir', str(run / 'pretraining'),
              '--batch-size', '512', '--lr', '0.001', '--weight-decay', '0.0001',
              '--clip-grad', '5.0', '--scheduler-factor', '0.5', '--scheduler-patience', '5',
              '--min-lr', '0.00001', '--num-workers', str(workers), '--device', a.device, '--no-amp']
    if a.preset == 'full_reconstruction':
        return [sys.executable, str(HERE / 'cifar10_conv_mae.py'), *common,
                '--condition-spec', 'full_reconstruction:0', '--ae-seeds', str(row['seed']),
                '--split-seed', str(2026 + row['seed']), '--pretrain-per-class', '4000',
                '--validation-per-class', '1000', '--probe-per-class', '0',
                '--epochs', '800', '--min-epochs', '800', '--patience', '25',
                '--checkpoint-epochs', *map(str, EPOCHS)]
    constant = a.preset in ('fixed_lr', 'fixed_budget', 'early_diagnostics')
    common += ['--reference-root', str(protocol), '--ae-seed', str(row['seed']),
               '--architecture', 'conv', '--patch-size', '4', '--min-epochs', '20',
               '--epochs', '50' if a.preset == 'fixed_budget' else '460',
               '--patience', '50' if constant else '15']
    if a.preset == 'original' and isinstance(row['k'], int):
        cmd = [sys.executable, str(HERE / 'cifar10_conv_mae_kmask_control.py'), *common,
               '--masks-per-sample', str(row['k']), '--rhos', str(row['rho'])]
    else:
        cmd = [sys.executable, str(HERE / 'cifar10_conv_mae_dynamic_step_matched_k32.py'), *common,
               '--reference-k', '32', '--rho', str(row['rho']),
               '--lr-schedule', 'constant' if constant else 'plateau']
        if isinstance(row['k'], int):
            cmd += ['--fixed-mask-k', str(row['k'])]
        if a.preset == 'fixed_budget':
            cmd += ['--no-early-stopping']
        if a.preset == 'early_diagnostics':
            cmd += ['--no-resume', '--diagnostic-first-epoch']
    if row['augmentation'] == 'crop_flip':
        cmd += ['--dynamic-train-augmentation']
    return cmd

def condition_dir(run):
    dirs = [p for p in (run / 'pretraining/conditions').iterdir() if p.is_dir()]
    if len(dirs) != 1:
        raise ValueError('Expected one pretraining condition per task')
    return dirs[0]

def fixed_checkpoint(run):
    import torch
    condition = condition_dir(run)
    final = torch.load(condition / 'last.pt', map_location='cpu', weights_only=False)
    if final['optimizer_steps'] != 125000 or final['epoch'] != 50:
        raise ValueError('Fixed-budget encoder must have exactly 125000 updates and 50 epochs')
    if final['condition']['early_stopping_enabled'] is not False or final['test_data_opened'] is not False:
        raise ValueError('Incorrect fixed-budget training protocol')
    if len(final['history']) != 50 or any(float(r['lr']) != 0.001 for r in final['history']):
        raise ValueError('Fixed-budget run must retain learning rate 0.001')
    final['checkpoint_selection'] = 'fixed_optimizer_update_budget'
    target = condition / 'update_125000.pt'
    temporary = target.with_suffix('.tmp')
    torch.save(final, temporary)
    temporary.replace(target)
    return target

def probe_command(a, row, run, protocol, epoch=None):
    workers = (0 if a.preset in ('original', 'full_reconstruction') else 4) if a.num_workers is None else a.num_workers
    out = run / 'probe' if epoch is None else run / 'probes' / f'epoch{epoch:04d}'
    cmd = [sys.executable, str(HERE / 'probe_one.py'), '--data-root', str(a.data_root),
           '--reference-root', str(protocol), '--pretraining-root', str(run / 'pretraining'),
           '--output-dir', str(out), '--device', a.device, '--num-workers', str(workers)]
    if a.preset in ('original', 'full_reconstruction'):
        cmd += ['--validation-order', 'protocol']
    if epoch is not None:
        cmd += ['--full-reconstruction', '--checkpoint', str(condition_dir(run) / f'epoch{epoch:04d}.pt')]
    if a.preset == 'fixed_budget':
        cmd += ['--checkpoint', str(fixed_checkpoint(run)), '--expected-updates', '125000']
    return cmd

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset', choices=PRESETS, required=True)
    p.add_argument('--list', action='store_true', help='Print indexed tasks; does not train')
    p.add_argument('--task', type=int, help='Zero-based task; defaults to SLURM_ARRAY_TASK_ID')
    p.add_argument('--data-root', type=Path)
    p.add_argument('--output-root', type=Path)
    p.add_argument('--protocol-root', type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--num-workers', type=int, default=None)
    p.add_argument('--stage', choices=('train', 'probe', 'both'), default='both')
    p.add_argument('--dry-run', action='store_true', help='Print the training command without reading data')
    a = p.parse_args()
    grid = tasks(a.preset)
    if a.list:
        print(json.dumps([dict(task=i, **row) for i, row in enumerate(grid)], indent=2))
        return
    index = a.task if a.task is not None else int(os.environ.get('SLURM_ARRAY_TASK_ID', '-1'))
    if not 0 <= index < len(grid):
        p.error(f'Choose a task from 0 through {len(grid)-1}')
    if a.data_root is None or a.output_root is None:
        p.error('--data-root and --output-root are required')
    a.data_root = a.data_root.expanduser().resolve()
    root = a.output_root.expanduser().resolve() / a.preset
    row = grid[index]
    run = directory(root, row)
    protocol = ((a.protocol_root or a.output_root / 'protocols').expanduser().resolve()
                / f"seed{row['seed']}" / 'protocol')
    if a.preset == 'full_reconstruction':
        protocol = run / 'pretraining'
    cmd = training_command(a, row, run, protocol)
    if a.dry_run:
        print(json.dumps({'task': row, 'train_command': cmd, 'run_dir': str(run)}, indent=2))
        return
    if a.preset != 'full_reconstruction' and not (protocol / 'split_indices.npz').is_file():
        raise FileNotFoundError('Run prepare_protocols.py before dispatching masked tasks')
    run.mkdir(parents=True, exist_ok=True)
    config = {'preset': a.preset, 'task': row, 'train_command': cmd}
    invocation = run / 'invocation.json'
    if invocation.exists() and json.loads(invocation.read_text()) != config:
        raise ValueError('Existing output has different parameters; use a separate output root')
    invocation.write_text(json.dumps(config, indent=2) + '\n')
    if a.stage in ('train', 'both'):
        subprocess.run(cmd, check=True)
    if a.preset == 'early_diagnostics':
        return
    if a.stage in ('probe', 'both'):
        for epoch in (EPOCHS if a.preset == 'full_reconstruction' else (None,)):
            subprocess.run(probe_command(a, row, run, protocol, epoch), check=True)

if __name__ == '__main__':
    main()
