"""Evaluate completed frozen probes and collect scalar CNN results and histories."""
from __future__ import annotations
import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from run_cnn import HERE, PRESETS, EPOCHS, tasks, directory, condition_dir

def read_csv(path):
    if not path.is_file():
        return []
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def aggregate(rows, fields, metrics):
    groups = {}
    for row in rows:
        key = tuple(row[f] for f in fields)
        groups.setdefault(key, []).append(row)
    result = []
    for key, group in groups.items():
        item = dict(zip(fields, key))
        item['n_seeds'] = len(group)
        for metric in metrics:
            values = [float(r[metric]) for r in group]
            sd = statistics.stdev(values) if len(values) > 1 else float('nan')
            item[metric + '_mean'] = statistics.mean(values)
            item[metric + '_std'] = sd
            item[metric + '_sem'] = sd / math.sqrt(len(values))
        result.append(item)
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset', required=True, choices=PRESETS)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--evaluate', action='store_true', help='Explicitly open official test data after probing is complete')
    p.add_argument('--data-root', type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--allow-partial', action='store_true', help='Report missing tasks instead of failing')
    a = p.parse_args()
    if a.evaluate and a.data_root is None:
        p.error('--evaluate requires --data-root')
    if a.evaluate and a.allow_partial:
        p.error('Complete the preset before official-test evaluation')
    root = a.output_root.expanduser().resolve() / a.preset
    entries, missing, histories, diagnostics = [], [], [], []
    for task_id, row in enumerate(tasks(a.preset)):
        run = directory(root, row)
        pretraining = run / 'pretraining'
        try:
            condition = condition_dir(run)
            status = json.loads((condition / 'status.json').read_text())
            if status['state'] != 'complete':
                raise ValueError('Pretraining is incomplete')
            history = read_csv(condition / 'history.csv')
            config = json.loads((pretraining / 'config.json').read_text())
            presentations_per_epoch = 40000 * (int(row['k']) if a.preset == 'original' and isinstance(row['k'], int)
                                               else 1 if a.preset == 'full_reconstruction' else 32)
            updates_per_epoch = math.ceil(presentations_per_epoch / int(config['batch_size']))
            for h in history:
                h = dict(h)
                h.setdefault('optimizer_updates', int(h.get('epoch', h.get('comparison_epoch'))) * updates_per_epoch)
                h['image_presentations'] = int(h.get('epoch', h.get('comparison_epoch'))) * presentations_per_epoch
                histories.append(dict(row, **h))
            for file in (condition / 'batch_history.csv', condition / 'validation_updates.csv'):
                diagnostics.extend(dict(row, diagnostic=file.name, **h) for h in read_csv(file))
            if a.preset == 'early_diagnostics':
                continue
            for epoch in (EPOCHS if a.preset == 'full_reconstruction' else (None,)):
                probe = run / 'probe' if epoch is None else run / 'probes' / f'epoch{epoch:04d}'
                result = json.loads((probe / 'status.json').read_text())
                if result['state'] != 'complete':
                    raise ValueError('Probe is incomplete')
                summary = result['summary']
                checkpoint = condition / (f'epoch{epoch:04d}.pt' if epoch is not None else
                                           'update_125000.pt' if a.preset == 'fixed_budget' else 'best.pt')
                selected_epoch = epoch if epoch is not None else (50 if a.preset == 'fixed_budget' else int(status['summary']['best_epoch']))
                scalar = dict(row, pretraining_epoch=epoch or '',
                              selected_encoder_epoch=selected_epoch,
                              selected_encoder_updates=selected_epoch * updates_per_epoch,
                              selected_encoder_presentations=selected_epoch * presentations_per_epoch,
                              total_pretraining_updates=int(history[-1].get('epoch', history[-1].get('comparison_epoch'))) * updates_per_epoch,
                              validation_accuracy=summary['validation_accuracy'],
                              validation_loss=summary['validation_loss'],
                              classifier_best_epoch=summary['classifier_best_epoch'])
                entries.append((scalar, run, probe, checkpoint))
        except (FileNotFoundError, ValueError, KeyError) as error:
            missing.append(dict(task=task_id, **row, reason=str(error)))
    if missing and not a.allow_partial:
        raise RuntimeError(f'{len(missing)} tasks are incomplete; first: {missing[0]}')
    root.mkdir(parents=True, exist_ok=True)
    if a.preset == 'full_reconstruction' and not missing:
        val = aggregate([e[0] for e in entries], ['pretraining_epoch'], ['validation_accuracy', 'validation_loss'])
        chosen = max(val, key=lambda r: (r['validation_accuracy_mean'], -r['validation_loss_mean']))
        selection = {'pretraining_epoch': chosen['pretraining_epoch'],
                     'rule': 'Maximum mean validation accuracy; minimum mean validation loss breaks ties',
                     'selected_without_test_data': True}
        lock = root / 'validation_selection.json'
        if lock.exists() and json.loads(lock.read_text()) != selection:
            raise ValueError('Existing full-reconstruction selection differs')
        lock.write_text(json.dumps(selection, indent=2) + '\n')
        write_csv(root / 'validation_by_epoch.csv', val)
    rows = []
    for scalar, run, probe, checkpoint in entries:
        if a.evaluate:
            cmd = [sys.executable, str(HERE / 'evaluate_test.py'), '--data-root', str(a.data_root),
                   '--run-dir', str(run), '--probe-dir', str(probe), '--checkpoint', str(checkpoint), '--device', a.device]
            if a.preset == 'fixed_budget':
                cmd += ['--expected-updates', '125000']
            subprocess.run(cmd, check=True)
        saved = probe / 'test_result.json'
        if saved.is_file():
            test = json.loads(saved.read_text())
            scalar.update({k: test[k] for k in ('test_accuracy', 'test_error', 'test_correct', 'test_total')})
        rows.append(scalar)
    write_csv(root / 'per_run_results.csv', rows)
    write_csv(root / 'training_history.csv', histories)
    write_csv(root / 'early_diagnostics.csv', diagnostics)
    metrics = ['validation_accuracy', 'validation_loss']
    if rows and all('test_accuracy' in r for r in rows):
        metrics += ['test_accuracy']
    write_csv(root / 'aggregate_results.csv', aggregate(rows, ['k', 'rho', 'augmentation', 'pretraining_epoch'], metrics))
    (root / 'coverage.json').write_text(json.dumps({'expected_tasks': len(tasks(a.preset)), 'missing': missing,
                                                 'probe_rows': len(rows), 'official_test_requested': a.evaluate}, indent=2) + '\n')
    print(f'Collected {len(rows)} probe rows; {len(missing)} incomplete tasks')

if __name__ == '__main__':
    main()
