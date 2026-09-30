"""Small file interface shared by the two independent-task runners."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd


def write_json(path, value):
    # Missing diagnostic values are JSON null, never non-standard NaN tokens.
    def clean(v):
        if isinstance(v, dict): return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)): return [clean(x) for x in v]
        if isinstance(v, np.generic): return clean(v.item())
        if isinstance(v, float) and not np.isfinite(v): return None
        return v
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_rows(path, rows):
    temporary = path.with_suffix('.csv.tmp')
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def prepare_run(root, config):
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()): raise FileExistsError('Use an empty output directory')
    for i, task in enumerate(config['tasks']): task['task_id'] = i
    config['task_count'] = len(config['tasks'])
    if not config['task_count']: raise ValueError('Empty task grid')
    (root / 'tasks').mkdir()
    write_json(root / 'run_config.json', config)
    print(json.dumps(dict(tasks=config['task_count'], task_id_range=[0, config['task_count']-1])))


def load_config(root):
    return json.loads((root / 'run_config.json').read_text())


def task_index(value, config):
    if value is None:
        value = os.environ.get('SLURM_ARRAY_TASK_ID')
    if value is None: raise ValueError('Provide --task-id or SLURM_ARRAY_TASK_ID')
    index = int(value)
    if not 0 <= index < len(config['tasks']): raise ValueError('task-id outside prepared grid')
    return index


def task_folder(root, index):
    folder = root / 'tasks' / f'task_{index:04d}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'COMPLETED.json').exists():
        print(f'Task {index} already complete; preserved')
        return None
    return folder


def aggregate_run(root, *, methods, group_columns, raw_name, summary_name, allow_missing=False):
    config = load_config(root)
    frames, missing = [], []
    for task in config['tasks']:
        folder = root / 'tasks' / f"task_{task['task_id']:04d}"
        if not (folder / 'COMPLETED.json').exists():
            missing.append(task['task_id'])
            continue
        frame = pd.read_csv(folder / 'final.csv', float_precision='round_trip')
        if len(frame) != len(methods) or sorted(frame.method) != sorted(methods):
            raise ValueError(f"Wrong estimator coverage for task {task['task_id']}")
        for key, value in task.items():
            if key == 'alpha':
                if not np.allclose(frame[key], value, rtol=1e-12, atol=0): raise ValueError('Alpha mismatch')
            elif not frame[key].eq(value).all(): raise ValueError(f'{key} mismatch')
        frames.append(frame)
    if missing and not allow_missing:
        raise RuntimeError(f'{len(missing)} incomplete tasks; use --allow-missing for an explicit partial aggregate')
    if not frames: raise RuntimeError('No completed tasks')
    raw = pd.concat(frames, ignore_index=True)
    if raw.duplicated(['task_id', 'method']).any(): raise ValueError('Duplicate fit')
    named = {'num_seeds': ('seed', 'nunique'), 'num_converged': ('converged', 'sum')}
    for metric in ('cosine_abs', 'cosine_sq', 'spectral_gap', 'inverse_participation_ratio'):
        if metric in raw:
            for stat in ('mean', 'std', 'sem'): named[f'{metric}_{stat}'] = (metric, stat)
    summary = raw.groupby(group_columns, as_index=False).agg(**named)
    write_rows(root / raw_name, raw.to_dict('records'))
    write_rows(root / summary_name, summary.to_dict('records'))
    coverage = dict(expected_tasks=len(config['tasks']), completed_tasks=len(frames),
                    missing_task_ids=missing, partial=bool(missing),
                    fits=len(raw), converged_fits=int(raw.converged.sum()),
                    all_converged=bool(raw.converged.all()),
                    aggregation='All completed terminal fits; sample SD (ddof=1) and SEM=SD/sqrt(n).')
    write_json(root / 'AGGREGATED.json', coverage)
    print(json.dumps(coverage))
