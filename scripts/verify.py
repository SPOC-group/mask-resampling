#!/usr/bin/env python3
"""Check saved inputs, figure mappings, notebook cleanliness and source syntax without training."""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    root = root.resolve()
    def local(name):
        target = (root / name).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f'Path escapes repository: {name}')
        return target
    manifest = json.loads((root / 'data/manifest.json').read_text())
    errors = []
    for row in manifest['files']:
        p = local(row['path'])
        if not p.is_file() or digest(p) != row['sha256']:
            errors.append('Missing or changed input: ' + row['path'])
    nb = json.loads((root / 'notebooks/figures.ipynb').read_text())
    ids = [c['id'] for c in nb['cells']]
    if len(set(ids)) != len(ids):
        errors.append('Duplicate notebook cell IDs')
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            if c.get('outputs') or c.get('execution_count') is not None:
                errors.append('Saved notebook execution state: ' + c['id'])
            try:
                ast.parse(''.join(c['source']))
            except SyntaxError as exc:
                errors.append(f"Notebook {c['id']}: {exc}")
    with (root / 'docs/figure_index.csv').open(newline='') as f:
        figures = list(csv.DictReader(f))
    for row in figures:
        if row['notebook_cell_id'] not in ids:
            errors.append('Missing figure cell: ' + row['notebook_cell_id'])
        for name in row['data_files'].split(';'):
            if not local(name).is_file():
                errors.append('Missing figure input: ' + name)
        if Path(row['pdf']).parent.as_posix() != 'plots':
            errors.append('Unexpected plot output: ' + row['pdf'])
    python_files = list((root / 'experiments').rglob('*.py')) + list((root / 'scripts').glob('*.py'))
    python_files = [p for p in python_files if not set(p.relative_to(root).parts) & {'vendor', '.venv', '__pycache__', 'runs'}]
    for p in python_files:
        try:
            ast.parse(p.read_text())
        except SyntaxError as exc:
            errors.append(f'{p.relative_to(root)}: {exc}')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'OK: {len(manifest["files"])} data checksums, {len(figures)} figure mappings, '
          f'{len(python_files)} Python sources, clean notebook.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    verify(ROOT)


if __name__ == '__main__':
    main()
