#!/usr/bin/env python3
"""Render selected paper figures from the single plotting notebook."""
from pathlib import Path
import argparse
import csv
import json
import os

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure', nargs='+', metavar='ID',
                        help='Figure IDs, e.g. 3 5 18; default: all 18 paper figures.')
    parser.add_argument('--list', action='store_true', help='List figure IDs and outputs; no plotting imports.')
    parser.add_argument('--no-tex', action='store_true', help='Use built-in mathtext; no TeX installation.')
    args = parser.parse_args()
    with (ROOT / 'docs/figure_index.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    by_id = {r['notebook_cell_id'].removeprefix('plot-').lstrip('0').upper(): r for r in rows}
    if args.list:
        for key, row in by_id.items():
            print(f"{key:>2}  {row['title']}\n    {row['pdf']}")
        return
    selected = [str(x).lstrip('0').upper() for x in args.figure] if args.figure else list(by_id)
    unknown = set(selected) - set(by_id)
    if unknown:
        parser.error('Unknown figure IDs: ' + ', '.join(sorted(unknown)))
    wanted = {by_id[key]['notebook_cell_id'] for key in selected}
    os.chdir(ROOT)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.show = lambda *a, **kw: None
    notebook_path = ROOT / 'notebooks/figures.ipynb'
    notebook = json.loads(notebook_path.read_text())
    namespace = {'__name__': '__main__'}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code' or cell['id'] not in wanted | {'setup'}:
            continue
        source = ''.join(cell['source'])
        if args.no_tex and cell['id'] == 'setup':
            source = source.replace('USE_TEX = shutil.which("latex") is not None and shutil.which("dvipng") is not None',
                                    'USE_TEX = False')
        print('Rendering', cell['id'], flush=True)
        exec(compile(source, str(notebook_path) + ':' + cell['id'], 'exec'), namespace)
        plt.close('all')
    missing = [by_id[key]['pdf'] for key in selected if not (ROOT / by_id[key]['pdf']).is_file()]
    if missing:
        raise RuntimeError('Missing PDFs: ' + ', '.join(missing))
    print(f'Finished: {len(set(selected))} PDFs in plots/. The notebook and input data were not modified.')


if __name__ == '__main__':
    main()
