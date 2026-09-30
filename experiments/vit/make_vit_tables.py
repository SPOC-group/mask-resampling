#!/usr/bin/env python3
"""Rebuild the two ViT control tables from anonymous final-probe records.

Standard library only. Accuracies are percentages and use the final probe
checkpoint. The adjacent probe_history.csv verifies complete probe histories.
"""
import argparse
import csv
import math
from pathlib import Path

CONDITIONS = {
    'main_dynamic_standard50': ('main', 'dynamic', 'standard50'),
    'main_static_standard50': ('main', 'static_per_image', 'standard50'),
    'umae200_dynamic_standard50': ('umae200', 'dynamic', 'standard50'),
    'umae200_dynamic_standard90': ('umae200', 'dynamic', 'standard90'),
    'mae641_dynamic_standard50': ('mae641', 'dynamic', 'standard50'),
    'mae641_dynamic_standard90': ('mae641', 'dynamic', 'standard90'),
    'main_dynamic_no_geom50': ('main', 'dynamic', 'no_geom50'),
    'main_static_no_geom50': ('main', 'static_per_image', 'no_geom50'),
}
SCORES = ('validation_top1_pct', 'validation_top5_pct', 'validation_loss')


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_verified(results_path):
    records = read_csv(results_path)
    rows = {r['run_id']: r for r in records}
    require(len(records) == len(rows) and set(rows) == set(CONDITIONS),
            'Expected exactly eight distinct ViT probe conditions.')
    history = read_csv(results_path.with_name('probe_history.csv'))
    require({h['run_id'] for h in history} == set(rows), 'History conditions do not match.')
    for rid, (recipe, mode, profile) in CONDITIONS.items():
        row = rows[rid]
        for key, value in dict(recipe=recipe, mode=mode, probe_profile=profile,
                               pretraining_augmentation='no_geom').items():
            require(row[key] == value, f'{rid}: unexpected {key}.')
        for key in ('init_seed', 'pretraining_seed', 'probe_seed', 'uniformity_coefficient'):
            require(float(row[key]) == 0, f'{rid}: unexpected {key}.')
        require(float(row['rho']) == .75 and int(row['n_validation_images']) == 5000,
                f'{rid}: unexpected masking ratio or validation count.')
        epochs, horizon, batch, accumulation = {
            'main': (100, 100, 64, 2), 'umae200': (200, 200, 128, 8),
            'mae641': (641, 1600, 64, 64)}[recipe]
        expected = dict(pretraining_completed_epochs=epochs, pretraining_schedule_epochs=horizon,
                        pretraining_batch_size=batch, pretraining_accum_iter=accumulation,
                        pretraining_effective_batch_size=batch * accumulation)
        probe_epochs, probe_batch, probe_accumulation = (90, 512, 32) if profile == 'standard90' else (50, 256, 1)
        expected.update(probe_epochs=probe_epochs, probe_batch_size=probe_batch,
                        probe_accum_iter=probe_accumulation,
                        probe_effective_batch_size=probe_batch * probe_accumulation)
        for key, value in expected.items():
            require(int(row[key]) == value, f'{rid}: unexpected {key}.')
        require(row['probe_augmentation'] == ('no_geom' if profile == 'no_geom50' else 'standard'),
                f'{rid}: unexpected probe augmentation.')
        run_history = [h for h in history if h['run_id'] == rid]
        require([int(h['epoch']) for h in run_history] == list(range(probe_epochs)),
                f'{rid}: missing, repeated or unordered probe epochs.')
        for h in run_history:
            values = [float(h[k]) for k in SCORES]
            require(all(math.isfinite(v) for v in values) and 0 <= values[0] <= values[1] <= 100
                    and values[2] >= 0, f'{rid}: invalid validation metric.')
        for key in SCORES:
            require(math.isclose(float(row[key]), float(run_history[-1][key]), rel_tol=0, abs_tol=1e-9),
                    f'{rid}: {key} is not the final-epoch value.')
    return rows


def table_rows(rows):
    budget = []
    for rid, label in [('main_dynamic_standard50', 'Main comparison'),
                       ('umae200_dynamic_standard50', 'U-MAE'),
                       ('umae200_dynamic_standard90', 'U-MAE'),
                       ('mae641_dynamic_standard50', 'MAE'),
                       ('mae641_dynamic_standard90', 'MAE')]:
        r = rows[rid]
        budget.append(dict(schedule=label, pretraining_epochs=r['pretraining_completed_epochs'],
                           pretraining_effective_batch=r['pretraining_effective_batch_size'],
                           probe_epochs=r['probe_epochs'], probe_effective_batch=r['probe_effective_batch_size'],
                           top1_pct=f"{float(r['validation_top1_pct']):.2f}",
                           top5_pct=f"{float(r['validation_top5_pct']):.2f}"))
    augmentation = []
    for mode, label in [('static', 'Static'), ('dynamic', 'Dynamic')]:
        aug, noaug = rows[f'main_{mode}_standard50'], rows[f'main_{mode}_no_geom50']
        augmentation.append(dict(masks=label,
            crop_flip_top1_pct=f"{float(aug['validation_top1_pct']):.2f}",
            crop_flip_top5_pct=f"{float(aug['validation_top5_pct']):.2f}",
            no_crop_flip_top1_pct=f"{float(noaug['validation_top1_pct']):.2f}",
            no_crop_flip_top5_pct=f"{float(noaug['validation_top5_pct']):.2f}"))
    return budget, augmentation


def markdown_table(rows):
    keys = list(rows[0])
    return '\n'.join(['| ' + ' | '.join(keys) + ' |', '| ' + ' | '.join(['---'] * len(keys)) + ' |'] +
                     ['| ' + ' | '.join(str(r[k]) for k in keys) + ' |' for r in rows])


def write_tables(rows, output):
    budget, augmentation = table_rows(rows)
    output.mkdir(parents=True, exist_ok=True)
    for name, records in [('training_budget.csv', budget), ('probe_augmentation.csv', augmentation)]:
        with (output / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    text = '# ViT control tables\n\n'
    text += ('ImageNet-100 validation accuracy (%, higher is better), from the final probe epoch. '
             'Each condition uses initialization and training/probe seed zero. All encoders use rho=0.75, '
             'no pretraining crop/flip and no uniformity penalty. Batch sizes below are effective batches. '
             'These are single-run results; no seed uncertainty is estimated.\n\n')
    text += '## Longer training\n\n' + markdown_table(budget) + '\n\n'
    text += ('Schedule names denote adaptations of U-MAE and MAE optimization recipes. '
             'Repeated encoder rows evaluate the same pretrained checkpoint. The MAE encoder is '
             'checkpoint 640 (641 completed epochs) of a 1,600-epoch learning-rate schedule; '
             'the original run continued to 647 completed epochs before stopping.\n\n')
    text += '## Probe augmentation\n\n' + markdown_table(augmentation) + '\n\n'
    text += ('Each pair reuses the same 100-epoch encoder and 50-epoch probe settings; only probe-training '
             'cropping/flipping changes. Validation preprocessing is always deterministic.\n\n'
             'The adjacent probe_results.csv retains the measured terminal metrics and settings; '
             'probe_history.csv contains all 480 probe epochs. Console-derived metrics retain their '
             'original three-decimal precision; scalar-JSON metrics retain the saved precision.\n')
    (output / 'tables.md').write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-csv', type=Path, required=True,
                        help='probe_results.csv; probe_history.csv must be in the same directory')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    rows = load_verified(args.input_csv)
    write_tables(rows, args.output_dir)
    print('Verified eight final probes and 480 epochs; wrote both ViT tables.')


if __name__ == '__main__':
    main()
