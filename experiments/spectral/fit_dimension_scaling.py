#!/usr/bin/env python3
"""Fit regenerated PCA results; save tables only, without rerunning simulations."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import gammaln

def random_direction(dimensions, metric):
    d = np.asarray(dimensions, dtype=float)
    if metric == "cosine_sq":
        return 1 / d
    if metric == "cosine_abs":
        # Exact expectation for a uniform unit vector, not RMS=1/sqrt(d).
        return np.exp(gammaln(d/2) - gammaln((d+1)/2)) / np.sqrt(np.pi)
    raise ValueError(f"Unknown metric: {metric}")


def fit_power_law(dimensions, means):
    """OLS in log space. Means may have arbitrary leading batch dimensions."""
    x = np.log(np.asarray(dimensions, dtype=float))
    y = np.log(np.asarray(means, dtype=float))
    if len(x) < 3 or not np.isfinite(y).all():
        raise ValueError("Need at least three dimensions and positive finite means")
    centered_x = x - x.mean()
    slope = np.sum((y - y.mean(axis=-1, keepdims=True)) * centered_x, axis=-1)
    slope /= np.sum(centered_x**2)
    intercept = y.mean(axis=-1) - slope * x.mean()
    prediction = intercept[..., None] + slope[..., None] * x
    residual_ss = np.sum((y - prediction)**2, axis=-1)
    total_ss = np.sum((y - y.mean(axis=-1, keepdims=True))**2, axis=-1)
    r_squared = np.divide(residual_ss, total_ss, out=np.zeros_like(residual_ss),
                          where=total_ss > 0)
    r_squared = np.where(total_ss > 0, 1 - r_squared, np.nan)
    return np.exp(intercept), -slope, r_squared


def fit_fixed_slope(dimensions, means, power=.5):
    """Fit log(A) only in log(mean) = log(A) - power*log(d)."""
    x = np.log(np.asarray(dimensions, dtype=float))
    y = np.log(np.asarray(means, dtype=float))
    if len(x) < 3 or not np.isfinite(y).all():
        raise ValueError("Need at least three dimensions and positive finite means")
    log_a = np.mean(y + power*x, axis=-1)
    residual = y - (log_a[..., None] - power*x)
    residual_ss = np.sum(residual**2, axis=-1)
    total_ss = np.sum((y - y.mean(axis=-1, keepdims=True))**2, axis=-1)
    ratio = np.divide(residual_ss, total_ss, out=np.zeros_like(residual_ss),
                      where=total_ss > 0)
    score = np.where(total_ss > 0, 1-ratio, np.nan)
    return np.exp(log_a), score, np.sqrt(np.mean(residual**2, axis=-1))


def compute_fixed_slope_fits(dimensions, alphas, profiles, tail_min, bootstrap_reps,
                              rng_seed, metric="cosine_abs"):
    power = .5 if metric == "cosine_abs" else 1.
    n_seeds = profiles.shape[1]
    draws = np.random.default_rng(rng_seed).integers(
        0, n_seeds, size=(bootstrap_reps, n_seeds))
    bootstrap_means = profiles[:, draws, :].mean(axis=2)
    means = profiles.mean(axis=1)
    rows, points = [], []
    for scope, mask in (("all_dimensions", np.ones(len(dimensions), dtype=bool)),
                        ("large_dimension_tail", dimensions >= tail_min)):
        ds = dimensions[mask]
        amplitudes, scores, rmses = fit_fixed_slope(ds, means[:, mask], power)
        boot_a, _, _ = fit_fixed_slope(ds, bootstrap_means[:, :, mask], power)
        for i, alpha in enumerate(alphas):
            low, high = np.percentile(boot_a[i], [2.5, 97.5])
            rows.append(dict(
                alpha=alpha, metric=metric, fit_scope=scope, n_dimensions=len(ds),
                d_min=int(ds.min()), d_max=int(ds.max()), num_seeds=n_seeds,
                p_fixed=power, amplitude_A=amplitudes[i], amplitude_A_ci95_low=low,
                amplitude_A_ci95_high=high, r_squared_loglog=scores[i],
                rmse_loglog=rmses[i], bootstrap_reps=bootstrap_reps,
                bootstrap_seed=rng_seed,
                method=f"unweighted log-space least squares; p fixed at {power:g}; amplitude only"))
            for d, mean in zip(ds, means[i, mask]):
                fitted = amplitudes[i] * float(d)**(-power)
                points.append({
                    "alpha": alpha, "metric": metric, "fit_scope": scope, "d": int(d),
                    f"{metric}_mean": mean, f"fitted_{metric}": fitted,
                    "p_fixed": power, "log_residual": np.log(mean/fitted),
                    "relative_residual": mean/fitted-1,
                    "random_direction": float(random_direction(d, metric))})
    return pd.DataFrame(rows), pd.DataFrame(points)


def load_data(run_dir, metric="cosine_sq"):
    with (run_dir / "run_config.json").open() as handle:
        config = json.load(handle)
    raw = pd.read_csv(run_dir / "pca_per_seed.csv")
    summary = pd.read_csv(run_dir / "pca_summary.csv")
    dimensions = np.sort(np.asarray(config["dimensions"], dtype=int))
    alphas = np.sort(np.asarray(config["alphas"], dtype=float))
    seeds = np.sort(np.asarray(config["seeds"], dtype=int))
    if (raw.duplicated(["alpha", "d", "seed"]).any()
            or len(raw) != len(dimensions) * len(alphas) * len(seeds)
            or not raw.converged.eq(True).all()):
        raise ValueError("Per-seed data must be complete, unique, and converged")
    if len(summary) != len(dimensions) * len(alphas):
        raise ValueError("Unexpected number of summary rows")
    if not summary.num_seeds.eq(len(seeds)).all():
        raise ValueError("Summary seed counts do not match configuration")
    np.testing.assert_allclose(raw.cosine_abs**2, raw.cosine_sq, atol=1e-14)
    selected = alphas[-3:]
    profiles = []
    for alpha in selected:
        subset = raw[np.isclose(raw.alpha, alpha, rtol=1e-12, atol=0)]
        bank = subset.pivot(index="seed", columns="d", values=metric)
        bank = bank.reindex(index=seeds, columns=dimensions).to_numpy(dtype=float)
        if not np.isfinite(bank).all() or (bank < 0).any() or (bank > 1).any():
            raise ValueError(f"Missing or invalid {metric} observations")
        group = summary[np.isclose(summary.alpha, alpha, rtol=1e-12, atol=0)]
        group = group.sort_values("d")
        np.testing.assert_array_equal(group.d, dimensions)
        np.testing.assert_allclose(group[f"{metric}_mean"], bank.mean(axis=0), atol=1e-14)
        np.testing.assert_allclose(group[f"{metric}_std"], bank.std(axis=0, ddof=1),
                                   atol=1e-14)
        profiles.append(bank)
    return config, summary, dimensions, selected, seeds, np.asarray(profiles)


def compute_fits(dimensions, alphas, profiles, tail_min, bootstrap_reps, rng_seed,
                 metric="cosine_sq"):
    # One set of seed indices is shared by every alpha/dimension: resample the
    # complete seed profile, not observations independently at each dimension.
    n_seeds = profiles.shape[1]
    draws = np.random.default_rng(rng_seed).integers(
        0, n_seeds, size=(bootstrap_reps, n_seeds))
    bootstrap_means = profiles[:, draws, :].mean(axis=2)
    means = profiles.mean(axis=1)
    rows, predictions = [], []
    for scope, mask in (("all_dimensions", np.ones(len(dimensions), dtype=bool)),
                        ("large_dimension_tail", dimensions >= tail_min)):
        ds = dimensions[mask]
        if len(ds) < 3:
            raise ValueError("Tail fit needs at least three observed dimensions")
        prefactors, powers, scores = fit_power_law(ds, means[:, mask])
        boot_c, boot_p, _ = fit_power_law(ds, bootstrap_means[:, :, mask])
        for i, alpha in enumerate(alphas):
            c_low, c_high = np.percentile(boot_c[i], [2.5, 97.5])
            p_low, p_high = np.percentile(boot_p[i], [2.5, 97.5])
            rows.append(dict(
                alpha=alpha, metric=metric, fit_scope=scope, n_dimensions=len(ds),
                d_min=int(ds.min()), d_max=int(ds.max()), num_seeds=n_seeds,
                prefactor_C=prefactors[i], p=powers[i], r_squared_loglog=scores[i],
                prefactor_C_ci95_low=c_low, prefactor_C_ci95_high=c_high,
                p_ci95_low=p_low, p_ci95_high=p_high,
                bootstrap_reps=bootstrap_reps, bootstrap_seed=rng_seed,
                method=f"unweighted OLS of log(mean {metric}) versus log(d)"))
            for d, mean in zip(ds, means[i, mask]):
                fitted = prefactors[i] * float(d)**(-powers[i])
                predictions.append({
                    "alpha": alpha, "metric": metric, "fit_scope": scope, "d": int(d),
                    f"{metric}_mean": mean, f"fitted_{metric}": fitted,
                    "log_residual": np.log(mean/fitted),
                    "random_direction": float(random_direction(d, metric))})
    return pd.DataFrame(rows), pd.DataFrame(predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--tail-min-d', type=int, default=1600)
    parser.add_argument('--bootstrap-reps', type=int, default=2000)
    parser.add_argument('--bootstrap-seed', type=int, default=1729)
    args = parser.parse_args()
    if args.bootstrap_reps<100: parser.error('Use at least 100 bootstrap replicates')
    config, summary, dimensions, alphas, seeds, profiles = load_data(args.run_dir, 'cosine_abs')
    if np.count_nonzero(dimensions>=args.tail_min_d)<3: parser.error('Tail needs at least three dimensions')
    fixed, fixed_points = compute_fixed_slope_fits(dimensions, alphas, profiles, args.tail_min_d,
                                                   args.bootstrap_reps, args.bootstrap_seed, 'cosine_abs')
    free, free_points = compute_fits(dimensions, alphas, profiles, args.tail_min_d,
                                    args.bootstrap_reps, args.bootstrap_seed, 'cosine_abs')
    for name, frame in [('fixed_slope_fits', fixed), ('fixed_slope_fit_points', fixed_points),
                        ('powerlaw_fits', free), ('powerlaw_fit_points', free_points)]:
        frame.to_csv(args.run_dir/f'pca_cosine_dimension_{name}.csv', index=False)
    metadata = dict(metric='cosine_abs', formula='mean(cosine_abs) = A_alpha * d**(-1/2)',
                    regression='Unweighted log-space least squares', tail_min_d=args.tail_min_d,
                    bootstrap_reps=args.bootstrap_reps, bootstrap_seed=args.bootstrap_seed,
                    bootstrap='Joint resampling of complete seed profiles; percentile 95% intervals',
                    alphas=alphas.tolist(), dimensions=dimensions.tolist(), seeds=seeds.tolist(),
                    uncertainty_note='Fit intervals are bootstrap 95% CIs; summary SEM is SD/sqrt(n).')
    (args.run_dir/'pca_cosine_dimension_fit_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print('Saved fixed-slope and free-exponent fit tables; no figures or simulation data were generated.')


if __name__ == '__main__': main()
