from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from run_balanced_improvement_experiment import density_knn_score_pair
from run_experiment_26_rocket import MAX_TRAIN_REFERENCE, RNG_SEED, load_dataset_names, make_kernels, rocket_transform
from run_experiment_29_train_normal_threshold_calibration import train_false_positive_stats
from run_experiment_40_original_score_normalization_sweep import count_cap_threshold
from run_experiment_60_62_rocket_imaging_selector_variants import evaluate_indices, results_path, summary_path
from run_rank_ensemble_calibration import load_dataset_data, z_normalize


EXPERIMENT_ID = 'experiment_131_rocket_block_b_calibration'
WORKERS = int(os.environ.get('RANK_EXPERIMENT_WORKERS', '6'))
RATES = (0.005, 0.01, 0.015)
NEIGHBORS = (3, 5, 7)
REFERENCE_SEED_OFFSET = 130


def fixed_reference(X_train, sequence_length):
    if len(X_train) <= MAX_TRAIN_REFERENCE:
        return X_train, 0
    rng = np.random.default_rng(RNG_SEED + REFERENCE_SEED_OFFSET + sequence_length + len(X_train))
    indices = rng.choice(len(X_train), size=MAX_TRAIN_REFERENCE, replace=False)
    return X_train[indices], REFERENCE_SEED_OFFSET


def score_block_b(X_train, X_test):
    reference, seed_offset = fixed_reference(X_train, X_train.shape[1])
    kernels = make_kernels(X_train.shape[1], num_kernels=512)
    train_features = rocket_transform(reference, kernels)[:, 512:]
    test_features = rocket_transform(X_test, kernels)[:, 512:]
    return reference, seed_offset, train_features, test_features


def config_name(neighbors, rate):
    label = {0.005: '0p5', 0.01: '1p0', 0.015: '1p5'}[rate]
    return f'block_b_extra256_knn{neighbors}_countcap_pct_{label}'


def threshold_name(rate):
    return {0.005: 'count_cap_pct_0p5', 0.01: 'count_cap_pct_1p0', 0.015: 'count_cap_pct_1p5'}[rate]


def run_one(dataset_name):
    X_train, X_test, y_test = load_dataset_data(dataset_name)
    X_train = z_normalize(X_train).astype(np.float32)
    X_test = z_normalize(X_test).astype(np.float32)
    reference, seed_offset, train_features, test_features = score_block_b(X_train, X_test)
    rows = []
    for neighbors in NEIGHBORS:
        train_scores, test_scores = density_knn_score_pair(train_features, test_features, neighbors, 'local_gap')
        for rate in RATES:
            threshold, q_effective, cap_target = count_cap_threshold(train_scores, rate)
            indices = set(np.flatnonzero(np.asarray(test_scores) > threshold).astype(int).tolist())
            metrics = evaluate_indices(y_test, test_scores, indices)
            train_exceed_count, train_exceed_rate = train_false_positive_stats(train_scores, threshold)
            rows.append({
                'experiment_id': EXPERIMENT_ID,
                'dataset_name': dataset_name,
                'config_name': config_name(neighbors, rate),
                'selector_name': config_name(neighbors, rate),
                'selector_reason': 'independent second 256-kernel block; train-normal count-cap threshold',
                'threshold_method': threshold_name(rate),
                'score_family': 'rocket_block_b_local_gap',
                'score_source_name': 'rocket_block_b_extra256',
                'num_kernels_total': 512,
                'active_kernel_block': '256-511',
                'feature_dimensions': 512,
                'neighbors': neighbors,
                'shared_reference': 1,
                'reference_count': len(reference),
                'reference_seed_offset': seed_offset,
                'test_size': len(y_test),
                'anomaly_count': int(np.sum(y_test)),
                'train_normal_count': len(X_train),
                'threshold': threshold,
                'q_effective': q_effective,
                'cap_target': cap_target,
                'train_exceed_count': train_exceed_count,
                'train_exceed_rate': train_exceed_rate,
                'selected_indices': ' '.join(str(index) for index in sorted(indices)),
                **metrics,
            })
    return rows


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    output = []
    for config in sorted({row['config_name'] for row in rows}):
        subset = [row for row in rows if row['config_name'] == config]
        values = lambda key: np.asarray([float(row.get(key, 0.0) or 0.0) for row in subset], dtype=float)
        f1s = values('f1')
        output.append({
            'experiment_id': EXPERIMENT_ID,
            'config_name': config,
            'selector_name': config,
            'threshold_method': subset[0]['threshold_method'],
            'neighbors': subset[0]['neighbors'],
            'num_datasets': len(subset),
            'mean_f1': float(np.mean(f1s)),
            'median_f1': float(np.median(f1s)),
            'zero_f1_count': int(np.sum(f1s == 0.0)),
            'mean_fp': float(np.mean(values('fp'))),
            'mean_tp': float(np.mean(values('tp'))),
            'mean_fn': float(np.mean(values('fn'))),
            'mean_auc_pr': float(np.mean(values('auc_pr'))),
            'mean_oracle_f1': float(np.mean(values('oracle_f1'))),
            'mean_train_exceed_rate': float(np.mean(values('train_exceed_rate'))),
            'shared_reference_rows': int(np.sum(values('shared_reference'))),
        })
    return sorted(output, key=lambda row: (row['mean_f1'], -row['mean_fp']), reverse=True)


def run_experiment(dataset_limit=None):
    names = sorted(load_dataset_names())
    if dataset_limit:
        names = names[:dataset_limit]
    rows, errors = [], []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(run_one, name): name for name in names}
        for done, future in enumerate(as_completed(futures), 1):
            name = futures[future]
            try:
                rows.extend(future.result())
            except Exception as exc:
                errors.append((name, repr(exc)))
                print(f'ERROR dataset={name} error={exc!r}', flush=True)
            if done % 25 == 0 or done == len(names):
                print(f'Progress: [{done:4d}/{len(names):4d}] rows={len(rows)} last={name} errors={len(errors)}', flush=True)
    expected = len(names) * len(NEIGHBORS) * len(RATES)
    if errors or len(rows) != expected:
        raise SystemExit(f'coverage failure {len(rows)}/{expected} errors={errors[:5]}')
    write_csv(results_path(EXPERIMENT_ID), rows)
    write_csv(summary_path(EXPERIMENT_ID), summarize(rows))
    print(f'{EXPERIMENT_ID} finished rows={len(rows)} datasets={len(names)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-limit', type=int)
    run_experiment(parser.parse_args().dataset_limit)
