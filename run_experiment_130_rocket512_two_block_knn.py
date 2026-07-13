from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from run_balanced_improvement_experiment import density_knn_score_pair, rank_normalize_scores
from run_experiment_26_rocket import MAX_TRAIN_REFERENCE, RNG_SEED, load_dataset_names, make_kernels, rocket_transform
from run_experiment_29_train_normal_threshold_calibration import train_false_positive_stats
from run_experiment_40_original_score_normalization_sweep import count_cap_threshold, score_metrics
from run_experiment_60_62_rocket_imaging_selector_variants import evaluate_indices, results_path, summary_path
from run_rank_ensemble_calibration import load_dataset_data, z_normalize


EXPERIMENT_ID = 'experiment_130_rocket512_two_block_knn'
DATA_DIR = Path('/Users/minho/Documents/Dataset')
WORKERS = int(os.environ.get('RANK_EXPERIMENT_WORKERS', '6'))
REFERENCE_SEED_OFFSET = 130
RATE = 0.015


def fixed_reference(X_train, sequence_length):
    if len(X_train) <= MAX_TRAIN_REFERENCE:
        return X_train, 0
    rng = np.random.default_rng(RNG_SEED + REFERENCE_SEED_OFFSET + sequence_length + len(X_train))
    indices = rng.choice(len(X_train), size=MAX_TRAIN_REFERENCE, replace=False)
    return X_train[indices], REFERENCE_SEED_OFFSET


def build_scores(X_train, X_test, sequence_length):
    reference, reference_seed_offset = fixed_reference(X_train, sequence_length)
    kernels = make_kernels(sequence_length, num_kernels=512)
    train_features = rocket_transform(reference, kernels)
    test_features = rocket_transform(X_test, kernels)
    block_a_train, block_a_test = density_knn_score_pair(train_features[:, :512], test_features[:, :512], 3, 'local_gap')
    block_b_train, block_b_test = density_knn_score_pair(train_features[:, 512:], test_features[:, 512:], 3, 'local_gap')
    full_train, full_test = density_knn_score_pair(train_features, test_features, 3, 'local_gap')
    rank_a_train, rank_a_test = rank_normalize_scores(block_a_train, block_a_test)
    rank_b_train, rank_b_test = rank_normalize_scores(block_b_train, block_b_test)
    return {
        'block_a_256': (block_a_train, block_a_test),
        'block_b_extra256': (block_b_train, block_b_test),
        'full_512_shared_reference': (full_train, full_test),
        'two_block_rank_mean': ((rank_a_train + rank_b_train) / 2.0, (rank_a_test + rank_b_test) / 2.0),
    }, {'reference_count': len(reference), 'reference_seed_offset': reference_seed_offset}


def metrics_row(dataset_name, y_test, name, train_scores, test_scores, diagnostics):
    threshold, q_effective, cap_target = count_cap_threshold(train_scores, RATE)
    indices = set(np.flatnonzero(np.asarray(test_scores) > threshold).astype(int).tolist())
    metrics = evaluate_indices(y_test, test_scores, indices)
    train_exceed_count, train_exceed_rate = train_false_positive_stats(train_scores, threshold)
    return {
        'experiment_id': EXPERIMENT_ID,
        'dataset_name': dataset_name,
        'config_name': name,
        'selector_name': name,
        'selector_reason': 'train-normal count-cap threshold; no test labels used in selection',
        'threshold_method': 'count_cap_15pct',
        'score_family': 'rocket512_two_block_knn',
        'score_source_name': name,
        'feature_dimensions': 512 if name in {'block_a_256', 'block_b_extra256'} else 1024,
        'num_kernels_total': 512,
        'block_a_kernels': '0-255',
        'block_b_kernels': '256-511',
        'shared_reference': 1,
        'reference_count': diagnostics['reference_count'],
        'reference_seed_offset': diagnostics['reference_seed_offset'],
        'test_size': len(y_test),
        'anomaly_count': int(np.sum(y_test)),
        'threshold': threshold,
        'q_effective': q_effective,
        'cap_target': cap_target,
        'train_exceed_count': train_exceed_count,
        'train_exceed_rate': train_exceed_rate,
        'selected_indices': ' '.join(str(index) for index in sorted(indices)),
        **metrics,
    }


def agreement_row(dataset_name, y_test, scores, diagnostics):
    a_train, a_test = scores['block_a_256']
    b_train, b_test = scores['block_b_extra256']
    a_threshold, q_effective, cap_target = count_cap_threshold(a_train, RATE)
    b_threshold, _, _ = count_cap_threshold(b_train, RATE)
    a_indices = set(np.flatnonzero(np.asarray(a_test) > a_threshold).astype(int).tolist())
    b_indices = set(np.flatnonzero(np.asarray(b_test) > b_threshold).astype(int).tolist())
    indices = a_indices & b_indices
    rank_a_train, rank_a_test = rank_normalize_scores(a_train, a_test)
    rank_b_train, rank_b_test = rank_normalize_scores(b_train, b_test)
    fused_test = (rank_a_test + rank_b_test) / 2.0
    metrics = evaluate_indices(y_test, fused_test, indices)
    train_exceed_count = int(np.sum((a_train > a_threshold) & (b_train > b_threshold)))
    return {
        'experiment_id': EXPERIMENT_ID,
        'dataset_name': dataset_name,
        'config_name': 'two_block_threshold_intersection',
        'selector_name': 'two_block_threshold_intersection',
        'selector_reason': 'alert only when both independent 256-kernel blocks exceed their train-normal thresholds',
        'threshold_method': 'two_block_count_cap_15pct_intersection',
        'score_family': 'rocket512_two_block_knn',
        'score_source_name': 'block_a_and_block_b',
        'feature_dimensions': 1024,
        'num_kernels_total': 512,
        'block_a_kernels': '0-255',
        'block_b_kernels': '256-511',
        'shared_reference': 1,
        'reference_count': diagnostics['reference_count'],
        'reference_seed_offset': diagnostics['reference_seed_offset'],
        'test_size': len(y_test),
        'anomaly_count': int(np.sum(y_test)),
        'threshold': '',
        'q_effective': q_effective,
        'cap_target': cap_target,
        'train_exceed_count': train_exceed_count,
        'train_exceed_rate': train_exceed_count / max(1, len(a_train)),
        'block_a_predicted_count': len(a_indices),
        'block_b_predicted_count': len(b_indices),
        'block_prediction_overlap': len(indices),
        'selected_indices': ' '.join(str(index) for index in sorted(indices)),
        **metrics,
    }


def run_one(dataset_name):
    X_train, X_test, y_test = load_dataset_data(dataset_name)
    X_train = z_normalize(X_train).astype(np.float32)
    X_test = z_normalize(X_test).astype(np.float32)
    scores, diagnostics = build_scores(X_train, X_test, X_train.shape[1])
    rows = [metrics_row(dataset_name, y_test, name, *pair, diagnostics) for name, pair in scores.items()]
    rows.append(agreement_row(dataset_name, y_test, scores, diagnostics))
    return rows


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
    if errors or len(rows) != len(names) * 5:
        raise SystemExit(f'coverage failure {len(rows)}/{len(names) * 5} errors={errors[:5]}')
    write_csv(results_path(EXPERIMENT_ID), rows)
    write_csv(summary_path(EXPERIMENT_ID), summarize(rows))
    print(f'{EXPERIMENT_ID} finished rows={len(rows)} datasets={len(names)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-limit', type=int)
    run_experiment(parser.parse_args().dataset_limit)
