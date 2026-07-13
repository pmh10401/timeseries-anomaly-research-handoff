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
from run_experiment_89_74d_with_exp84_candidate import as_float, parse_indices
from run_rank_ensemble_calibration import load_dataset_data, z_normalize


DATA_DIR = Path('/Users/minho/Documents/Dataset')
EXPERIMENT_ID = 'experiment_132_block_b_review_integration'
VALIDATED_EXP93_PATH = DATA_DIR / 'experiment_119a_exp93_rank_order_validation_results.csv'
EXP55_PATH = DATA_DIR / 'experiment_55_imaging_scaling_sweep_results.csv'
EXP55_CONFIG = 'train_global_minmax_clip_spectrogram_32_pca32_knn3'
RATE = 0.015
WORKERS = int(os.environ.get('RANK_EXPERIMENT_WORKERS', '6'))
REFERENCE_SEED_OFFSET = 130


def read_rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def load_maps():
    hard = {row['dataset_name']: row for row in read_rows(VALIDATED_EXP93_PATH) if row.get('selector_name') == 'exp93_rank_order_validated'}
    exp55 = {
        row['dataset_name']: row for row in read_rows(EXP55_PATH)
        if row.get('config_name') == EXP55_CONFIG and row.get('threshold_method') == 'count_cap_3pct'
    }
    if len(hard) != 1117 or set(hard) != set(exp55):
        raise SystemExit(f'coverage mismatch hard={len(hard)} exp55={len(exp55)}')
    return hard, exp55


def fixed_reference(X_train, sequence_length):
    if len(X_train) <= MAX_TRAIN_REFERENCE:
        return X_train, 0
    rng = np.random.default_rng(RNG_SEED + REFERENCE_SEED_OFFSET + sequence_length + len(X_train))
    indices = rng.choice(len(X_train), size=MAX_TRAIN_REFERENCE, replace=False)
    return X_train[indices], REFERENCE_SEED_OFFSET


def block_scores(X_train, X_test):
    reference, seed_offset = fixed_reference(X_train, X_train.shape[1])
    kernels = make_kernels(X_train.shape[1], num_kernels=512)
    train_features = rocket_transform(reference, kernels)
    test_features = rocket_transform(X_test, kernels)
    a_train, a_test = density_knn_score_pair(train_features[:, :512], test_features[:, :512], 3, 'local_gap')
    b_train, b_test = density_knn_score_pair(train_features[:, 512:], test_features[:, 512:], 3, 'local_gap')
    return a_train, a_test, b_train, b_test, len(reference), seed_offset


def review_metrics(y_test, hard_indices, review_indices):
    hard, review = set(hard_indices), set(review_indices)
    combined = hard | review
    truth = {idx for idx, value in enumerate(y_test) if int(value) == 1}
    review_only = review - hard
    review_tp = len(review_only & truth)
    review_fp = len(review_only - truth)
    combined_tp = len(combined & truth)
    combined_fp = len(combined - truth)
    combined_fn = len(truth - combined)
    denominator = 2 * combined_tp + combined_fp + combined_fn
    combined_f1 = 2 * combined_tp / denominator if denominator else 0.0
    return {
        'review_candidate_count': len(review_only), 'review_tp': review_tp, 'review_fp': review_fp,
        'combined_tp': combined_tp, 'combined_fp': combined_fp, 'combined_fn': combined_fn,
        'combined_f1': combined_f1, 'review_hit': int(review_tp > 0), 'combined_zero_f1': int(combined_f1 == 0.0),
    }


def make_row(dataset_name, y_test, hard_row, config_name, review, reason, diagnostics):
    hard = parse_indices(hard_row.get('selected_indices'))
    row = dict(hard_row)
    row.update({
        'experiment_id': EXPERIMENT_ID, 'dataset_name': dataset_name, 'config_name': config_name,
        'selector_name': config_name, 'selector_reason': reason, 'threshold_method': 'review_lane',
        'score_family': 'block_b_review_integration', 'score_source_name': 'rocket_block_b_extra256',
        'selected_indices': ' '.join(str(index) for index in sorted(hard)),
        'review_candidate_indices': ' '.join(str(index) for index in sorted(review)),
        'uses_rocket256_current_exp40': 0, 'shared_reference': 1,
        **diagnostics, **review_metrics(y_test, hard, review),
    })
    return row


def run_one(args):
    dataset_name, hard_row, exp55_row = args
    X_train, X_test, y_test = load_dataset_data(dataset_name)
    X_train = z_normalize(X_train).astype(np.float32)
    X_test = z_normalize(X_test).astype(np.float32)
    a_train, a_test, b_train, b_test, reference_count, seed_offset = block_scores(X_train, X_test)
    a_threshold, _, _ = count_cap_threshold(a_train, RATE)
    b_threshold, b_q, b_cap = count_cap_threshold(b_train, RATE)
    a_indices = set(np.flatnonzero(a_test > a_threshold).astype(int).tolist())
    b_indices = set(np.flatnonzero(b_test > b_threshold).astype(int).tolist())
    exp55_indices = parse_indices(exp55_row.get('selected_indices'))
    hard = parse_indices(hard_row.get('selected_indices'))
    tiny_train = len(X_train) <= 10
    base_diagnostics = {
        'train_normal_count': len(X_train), 'tiny_train': int(tiny_train), 'test_size': len(y_test),
        'reference_count': reference_count, 'reference_seed_offset': seed_offset,
        'block_a_candidate_count': len(a_indices), 'block_b_candidate_count': len(b_indices),
        'exp55_candidate_count': len(exp55_indices), 'block_b_train_exceed_rate': train_false_positive_stats(b_train, b_threshold)[1],
        'block_b_q_effective': b_q, 'block_b_cap_target': b_cap,
    }
    rows = [make_row(dataset_name, y_test, hard_row, 'baseline_validated_exp93_hard_only', set(), 'control: validated Exp93 hard alert only', base_diagnostics)]
    candidates = b_indices & (a_indices | exp55_indices) - hard
    review = set()
    reason = 'no Block-B candidate passed independent index agreement'
    if not tiny_train and candidates:
        best = max(candidates, key=lambda idx: (float(b_test[idx]), -idx))
        review = {best}
        sources = []
        if best in a_indices:
            sources.append('block_a')
        if best in exp55_indices:
            sources.append('exp55')
        reason = 'Block-B top score agrees with ' + '+'.join(sources)
    rows.append(make_row(dataset_name, y_test, hard_row, 'review_block_b_agree_block_a_or_exp55_top1', review, reason, {**base_diagnostics, 'agreement_sources': '+'.join([name for name, indices in [('block_a', a_indices), ('exp55', exp55_indices)] if review & indices])}))
    strict_candidates = b_indices & a_indices & exp55_indices - hard
    strict_review = set()
    strict_reason = 'no Block-B candidate agreed with both Block-A and Exp55'
    if not tiny_train and strict_candidates:
        strict_review = {max(strict_candidates, key=lambda idx: (float(b_test[idx]), -idx))}
        strict_reason = 'Block-B top score agrees with both Block-A and Exp55'
    rows.append(make_row(dataset_name, y_test, hard_row, 'review_block_b_agree_block_a_and_exp55_top1', strict_review, strict_reason, base_diagnostics))
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
        values = lambda key: np.asarray([as_float(row.get(key)) for row in subset], dtype=float)
        output.append({
            'experiment_id': EXPERIMENT_ID, 'config_name': config, 'selector_name': config,
            'threshold_method': 'review_lane', 'num_datasets': len(subset),
            'mean_hard_f1': float(np.mean(values('f1'))), 'hard_zero_f1_count': int(np.sum(values('f1') == 0.0)),
            'mean_hard_fp': float(np.mean(values('fp'))), 'mean_combined_f1': float(np.mean(values('combined_f1'))),
            'combined_zero_f1_count': int(np.sum(values('combined_zero_f1'))),
            'review_candidate_datasets': int(np.sum(values('review_candidate_count') > 0)),
            'review_hit_datasets': int(np.sum(values('review_hit') > 0)),
            'mean_review_tp': float(np.mean(values('review_tp'))), 'mean_review_fp': float(np.mean(values('review_fp'))),
            'mean_combined_fp': float(np.mean(values('combined_fp'))),
            'tiny_train_datasets': int(np.sum(values('tiny_train') > 0)),
        })
    return sorted(output, key=lambda row: (row['mean_combined_f1'], -row['mean_review_fp']), reverse=True)


def run_experiment(dataset_limit=None):
    hard, exp55 = load_maps()
    names = sorted(load_dataset_names())
    if dataset_limit:
        names = names[:dataset_limit]
    rows, errors = [], []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(run_one, (name, hard[name], exp55[name])): name for name in names}
        for done, future in enumerate(as_completed(futures), 1):
            name = futures[future]
            try:
                rows.extend(future.result())
            except Exception as exc:
                errors.append((name, repr(exc)))
                print(f'ERROR dataset={name} error={exc!r}', flush=True)
            if done % 25 == 0 or done == len(names):
                print(f'Progress: [{done:4d}/{len(names):4d}] rows={len(rows)} last={name} errors={len(errors)}', flush=True)
    if errors or len(rows) != len(names) * 3:
        raise SystemExit(f'coverage failure {len(rows)}/{len(names) * 3} errors={errors[:5]}')
    write_csv(results_path(EXPERIMENT_ID), rows)
    write_csv(summary_path(EXPERIMENT_ID), summarize(rows))
    print(f'{EXPERIMENT_ID} finished rows={len(rows)} datasets={len(names)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-limit', type=int)
    run_experiment(parser.parse_args().dataset_limit)
