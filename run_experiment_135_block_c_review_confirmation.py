from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import run_experiment_132_block_b_review_integration as base
from run_balanced_improvement_experiment import density_knn_score_pair
from run_experiment_26_rocket import make_kernels, rocket_transform
from run_experiment_40_original_score_normalization_sweep import count_cap_threshold
from run_experiment_89_74d_with_exp84_candidate import as_float, parse_indices
from run_rank_ensemble_calibration import load_dataset_data, z_normalize


EXPERIMENT_ID = 'experiment_135_block_c_review_confirmation'
EXP134_PATH = base.DATA_DIR / 'experiment_134_block_b_review_tail_guard_results.csv'
EXP133_PATH = base.DATA_DIR / 'experiment_133_block_b_confidence_tiers_results.csv'
EXP134_CONFIG = 'review_block_ba_tail_pct_1p0_top1'
EXP133_CONFIG = 'tiered_all_validated_exp93_hard_alerts'
BLOCK_C_RATE = 0.01
base.EXPERIMENT_ID = EXPERIMENT_ID


def read_rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def load_maps():
    exp134 = {row['dataset_name']: row for row in read_rows(EXP134_PATH) if row.get('config_name') == EXP134_CONFIG}
    exp133 = {row['dataset_name']: row for row in read_rows(EXP133_PATH) if row.get('config_name') == EXP133_CONFIG}
    if len(exp134) != 1117 or set(exp134) != set(exp133):
        raise SystemExit(f'coverage mismatch exp134={len(exp134)} exp133={len(exp133)}')
    return exp134, exp133


def block_c_candidates(X_train, X_test):
    reference, seed_offset = base.fixed_reference(X_train, X_train.shape[1])
    kernels = make_kernels(X_train.shape[1], num_kernels=768)[512:768]
    train_features = rocket_transform(reference, kernels)
    test_features = rocket_transform(X_test, kernels)
    train_scores, test_scores = density_knn_score_pair(train_features, test_features, 3, 'local_gap')
    threshold, q_effective, cap_target = count_cap_threshold(train_scores, BLOCK_C_RATE)
    indices = set(np.flatnonzero(test_scores > threshold).astype(int).tolist())
    return indices, test_scores, len(reference), seed_offset, q_effective, cap_target


def run_one(args):
    dataset_name, row134, row133 = args
    _, _, y_test = load_dataset_data(dataset_name)
    hard = parse_indices(row134.get('selected_indices'))
    candidate = parse_indices(row134.get('review_candidate_indices'))
    high_count = int(as_float(row133.get('high_confidence_count')))
    standard_count = int(as_float(row133.get('standard_confidence_count')))
    all_standard = high_count == 0 and standard_count > 0
    c_confirmed = set()
    diagnostics = {
        'train_normal_count': int(as_float(row134.get('train_normal_count'))),
        'tiny_train': int(as_float(row134.get('tiny_train'))),
        'test_size': len(y_test), 'exp134_candidate_count': len(candidate),
        'high_confidence_count': high_count, 'standard_confidence_count': standard_count,
        'all_existing_hard_standard': int(all_standard), 'block_c_evaluated': 0,
    }
    if candidate:
        X_train, X_test, _ = load_dataset_data(dataset_name)
        X_train = z_normalize(X_train).astype(np.float32)
        X_test = z_normalize(X_test).astype(np.float32)
        c_indices, _, reference_count, seed_offset, q_effective, cap_target = block_c_candidates(X_train, X_test)
        c_confirmed = candidate & c_indices
        diagnostics.update({
            'block_c_evaluated': 1, 'block_c_candidate_count': len(c_indices),
            'block_c_confirmed_count': len(c_confirmed), 'reference_count': reference_count,
            'reference_seed_offset': seed_offset, 'block_c_q_effective': q_effective,
            'block_c_cap_target': cap_target,
        })
    rows = [base.make_row(dataset_name, y_test, row134, 'baseline_validated_exp93_hard_only', set(), 'control: validated Exp93 hard alert only', diagnostics)]
    rows.append(base.make_row(
        dataset_name, y_test, row134, 'review_tail1pct_block_c_confirmed', c_confirmed,
        'Exp134 Block-A/B candidate retained only when independent Block C agrees', diagnostics,
    ))
    standard_review = candidate if all_standard else set()
    rows.append(base.make_row(
        dataset_name, y_test, row134, 'research_review_tail1pct_all_standard_only', standard_review,
        'research comparison: candidate retained only when existing hard alerts have no Block-B High support', diagnostics,
    ))
    combined_review = c_confirmed if all_standard else set()
    rows.append(base.make_row(
        dataset_name, y_test, row134, 'review_tail1pct_all_standard_and_block_c', combined_review,
        'candidate requires all-Standard existing alerts and independent Block-C confirmation', diagnostics,
    ))
    return rows


def run_experiment(dataset_limit=None):
    exp134, exp133 = load_maps()
    names = sorted(exp134)
    if dataset_limit:
        names = names[:dataset_limit]
    rows, errors = [], []
    with ProcessPoolExecutor(max_workers=base.WORKERS) as executor:
        futures = {executor.submit(run_one, (name, exp134[name], exp133[name])): name for name in names}
        for done, future in enumerate(as_completed(futures), 1):
            name = futures[future]
            try:
                rows.extend(future.result())
            except Exception as exc:
                errors.append((name, repr(exc)))
                print(f'ERROR dataset={name} error={exc!r}', flush=True)
            if done % 25 == 0 or done == len(names):
                print(f'Progress: [{done:4d}/{len(names):4d}] rows={len(rows)} last={name} errors={len(errors)}', flush=True)
    if errors or len(rows) != len(names) * 4:
        raise SystemExit(f'coverage failure {len(rows)}/{len(names) * 4} errors={errors[:5]}')
    base.write_csv(base.results_path(EXPERIMENT_ID), rows)
    base.write_csv(base.summary_path(EXPERIMENT_ID), base.summarize(rows))
    print(f'{EXPERIMENT_ID} finished rows={len(rows)} datasets={len(names)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-limit', type=int)
    run_experiment(parser.parse_args().dataset_limit)
