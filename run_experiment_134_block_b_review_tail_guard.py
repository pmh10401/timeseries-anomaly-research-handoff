from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import run_experiment_132_block_b_review_integration as base
from run_experiment_40_original_score_normalization_sweep import count_cap_threshold
from run_experiment_89_74d_with_exp84_candidate import parse_indices
from run_rank_ensemble_calibration import load_dataset_data, z_normalize


EXPERIMENT_ID = 'experiment_134_block_b_review_tail_guard'
RATES = (0.005, 0.01, 0.015)
base.EXPERIMENT_ID = EXPERIMENT_ID


def run_one(args):
    dataset_name, hard_row, _exp55_row = args
    X_train, X_test, y_test = load_dataset_data(dataset_name)
    X_train = z_normalize(X_train).astype(np.float32)
    X_test = z_normalize(X_test).astype(np.float32)
    a_train, a_test, b_train, b_test, reference_count, seed_offset = base.block_scores(X_train, X_test)
    a_threshold, _, _ = count_cap_threshold(a_train, 0.015)
    a_indices = set(np.flatnonzero(a_test > a_threshold).astype(int).tolist())
    hard = parse_indices(hard_row.get('selected_indices'))
    tiny_train = len(X_train) <= 10
    diagnostics = {
        'train_normal_count': len(X_train), 'tiny_train': int(tiny_train), 'test_size': len(y_test),
        'reference_count': reference_count, 'reference_seed_offset': seed_offset,
        'block_a_candidate_count': len(a_indices), 'uses_rocket256_current_exp40': 0,
    }
    rows = [base.make_row(dataset_name, y_test, hard_row, 'baseline_validated_exp93_hard_only', set(), 'control: validated Exp93 hard alert only', diagnostics)]
    for rate in RATES:
        b_threshold, b_q, b_cap = count_cap_threshold(b_train, rate)
        b_indices = set(np.flatnonzero(b_test > b_threshold).astype(int).tolist())
        candidates = (a_indices & b_indices) - hard
        review = set()
        reason = f'no Block-B/A agreement passed train-normal tail {rate:.1%}'
        if not tiny_train and candidates:
            review = {max(candidates, key=lambda index: (float(b_test[index]), -index))}
            reason = f'Block-B and Block-A agree; Block-B exceeds train-normal tail {rate:.1%}'
        label = {0.005: '0p5', 0.01: '1p0', 0.015: '1p5'}[rate]
        rows.append(base.make_row(
            dataset_name, y_test, hard_row, f'review_block_ba_tail_pct_{label}_top1', review, reason,
            {**diagnostics, 'block_b_tail_rate': rate, 'block_b_q_effective': b_q, 'block_b_cap_target': b_cap, 'block_b_candidate_count': len(b_indices)},
        ))
    return rows


def run_experiment(dataset_limit=None):
    hard, exp55 = base.load_maps()
    names = sorted(hard)
    if dataset_limit:
        names = names[:dataset_limit]
    rows, errors = [], []
    with ProcessPoolExecutor(max_workers=base.WORKERS) as executor:
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
    if errors or len(rows) != len(names) * 4:
        raise SystemExit(f'coverage failure {len(rows)}/{len(names) * 4} errors={errors[:5]}')
    base.write_csv(base.results_path(EXPERIMENT_ID), rows)
    base.write_csv(base.summary_path(EXPERIMENT_ID), base.summarize(rows))
    print(f'{EXPERIMENT_ID} finished rows={len(rows)} datasets={len(names)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-limit', type=int)
    run_experiment(parser.parse_args().dataset_limit)
