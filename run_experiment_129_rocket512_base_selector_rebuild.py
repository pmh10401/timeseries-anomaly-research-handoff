from __future__ import annotations

import argparse
import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from run_experiment_40_original_score_normalization_sweep import count_cap_threshold, score_pair_for_config
from run_experiment_60_62_rocket_imaging_selector_variants import (
    CALIBRATION_PROFILES,
    IMAGING_CONFIGS,
    evaluate_indices,
    prediction_bundle,
    results_path,
    summary_path,
)
from run_experiment_89_74d_with_exp84_candidate import EXP87_CONFIG, as_float, exp84_bundle_from_row, format_indices
from run_experiment_90_zero_f1_repair_selector import first_top_index_from_bundle
from run_model_hard_research_experiments import prepare_series_pair_for_scale, score_pair_for_config as imaging_score_pair_for_config
from run_original_improvement_experiment import DB_PATH, load_original_record, target_len_for_record
from run_rank_ensemble_calibration import align_series_lengths, load_dataset_data, z_normalize


DATA_DIR = Path('/Users/minho/Documents/Dataset')
EXPERIMENT_ID = 'experiment_129_rocket512_base_selector_rebuild'
EXP87_PATH = DATA_DIR / 'experiment_87_exp84_index_diagnostics_results.csv'
VALIDATED_EXP93_PATH = DATA_DIR / 'experiment_119a_exp93_rank_order_validation_results.csv'
WORKERS = int(os.environ.get('RANK_EXPERIMENT_WORKERS', '6'))
ROCKET_CONFIGS = {
    'rocket256_control': {'name': 'rocket_256_knn3_local_gap', 'kind': 'density_knn', 'num_kernels': 256, 'neighbors': 3, 'mode': 'local_gap'},
    'rocket512_replacement': {'name': 'rocket_512_knn3_local_gap', 'kind': 'density_knn', 'num_kernels': 512, 'neighbors': 3, 'mode': 'local_gap'},
}


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


def load_exp87_rows():
    out = {}
    with EXP87_PATH.open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row.get('config_name') == EXP87_CONFIG:
                out[(row['dataset_name'], row['threshold_method'])] = row
    if not out:
        raise SystemExit('Exp87 candidate rows unavailable')
    return out


def load_dataset_names():
    with VALIDATED_EXP93_PATH.open(newline='') as handle:
        names = {
            row['dataset_name']
            for row in csv.DictReader(handle)
            if row.get('selector_name') == 'exp93_rank_order_validated'
        }
    if len(names) != 1117:
        raise SystemExit(f'validated dataset coverage mismatch: {len(names)}/1117')
    return sorted(names)


def build_bundles(dataset_name, rocket_config):
    x_train, x_test, y_test = load_dataset_data(dataset_name)
    sequence_length = x_train.shape[1]
    rocket_train, rocket_test = score_pair_for_config(
        z_normalize(x_train).astype(np.float32),
        z_normalize(x_test).astype(np.float32),
        sequence_length,
        rocket_config,
        {},
    )
    rate = CALIBRATION_PROFILES['relaxed_15pct']['rocket_exp40']
    threshold, q_effective, cap_target = count_cap_threshold(rocket_train, rate)
    rocket = prediction_bundle(rocket_config['name'], y_test, rocket_train, rocket_test, threshold, q_effective, cap_target)

    record = load_original_record(dataset_name, DB_PATH)
    target_len = min(max(8, target_len_for_record(record, 'actual_median')), 2048)
    raw_train = align_series_lengths(record['train_series'], target_len)
    raw_test = align_series_lengths(record['test_series'], target_len)
    z_train, z_test = z_normalize(raw_train).astype(np.float32), z_normalize(raw_test).astype(np.float32)
    bundles = {'rocket': rocket}
    for name in ('exp55_best', 'exp56_best'):
        config = IMAGING_CONFIGS[name]
        scaled_train, scaled_test = prepare_series_pair_for_scale(config.get('series_scale', 'per_series_z'), raw_train, raw_test, z_train, z_test)
        train_scores, test_scores = imaging_score_pair_for_config(scaled_train, scaled_test, target_len, config, record)
        source_rate = CALIBRATION_PROFILES['relaxed_15pct'][name]
        threshold, q_effective, cap_target = count_cap_threshold(train_scores, source_rate)
        bundles[name] = prediction_bundle(name, y_test, train_scores, test_scores, threshold, q_effective, cap_target)
    return record, y_test, bundles


def budget(bundle, rate=0.02):
    return max(1, int(math.ceil(len(bundle['test_scores']) * rate)))


def passes_guard(bundle, rate=0.02, train_rate=0.025):
    return 0 < len(bundle['indices']) <= budget(bundle, rate) and as_float(bundle['train_exceed_rate']) <= train_rate


def choose_base(bundles):
    # This reproduces the existing label-free confidence policy, but substitutes only the ROCKET source.
    rocket = bundles['rocket']
    if passes_guard(rocket, rate=0.03, train_rate=0.035):
        return set(rocket['indices']), 'rocket', 'rocket_guard_pass'
    imaging = [name for name in ('exp55_best', 'exp56_best') if passes_guard(bundles[name])]
    if imaging:
        source = min(imaging, key=lambda name: (as_float(bundles[name]['train_exceed_rate']), len(bundles[name]['indices'])))
        return set(bundles[source]['indices']), source, 'rocket_weak_imaging_guard_pass'
    if rocket['indices'] and len(rocket['indices']) / max(1, len(rocket['test_scores'])) <= 0.05:
        return set(rocket['indices']), 'rocket', 'rocket_nonzero_fallback'
    return set(), 'none', 'no_candidate_passed_confidence'


def apply_noalert_repair(base_indices, bundles, exp87_rows, dataset_name, tiny_train):
    if base_indices or tiny_train:
        return set(base_indices), 'none'
    fg = exp84_bundle_from_row(exp87_rows.get((dataset_name, 'family_guard_v1')), len(bundles['rocket']['test_scores']))
    exp84_safe = fg['train_exceed_rate'] <= 0.015 and fg['top1_threshold_margin'] >= 0.0 and fg['top1_top2_margin'] >= 0.0
    if exp84_safe and fg.get('top1') is not None:
        return {fg['top1']}, 'exp84'
    if bundles['rocket']['train_exceed_rate'] <= 0.02:
        top = first_top_index_from_bundle(bundles['rocket'])
        if top is not None:
            return {top}, 'rocket'
    return set(), 'none'


def make_row(dataset_name, record, y_test, bundles, variant, config_name, indices, selected_source, reason, repair_source, tiny_train):
    rocket = bundles['rocket']
    metrics = evaluate_indices(y_test, rocket['test_scores'], indices)
    return {
        'experiment_id': EXPERIMENT_ID,
        'dataset_name': dataset_name,
        'family': record['family'],
        'config_name': config_name,
        'selector_name': config_name,
        'selector_reason': reason,
        'threshold_method': 'selector',
        'score_family': 'clean_rocket_base_selector_rebuild',
        'score_source_name': rocket['name'],
        'rocket_variant': variant,
        'rocket_num_kernels': 256 if variant == 'rocket256_control' else 512,
        'uses_rocket256_in_selection': int(variant == 'rocket256_control'),
        'uses_rocket512_in_selection': int(variant == 'rocket512_replacement'),
        'selected_source': selected_source,
        'repair_source': repair_source,
        'sequence_length': len(record['test_series'][0]) if record['test_series'] else '',
        'test_size': len(y_test),
        'anomaly_count': int(np.sum(y_test)),
        'selected_indices': format_indices(indices),
        'predicted_count': metrics['predicted_count'],
        'tp': metrics['tp'], 'fp': metrics['fp'], 'fn': metrics['fn'],
        'f1': metrics['f1'], 'auc_roc': metrics['auc_roc'], 'auc_pr': metrics['auc_pr'], 'oracle_f1': metrics['oracle_f1'],
        'train_exceed_rate': rocket['train_exceed_rate'],
        'rocket_predicted_count': len(rocket['indices']),
        'exp55_predicted_count': len(bundles['exp55_best']['indices']),
        'exp56_predicted_count': len(bundles['exp56_best']['indices']),
        'train_normal_count': len(record['train_series']),
        'tiny_train': int(tiny_train),
    }


def run_one(args):
    dataset_name, exp87_rows = args
    rows = []
    for variant, config in ROCKET_CONFIGS.items():
        record, y_test, bundles = build_bundles(dataset_name, config)
        tiny_train = len(record['train_series']) <= 10
        base, source, reason = choose_base(bundles)
        rows.append(make_row(dataset_name, record, y_test, bundles, variant, f'{variant}_base_label_free', base, source, reason, 'none', tiny_train))
        repaired, repair_source = apply_noalert_repair(base, bundles, exp87_rows, dataset_name, tiny_train)
        rows.append(make_row(dataset_name, record, y_test, bundles, variant, f'{variant}_noalert_train_safe_repair', repaired, source if base else repair_source, f'{reason}; no-alert repair={repair_source}', repair_source, tiny_train))
    return rows


def summarize(rows):
    summary = []
    for config_name in sorted({row['config_name'] for row in rows}):
        subset = [row for row in rows if row['config_name'] == config_name]
        values = lambda key: [as_float(row[key]) for row in subset]
        f1s = values('f1')
        summary.append({
            'experiment_id': EXPERIMENT_ID, 'config_name': config_name, 'selector_name': config_name,
            'threshold_method': 'selector', 'num_datasets': len(subset),
            'mean_f1': float(np.mean(f1s)), 'median_f1': float(np.median(f1s)), 'zero_f1_count': sum(x == 0.0 for x in f1s),
            'mean_fp': float(np.mean(values('fp'))), 'mean_tp': float(np.mean(values('tp'))), 'mean_fn': float(np.mean(values('fn'))),
            'mean_auc_pr': float(np.mean(values('auc_pr'))), 'mean_oracle_f1': float(np.mean(values('oracle_f1'))),
            'rocket_selected_datasets': sum(row['selected_source'] == 'rocket' for row in subset),
            'exp55_selected_datasets': sum(row['selected_source'] == 'exp55_best' for row in subset),
            'exp56_selected_datasets': sum(row['selected_source'] == 'exp56_best' for row in subset),
            'noalert_rocket_repairs': sum(row['repair_source'] == 'rocket' for row in subset),
            'noalert_exp84_repairs': sum(row['repair_source'] == 'exp84' for row in subset),
            'rocket256_selection_rows': sum(row['uses_rocket256_in_selection'] for row in subset),
            'rocket512_selection_rows': sum(row['uses_rocket512_in_selection'] for row in subset),
        })
    return sorted(summary, key=lambda row: (row['mean_f1'], -row['mean_fp']), reverse=True)


def run_experiment(dataset_limit=None):
    exp87 = load_exp87_rows()
    names = load_dataset_names()
    if dataset_limit:
        names = names[:dataset_limit]
    rows, errors = [], []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(run_one, (name, exp87)): name for name in names}
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
    write_csv(results_path(EXPERIMENT_ID), rows)
    write_csv(summary_path(EXPERIMENT_ID), summarize(rows))
    print(f'{EXPERIMENT_ID} finished rows={len(rows)} datasets={len(names)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-limit', type=int)
    run_experiment(parser.parse_args().dataset_limit)
