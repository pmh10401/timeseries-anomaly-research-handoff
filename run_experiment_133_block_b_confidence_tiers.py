from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from run_experiment_60_62_rocket_imaging_selector_variants import evaluate_indices, results_path, summary_path
from run_experiment_89_74d_with_exp84_candidate import as_float, parse_indices
from run_rank_ensemble_calibration import load_dataset_data


DATA_DIR = Path('/Users/minho/Documents/Dataset')
EXPERIMENT_ID = 'experiment_133_block_b_confidence_tiers'
EXP93_PATH = DATA_DIR / 'experiment_119a_exp93_rank_order_validation_results.csv'
BLOCK_B_PATH = DATA_DIR / 'experiment_131_rocket_block_b_calibration_results.csv'
BLOCK_B_CONFIG = 'block_b_extra256_knn3_countcap_pct_1p5'


def read_rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def load_maps():
    hard = {row['dataset_name']: row for row in read_rows(EXP93_PATH) if row.get('selector_name') == 'exp93_rank_order_validated'}
    block_b = {row['dataset_name']: row for row in read_rows(BLOCK_B_PATH) if row.get('config_name') == BLOCK_B_CONFIG}
    if len(hard) != 1117 or set(hard) != set(block_b):
        raise SystemExit(f'coverage mismatch hard={len(hard)} block_b={len(block_b)}')
    return hard, block_b


def metrics(y_test, selected):
    scores = np.arange(len(y_test), dtype=float)
    return evaluate_indices(y_test, scores, selected)


def make_row(dataset_name, y_test, hard_row, block_row, config_name, selected, high, standard, reason):
    values = metrics(y_test, selected)
    high_values = metrics(y_test, high)
    standard_values = metrics(y_test, standard)
    return {
        **hard_row,
        'experiment_id': EXPERIMENT_ID, 'dataset_name': dataset_name, 'family': hard_row.get('family', ''),
        'config_name': config_name, 'selector_name': config_name, 'selector_reason': reason,
        'threshold_method': 'confidence_tier', 'score_family': 'block_b_confidence_tier',
        'score_source_name': 'validated_exp93_plus_block_b_agreement',
        'selected_indices': ' '.join(str(index) for index in sorted(selected)),
        'hard_alert_indices': ' '.join(str(index) for index in sorted(high | standard)),
        'high_confidence_indices': ' '.join(str(index) for index in sorted(high)),
        'standard_confidence_indices': ' '.join(str(index) for index in sorted(standard)),
        'high_confidence_count': len(high), 'standard_confidence_count': len(standard),
        'block_b_candidate_count': len(parse_indices(block_row.get('selected_indices'))),
        'block_b_train_exceed_rate': as_float(block_row.get('train_exceed_rate')),
        'train_normal_count': int(as_float(hard_row.get('train_normal_count'))), 'test_size': len(y_test), 'anomaly_count': int(np.sum(y_test)),
        'high_tp': high_values['tp'], 'high_fp': high_values['fp'],
        'standard_tp': standard_values['tp'], 'standard_fp': standard_values['fp'],
        'uses_test_labels_for_tiering': 0,
        **values,
    }


def run_one(args):
    dataset_name, hard_row, block_row = args
    _, _, y_test = load_dataset_data(dataset_name)
    hard = parse_indices(hard_row.get('selected_indices'))
    block_b = parse_indices(block_row.get('selected_indices'))
    high, standard = hard & block_b, hard - block_b
    return [
        make_row(dataset_name, y_test, hard_row, block_row, 'tiered_all_validated_exp93_hard_alerts', hard, high, standard, 'hard alerts unchanged; Block-B index agreement supplies confidence tier'),
        make_row(dataset_name, y_test, hard_row, block_row, 'high_confidence_block_b_supported', high, high, standard, 'subset only: Exp93 hard alert index is independently supported by Block B'),
        make_row(dataset_name, y_test, hard_row, block_row, 'standard_confidence_block_b_unsupported', standard, high, standard, 'subset only: Exp93 hard alert index is not supported by Block B; retained, not suppressed'),
    ]


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
        total_tp, total_fp = int(np.sum(values('tp'))), int(np.sum(values('fp')))
        output.append({
            'experiment_id': EXPERIMENT_ID, 'config_name': config, 'selector_name': config,
            'threshold_method': 'confidence_tier', 'num_datasets': len(subset),
            'mean_f1': float(np.mean(values('f1'))), 'median_f1': float(np.median(values('f1'))),
            'zero_f1_count': int(np.sum(values('f1') == 0.0)), 'mean_fp': float(np.mean(values('fp'))),
            'mean_tp': float(np.mean(values('tp'))), 'mean_fn': float(np.mean(values('fn'))),
            'total_tp': total_tp, 'total_fp': total_fp,
            'alert_precision': total_tp / max(1, total_tp + total_fp),
            'high_confidence_alerts': int(np.sum(values('high_confidence_count'))),
            'standard_confidence_alerts': int(np.sum(values('standard_confidence_count'))),
            'high_confidence_datasets': int(np.sum(values('high_confidence_count') > 0)),
            'uses_test_labels_for_tiering_rows': int(np.sum(values('uses_test_labels_for_tiering'))),
        })
    return output


def run_experiment(dataset_limit=None):
    hard, block_b = load_maps()
    names = sorted(hard)
    if dataset_limit:
        names = names[:dataset_limit]
    rows = []
    for done, name in enumerate(names, 1):
        rows.extend(run_one((name, hard[name], block_b[name])))
        if done % 25 == 0 or done == len(names):
            print(f'Progress: [{done:4d}/{len(names):4d}] rows={len(rows)} last={name} errors=0', flush=True)
    write_csv(results_path(EXPERIMENT_ID), rows)
    write_csv(summary_path(EXPERIMENT_ID), summarize(rows))
    print(f'{EXPERIMENT_ID} finished rows={len(rows)} datasets={len(names)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-limit', type=int)
    run_experiment(parser.parse_args().dataset_limit)
