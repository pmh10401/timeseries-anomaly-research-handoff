from __future__ import annotations

import argparse
import csv
import hashlib

import numpy as np

import run_experiment_132_block_b_review_integration as base
from run_experiment_89_74d_with_exp84_candidate import as_float


EXPERIMENT_ID = 'experiment_136_family_holdout_review_audit'
SOURCE_PATH = base.DATA_DIR / 'experiment_135_block_c_review_confirmation_results.csv'
FOLD_COUNT = 5
BASELINE_CONFIG = 'baseline_validated_exp93_hard_only'
POLICY_CONFIG = 'review_tail1pct_all_standard_and_block_c'
POLICY_LABEL = 'family_holdout_all_standard_block_c_review'
BASELINE_LABEL = 'family_holdout_baseline_hard_only'


def read_rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def family_fold(family: str) -> int:
    """Assign a whole family to a stable audit fold without using outcomes."""
    digest = hashlib.sha256(family.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], byteorder='big') % FOLD_COUNT


def load_source_maps():
    rows = read_rows(SOURCE_PATH)
    baseline = {row['dataset_name']: row for row in rows if row.get('config_name') == BASELINE_CONFIG}
    policy = {row['dataset_name']: row for row in rows if row.get('config_name') == POLICY_CONFIG}
    if len(baseline) != 1117 or set(baseline) != set(policy):
        raise SystemExit(f'coverage mismatch baseline={len(baseline)} policy={len(policy)}')
    return baseline, policy


def audit_row(source_row, config_name, selector_reason):
    row = dict(source_row)
    row.update({
        'experiment_id': EXPERIMENT_ID,
        'config_name': config_name,
        'selector_name': config_name,
        'selector_reason': selector_reason,
        'threshold_method': 'retrospective_family_holdout_audit',
        'family_holdout_fold': family_fold(row['family']),
        'family_holdout_fold_count': FOLD_COUNT,
        'uses_test_labels_for_policy': 0,
        'policy_discovered_posthoc': 1,
        'prospective_validation_required': 1,
    })
    return row


def summarize(rows):
    output = []
    for config_name in sorted({row['config_name'] for row in rows}):
        subset = [row for row in rows if row['config_name'] == config_name]
        values = lambda key: np.asarray([as_float(row.get(key)) for row in subset], dtype=float)
        fold_precisions = []
        for fold in range(FOLD_COUNT):
            fold_rows = [row for row in subset if int(as_float(row.get('family_holdout_fold'))) == fold]
            fold_tp = int(np.sum([as_float(row.get('review_tp')) for row in fold_rows]))
            fold_fp = int(np.sum([as_float(row.get('review_fp')) for row in fold_rows]))
            if fold_tp + fold_fp:
                fold_precisions.append(fold_tp / (fold_tp + fold_fp))
        total_tp = int(np.sum(values('review_tp')))
        total_fp = int(np.sum(values('review_fp')))
        output.append({
            'experiment_id': EXPERIMENT_ID,
            'config_name': config_name,
            'selector_name': config_name,
            'threshold_method': 'retrospective_family_holdout_audit',
            'num_datasets': len(subset),
            'num_families': len({row.get('family', '') for row in subset}),
            'fold_count': FOLD_COUNT,
            'mean_hard_f1': float(np.mean(values('f1'))),
            'hard_zero_f1_count': int(np.sum(values('f1') == 0.0)),
            'mean_hard_fp': float(np.mean(values('fp'))),
            'mean_combined_f1': float(np.mean(values('combined_f1'))),
            'combined_zero_f1_count': int(np.sum(values('combined_zero_f1'))),
            'review_candidate_datasets': int(np.sum(values('review_candidate_count') > 0)),
            'review_hit_datasets': int(np.sum(values('review_hit') > 0)),
            'review_total_tp': total_tp,
            'review_total_fp': total_fp,
            'review_alert_precision': total_tp / max(1, total_tp + total_fp),
            'fold_precision_min': float(min(fold_precisions)) if fold_precisions else 0.0,
            'fold_precision_max': float(max(fold_precisions)) if fold_precisions else 0.0,
            'mean_combined_fp': float(np.mean(values('combined_fp'))),
            'uses_test_labels_for_policy_rows': int(np.sum(values('uses_test_labels_for_policy'))),
            'policy_discovered_posthoc_rows': int(np.sum(values('policy_discovered_posthoc'))),
            'prospective_validation_required_rows': int(np.sum(values('prospective_validation_required'))),
        })
    return sorted(output, key=lambda row: row['config_name'])


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
    baseline, policy = load_source_maps()
    names = sorted(baseline)
    if dataset_limit:
        names = names[:dataset_limit]
    rows = []
    for name in names:
        rows.append(audit_row(
            baseline[name], BASELINE_LABEL,
            'control: validated Exp93 hard alert; family holdout audit only',
        ))
        rows.append(audit_row(
            policy[name], POLICY_LABEL,
            'fixed Exp135 all-Standard plus Block-C review rule; no label-based policy decision',
        ))
    expected = len(names) * 2
    if len(rows) != expected:
        raise SystemExit(f'coverage failure {len(rows)}/{expected}')
    write_csv(base.results_path(EXPERIMENT_ID), rows)
    write_csv(base.summary_path(EXPERIMENT_ID), summarize(rows))
    print(f'{EXPERIMENT_ID} finished rows={len(rows)} datasets={len(names)} folds={FOLD_COUNT}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-limit', type=int)
    run_experiment(parser.parse_args().dataset_limit)
