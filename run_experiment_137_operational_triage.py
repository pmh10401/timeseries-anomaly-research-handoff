from __future__ import annotations

import argparse
import csv

import numpy as np

import run_experiment_132_block_b_review_integration as base
from run_experiment_89_74d_with_exp84_candidate import as_float, parse_indices
from run_rank_ensemble_calibration import load_dataset_data


EXPERIMENT_ID = 'experiment_137_operational_triage'
CONFIG_NAME = 'operational_high_hard_standard_review_priority_review'
EXP133_PATH = base.DATA_DIR / 'experiment_133_block_b_confidence_tiers_results.csv'
EXP135_PATH = base.DATA_DIR / 'experiment_135_block_c_review_confirmation_results.csv'
EXP133_CONFIG = 'tiered_all_validated_exp93_hard_alerts'
EXP135_CONFIG = 'review_tail1pct_all_standard_and_block_c'


def route_tiers(high, standard, priority, test_size):
    hard = set(high)
    standard_review = set(standard) - hard
    priority_review = set(priority) - hard - standard_review
    all_indices = hard | standard_review | priority_review
    invalid = sorted(index for index in all_indices if index < 0 or index >= test_size)
    if invalid:
        raise ValueError(f'out-of-bounds indices test_size={test_size} indices={invalid[:10]}')
    return {
        'hard': hard,
        'standard_review': standard_review,
        'priority_review': priority_review,
        'no_alert': set(range(test_size)) - all_indices,
    }


def tier_metrics(y_test, indices):
    selected = set(indices)
    truth = set(np.flatnonzero(np.asarray(y_test, dtype=int) == 1).astype(int).tolist())
    tp = len(selected & truth)
    fp = len(selected - truth)
    fn = len(truth - selected)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    denominator = 2 * tp + fp + fn
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': 2 * tp / denominator if denominator else 0.0,
    }


def read_rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def format_indices(indices):
    return ' '.join(str(index) for index in sorted(indices))


def load_maps():
    exp133 = {
        row['dataset_name']: row
        for row in read_rows(EXP133_PATH)
        if row.get('config_name') == EXP133_CONFIG
    }
    exp135 = {
        row['dataset_name']: row
        for row in read_rows(EXP135_PATH)
        if row.get('config_name') == EXP135_CONFIG
    }
    if len(exp133) != 1117 or set(exp133) != set(exp135):
        raise SystemExit(f'coverage mismatch exp133={len(exp133)} exp135={len(exp135)}')
    return exp133, exp135


def make_row(dataset_name, y_test, row133, row135, tiers):
    hard = tier_metrics(y_test, tiers['hard'])
    standard = tier_metrics(y_test, tiers['standard_review'])
    priority = tier_metrics(y_test, tiers['priority_review'])
    combined_indices = tiers['hard'] | tiers['standard_review'] | tiers['priority_review']
    combined = tier_metrics(y_test, combined_indices)
    return {
        'experiment_id': EXPERIMENT_ID,
        'dataset_name': dataset_name,
        'family': row133.get('family', ''),
        'config_name': CONFIG_NAME,
        'selector_name': CONFIG_NAME,
        'selector_reason': 'Exp133 Block-B-supported alerts are hard; unsupported alerts are standard review; Exp135 narrow candidates are priority review',
        'threshold_method': 'operational_triage',
        'score_family': 'validated_exp93_block_b_c_operational_triage',
        'score_source_name': 'exp133_high_standard_plus_exp135_priority_review',
        'hard_alert_indices': format_indices(tiers['hard']),
        'standard_review_indices': format_indices(tiers['standard_review']),
        'priority_review_indices': format_indices(tiers['priority_review']),
        'hard_alert_count': len(tiers['hard']),
        'standard_review_count': len(tiers['standard_review']),
        'priority_review_count': len(tiers['priority_review']),
        'no_alert_count': len(tiers['no_alert']),
        **{f'hard_{key}': value for key, value in hard.items()},
        **{f'standard_review_{key}': value for key, value in standard.items()},
        **{f'priority_review_{key}': value for key, value in priority.items()},
        **{f'combined_{key}': value for key, value in combined.items()},
        'combined_metric_scope': 'human_assisted_diagnostic_only',
        'routing_uses_test_labels': 0,
        'routing_uses_test_position': 0,
        'routing_uses_family_performance': 0,
        'priority_review_posthoc_research_rule': 1,
        'prospective_priority_validation_required': 1,
        'source_exp133_config': EXP133_CONFIG,
        'source_exp135_config': EXP135_CONFIG,
        'source_exp93_hard_count': int(as_float(row133.get('high_confidence_count'))) + int(as_float(row133.get('standard_confidence_count'))),
        'source_exp135_review_count': int(as_float(row135.get('review_candidate_count'))),
        'train_normal_count': int(as_float(row133.get('train_normal_count'))),
        'sequence_length': int(as_float(row133.get('sequence_length'))),
        'test_size': len(y_test),
        'anomaly_count': int(np.sum(np.asarray(y_test, dtype=int) == 1)),
    }


def summarize(rows):
    total = lambda key: int(sum(as_float(row.get(key)) for row in rows))
    mean = lambda key: float(np.mean([as_float(row.get(key)) for row in rows]))
    hard_tp, hard_fp = total('hard_tp'), total('hard_fp')
    standard_tp, standard_fp = total('standard_review_tp'), total('standard_review_fp')
    priority_tp, priority_fp = total('priority_review_tp'), total('priority_review_fp')
    n = len(rows)
    mean_hard_f1 = mean('hard_f1')
    return [{
        'experiment_id': EXPERIMENT_ID,
        'config_name': CONFIG_NAME,
        'selector_name': CONFIG_NAME,
        'threshold_method': 'operational_triage',
        'num_datasets': n,
        'num_families': len({row.get('family', '') for row in rows}),
        'hard_total_alerts': total('hard_alert_count'),
        'hard_total_tp': hard_tp,
        'hard_total_fp': hard_fp,
        'hard_alert_precision': hard_tp / max(1, hard_tp + hard_fp),
        'mean_hard_f1': mean_hard_f1,
        'mean_f1': mean_hard_f1,
        'mean_f1_scope': 'autonomous_hard_alert',
        'hard_zero_f1_count': int(sum(as_float(row.get('hard_f1')) == 0 for row in rows)),
        'standard_review_total_candidates': total('standard_review_count'),
        'standard_review_total_tp': standard_tp,
        'standard_review_total_fp': standard_fp,
        'standard_review_precision': standard_tp / max(1, standard_tp + standard_fp),
        'priority_review_total_candidates': total('priority_review_count'),
        'priority_review_total_tp': priority_tp,
        'priority_review_total_fp': priority_fp,
        'priority_review_precision': priority_tp / max(1, priority_tp + priority_fp),
        'mean_combined_f1': mean('combined_f1'),
        'combined_zero_f1_count': int(sum(as_float(row.get('combined_f1')) == 0 for row in rows)),
        'mean_combined_fp': mean('combined_fp'),
        'hard_alerts_per_100_datasets': 100 * total('hard_alert_count') / max(1, n),
        'review_requests_per_100_datasets': 100 * (total('standard_review_count') + total('priority_review_count')) / max(1, n),
        'routing_uses_test_labels_rows': total('routing_uses_test_labels'),
        'routing_uses_test_position_rows': total('routing_uses_test_position'),
        'routing_uses_family_performance_rows': total('routing_uses_family_performance'),
        'combined_metric_scope': 'human_assisted_diagnostic_only',
        'priority_review_posthoc_research_rule': 1,
        'prospective_priority_validation_required': 1,
    }]


def run_one(dataset_name, row133, row135):
    _, X_test, y_test = load_dataset_data(dataset_name)
    high = parse_indices(row133.get('high_confidence_indices'))
    standard = parse_indices(row133.get('standard_confidence_indices'))
    priority = parse_indices(row135.get('review_candidate_indices'))
    source_hard = parse_indices(row133.get('selected_indices'))
    if high | standard != source_hard:
        raise ValueError(f'Exp133 tier union mismatch dataset={dataset_name}')
    tiers = route_tiers(high, standard, priority, len(X_test))
    return make_row(dataset_name, y_test, row133, row135, tiers)


def run_experiment(dataset_limit=None):
    exp133, exp135 = load_maps()
    names = sorted(exp133)
    if dataset_limit:
        names = names[:dataset_limit]
    rows = []
    errors = []
    for done, name in enumerate(names, 1):
        try:
            rows.append(run_one(name, exp133[name], exp135[name]))
        except Exception as exc:
            errors.append((name, repr(exc)))
            print(f'ERROR dataset={name} error={exc!r}', flush=True)
        if done % 25 == 0 or done == len(names):
            print(f'Progress: [{done:4d}/{len(names):4d}] rows={len(rows)} last={name} errors={len(errors)}', flush=True)
    if errors or len(rows) != len(names):
        raise SystemExit(f'coverage failure {len(rows)}/{len(names)} errors={errors[:5]}')
    base.write_csv(base.results_path(EXPERIMENT_ID), rows)
    base.write_csv(base.summary_path(EXPERIMENT_ID), summarize(rows))
    print(f'{EXPERIMENT_ID} finished rows={len(rows)} datasets={len(names)} errors=0', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-limit', type=int)
    run_experiment(parser.parse_args().dataset_limit)
