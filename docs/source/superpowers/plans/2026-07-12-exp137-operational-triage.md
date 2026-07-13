# Exp137 Operational Triage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run an operational three-tier routing experiment that turns validated Exp133 and Exp135 indices into hard alerts, standard review, priority review, and no-alert outcomes without label-based routing.

**Architecture:** A new result-integration script reads the full Exp133 and Exp135 detail CSVs, validates exact dataset coverage, and routes saved indices with a pure function. Labels are loaded only after routing to compute separate autonomous hard-alert metrics, review workload metrics, and explicitly human-assisted combined metrics. The sequential runner and dashboard receive one new registry entry.

**Tech Stack:** Python standard library (`argparse`, `csv`, `pathlib`), NumPy, existing dataset/result helpers, `unittest`, sequential experiment runner, local dashboard.

## Global Constraints

- Use exactly the Exp133 `tiered_all_validated_exp93_hard_alerts` and Exp135 `review_tail1pct_all_standard_and_block_c` inputs.
- The routing function must not receive test labels, anomaly counts, family performance, tail positions, or oracle metrics.
- Hard alert, standard review, and priority review indices must be pairwise disjoint and within test bounds.
- Combined F1 is human-assisted diagnostic performance, never autonomous model performance.
- Full execution must produce exactly `1117` detail rows, zero dataset errors, `1691` hard TP, `314` hard FP, `8` priority-review TP, and `1` priority-review FP.
- Keep GitHub synchronization disabled. Local commits may include only files created or modified by this plan.

---

### Task 1: Pure routing and separated metric contracts

**Files:**
- Create: `run_experiment_137_operational_triage.py`
- Create: `scratch/test_experiment_137_operational_triage.py`

**Interfaces:**
- Consumes: string-encoded index fields from Exp133 and Exp135 rows.
- Produces: `route_tiers(high: set[int], standard: set[int], priority: set[int], test_size: int) -> dict[str, set[int]]` and `tier_metrics(y_test: np.ndarray, indices: set[int]) -> dict[str, float | int]`.

- [ ] **Step 1: Write failing routing tests**

```python
import unittest
import numpy as np
import run_experiment_137_operational_triage as target


class OperationalTriageTests(unittest.TestCase):
    def test_route_tiers_applies_precedence_and_is_disjoint(self):
        tiers = target.route_tiers({1, 3}, {2, 3}, {2, 4}, test_size=6)
        self.assertEqual(tiers['hard'], {1, 3})
        self.assertEqual(tiers['standard_review'], {2})
        self.assertEqual(tiers['priority_review'], {4})
        self.assertFalse(tiers['hard'] & tiers['standard_review'])
        self.assertFalse(tiers['hard'] & tiers['priority_review'])
        self.assertFalse(tiers['standard_review'] & tiers['priority_review'])

    def test_route_tiers_rejects_out_of_bounds_index(self):
        with self.assertRaises(ValueError):
            target.route_tiers({5}, set(), set(), test_size=5)

    def test_labels_do_not_change_routing(self):
        first = target.route_tiers({1}, {2}, {3}, test_size=4)
        second = target.route_tiers({1}, {2}, {3}, test_size=4)
        self.assertEqual(first, second)

    def test_metrics_keep_hard_and_review_meaning_separate(self):
        y = np.asarray([0, 1, 0, 1], dtype=int)
        hard = target.tier_metrics(y, {1, 2})
        review = target.tier_metrics(y, {3})
        self.assertEqual((hard['tp'], hard['fp'], hard['fn']), (1, 1, 1))
        self.assertEqual((review['tp'], review['fp']), (1, 0))
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m unittest scratch/test_experiment_137_operational_triage.py -v`

Expected: import failure because `run_experiment_137_operational_triage.py` does not exist.

- [ ] **Step 3: Implement the pure functions**

```python
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
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision,
            'recall': recall, 'f1': 2 * tp / denominator if denominator else 0.0}
```

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python3 -m unittest scratch/test_experiment_137_operational_triage.py -v`

Expected: four tests pass.

- [ ] **Step 5: Commit Task 1 locally**

```bash
git add run_experiment_137_operational_triage.py scratch/test_experiment_137_operational_triage.py
git commit -m "feat: add exp137 operational tier routing"
```

### Task 2: Full Exp133/135 integration and CSV summaries

**Files:**
- Modify: `run_experiment_137_operational_triage.py`
- Modify: `scratch/test_experiment_137_operational_triage.py`

**Interfaces:**
- Consumes: `/Users/minho/Documents/Dataset/experiment_133_block_b_confidence_tiers_results.csv` and `/Users/minho/Documents/Dataset/experiment_135_block_c_review_confirmation_results.csv`.
- Produces: `load_maps()`, `make_row(...)`, `summarize(rows)`, and `run_experiment(dataset_limit=None)` plus Exp137 detail and summary CSVs.

- [ ] **Step 1: Add failing integration-summary tests**

```python
def test_summary_keeps_autonomous_and_review_totals_separate(self):
    rows = [
        {'hard_tp': 2, 'hard_fp': 1, 'hard_fn': 1, 'hard_f1': 2 / 3,
         'standard_review_tp': 1, 'standard_review_fp': 2,
         'priority_review_tp': 1, 'priority_review_fp': 0,
         'combined_tp': 4, 'combined_fp': 3, 'combined_fn': 0,
         'combined_f1': 8 / 11, 'routing_uses_test_labels': 0,
         'hard_alert_count': 3, 'standard_review_count': 3,
         'priority_review_count': 1}
    ]
    summary = target.summarize(rows)[0]
    self.assertEqual(summary['hard_total_tp'], 2)
    self.assertEqual(summary['hard_total_fp'], 1)
    self.assertEqual(summary['priority_review_total_tp'], 1)
    self.assertEqual(summary['priority_review_total_fp'], 0)
    self.assertEqual(summary['routing_uses_test_labels_rows'], 0)
```

- [ ] **Step 2: Run the new test and verify RED**

Run: `python3 -m unittest scratch/test_experiment_137_operational_triage.py -v`

Expected: failure because `summarize` is not implemented.

- [ ] **Step 3: Implement source loading, row construction, and summary**

```python
EXP133_CONFIG = 'tiered_all_validated_exp93_hard_alerts'
EXP135_CONFIG = 'review_tail1pct_all_standard_and_block_c'


def load_maps():
    exp133 = {row['dataset_name']: row for row in read_rows(EXP133_PATH)
              if row.get('config_name') == EXP133_CONFIG}
    exp135 = {row['dataset_name']: row for row in read_rows(EXP135_PATH)
              if row.get('config_name') == EXP135_CONFIG}
    if len(exp133) != 1117 or set(exp133) != set(exp135):
        raise SystemExit(f'coverage mismatch exp133={len(exp133)} exp135={len(exp135)}')
    return exp133, exp135


def run_one(dataset_name, row133, row135):
    _, _, y_test = load_dataset_data(dataset_name)
    tiers = route_tiers(
        parse_indices(row133.get('high_confidence_indices')),
        parse_indices(row133.get('standard_confidence_indices')),
        parse_indices(row135.get('review_candidate_indices')),
        len(y_test),
    )
    return make_row(dataset_name, y_test, row133, row135, tiers)


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
        'config_name': 'operational_high_hard_standard_review_priority_review',
        'selector_name': 'operational_high_hard_standard_review_priority_review',
        'threshold_method': 'operational_triage',
        'hard_alert_indices': format_indices(tiers['hard']),
        'standard_review_indices': format_indices(tiers['standard_review']),
        'priority_review_indices': format_indices(tiers['priority_review']),
        'hard_alert_count': len(tiers['hard']),
        'standard_review_count': len(tiers['standard_review']),
        'priority_review_count': len(tiers['priority_review']),
        **{f'hard_{key}': value for key, value in hard.items()},
        **{f'standard_review_{key}': value for key, value in standard.items()},
        **{f'priority_review_{key}': value for key, value in priority.items()},
        **{f'combined_{key}': value for key, value in combined.items()},
        'combined_metric_scope': 'human_assisted_diagnostic_only',
        'routing_uses_test_labels': 0,
        'routing_uses_test_position': 0,
        'routing_uses_family_performance': 0,
        'train_normal_count': int(as_float(row133.get('train_normal_count'))),
        'test_size': len(y_test),
    }


def summarize(rows):
    total = lambda key: int(sum(as_float(row.get(key)) for row in rows))
    mean = lambda key: float(np.mean([as_float(row.get(key)) for row in rows]))
    hard_tp, hard_fp = total('hard_tp'), total('hard_fp')
    standard_tp, standard_fp = total('standard_review_tp'), total('standard_review_fp')
    priority_tp, priority_fp = total('priority_review_tp'), total('priority_review_fp')
    n = len(rows)
    return [{
        'experiment_id': EXPERIMENT_ID,
        'config_name': 'operational_high_hard_standard_review_priority_review',
        'selector_name': 'operational_high_hard_standard_review_priority_review',
        'threshold_method': 'operational_triage',
        'num_datasets': n,
        'hard_total_alerts': total('hard_alert_count'),
        'hard_total_tp': hard_tp,
        'hard_total_fp': hard_fp,
        'hard_alert_precision': hard_tp / max(1, hard_tp + hard_fp),
        'mean_hard_f1': mean('hard_f1'),
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
        'hard_alerts_per_100_datasets': 100 * total('hard_alert_count') / max(1, n),
        'review_requests_per_100_datasets': 100 * (total('standard_review_count') + total('priority_review_count')) / max(1, n),
        'routing_uses_test_labels_rows': total('routing_uses_test_labels'),
        'combined_metric_scope': 'human_assisted_diagnostic_only',
    }]
```

Use a local `format_indices(indices)` helper that joins sorted integer indices with spaces. Keep all tier precision values based on aggregated TP/FP totals rather than averaging per-row precision.

- [ ] **Step 4: Run unit tests and smoke execution**

Run: `python3 -m unittest scratch/test_experiment_137_operational_triage.py -v`

Expected: all tests pass.

Run: `python3 run_experiment_137_operational_triage.py --dataset-limit 10`

Expected: `rows=10 datasets=10 errors=0`, with no out-of-bounds or overlap error.

- [ ] **Step 5: Commit Task 2 locally**

```bash
git add run_experiment_137_operational_triage.py scratch/test_experiment_137_operational_triage.py
git commit -m "feat: integrate exp137 operating metrics"
```

### Task 3: Queue, dashboard, full execution, and result record

**Files:**
- Modify: `run_rank_experiments_sequential.py`
- Modify: `serve_rank_dashboard.py`
- Modify: `docs/task.md`

**Interfaces:**
- Consumes: Exp137 script and its detail/summary/stdout paths.
- Produces: sequential queue entry, dashboard experiment entry, completed queue state, and a concise project record.

- [ ] **Step 1: Add Exp137 registry entries**

Add to `EXPERIMENTS` in `run_rank_experiments_sequential.py`:

```python
{
    'id': 'experiment_137_operational_triage',
    'script': ROOT / 'run_experiment_137_operational_triage.py',
    'detail_csv': DATA_DIR / 'experiment_137_operational_triage_results.csv',
    'summary_csv': DATA_DIR / 'experiment_137_operational_triage_summary.csv',
    'stdout_log': DATA_DIR / 'experiment_137_operational_triage_stdout.log',
},
```

Add to `EXPERIMENTS` in `serve_rank_dashboard.py`:

```python
'experiment_137_operational_triage': {
    'label': 'Experiment 137 · Operational alert triage',
    'detail_csv': DATA_DIR / 'experiment_137_operational_triage_results.csv',
    'summary_csv': DATA_DIR / 'experiment_137_operational_triage_summary.csv',
    'stdout_log': DATA_DIR / 'experiment_137_operational_triage_stdout.log',
    'expected_datasets': 1117,
},
```

- [ ] **Step 2: Verify syntax, tests, and registry discovery**

Run: `python3 -m py_compile run_experiment_137_operational_triage.py run_rank_experiments_sequential.py serve_rank_dashboard.py`

Run: `python3 -m unittest scratch/test_experiment_137_operational_triage.py -v`

Run: `python3 run_rank_experiments_sequential.py list | tail -5`

Expected: Exp137 appears as pending and all tests pass.

- [ ] **Step 3: Run Exp137 through the sequential runner**

Run: `python3 run_rank_experiments_sequential.py run experiment_137_operational_triage`

Expected: return code `0`, `1117` datasets, `1117` detail rows, and no coverage errors.

- [ ] **Step 4: Verify acceptance totals and dashboard**

Run:

```bash
python3 - <<'PY'
import csv
p = '/Users/minho/Documents/Dataset/experiment_137_operational_triage_summary.csv'
row = next(csv.DictReader(open(p)))
assert int(row['num_datasets']) == 1117
assert int(row['hard_total_tp']) == 1691
assert int(row['hard_total_fp']) == 314
assert int(row['priority_review_total_tp']) == 8
assert int(row['priority_review_total_fp']) == 1
assert int(row['routing_uses_test_labels_rows']) == 0
print(row)
PY
```

Run: `python3 dashboardctl.py restart && python3 dashboardctl.py status`

Expected: dashboard is reachable at `http://127.0.0.1:8765/` and Exp137 is listed as completed.

- [ ] **Step 5: Record the result and commit locally**

Add Exp137 totals, interpretation, and the prospective-validation caveat under `Recently Completed` in `docs/task.md`.

```bash
git add run_rank_experiments_sequential.py serve_rank_dashboard.py docs/task.md
git commit -m "chore: register and record exp137"
```

Do not push either local commit to GitHub.
