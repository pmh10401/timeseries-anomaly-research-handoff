from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import run_experiment_133_block_b_confidence_tiers as exp133
import run_experiment_139_family_neutral_common_support as b1
import run_model_hard_research_experiments as model
from run_experiment_29_train_normal_threshold_calibration import train_false_positive_stats
from run_experiment_40_original_score_normalization_sweep import count_cap_threshold
from run_rank_ensemble_calibration import load_dataset_data, z_normalize


DATA_DIR = Path("/Users/minho/Documents/Dataset")
OUTPUT_DIR = Path("outputs/exp137_policy_train_only_validation")
EXPERIMENT_ID = "experiment_141_family_neutral_full_coverage"
BASELINE_PATH = DATA_DIR / "experiment_137_operational_triage_results.csv"
B1_PATH = Path("outputs/exp137_strict_train_only_execution/b1_full/b1_family_neutral_common_support_results.csv")
WORKERS = int(os.environ.get("RANK_EXPERIMENT_WORKERS", "6"))
CONFIG = {
    # Exp84's fixed feature configuration is persisted under the Exp87
    # diagnostics name used by the existing candidate-source CSV.
    "name": b1.EXP87_CONFIG,
    "kind": "aeon_multirocket_hydra",
    "num_kernels": 1024,
    "hydra_kernels": 4,
    "hydra_groups": 32,
    "neighbors": 3,
    "score_mode": "local_gap",
    "feature_prune": "stable_tail",
    "feature_keep": 1024,
    "random_state": 20260717,
}


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_indices(indices):
    return " ".join(str(int(index)) for index in sorted(indices))


def source_row_from_scores(dataset_name, family, train_scores, test_scores, rate, method):
    threshold, q_effective, cap_target = count_cap_threshold(train_scores, rate)
    selected = set(np.flatnonzero(np.asarray(test_scores) > threshold).astype(int).tolist())
    order = np.argsort(-np.asarray(test_scores), kind="stable")[: min(12, len(test_scores))]
    train_exceed_count, train_exceed_rate = train_false_positive_stats(train_scores, threshold)
    return {
        "experiment_id": EXPERIMENT_ID,
        "dataset_name": dataset_name,
        "family": family,
        "config_name": CONFIG["name"],
        "threshold_method": method,
        "threshold_family": "count_cap_rate",
        "random_state": CONFIG["random_state"],
        "source_uses_family_name": 0,
        "source_uses_test_length": 0,
        "threshold": threshold,
        "q_effective": q_effective,
        "cap_target": cap_target,
        "train_exceed_count": train_exceed_count,
        "train_exceed_rate": train_exceed_rate,
        "selected_indices": format_indices(selected),
        "top_score_indices": " ".join(str(int(index)) for index in order),
        "top1_threshold_margin": float(test_scores[order[0]] - threshold) if len(order) else "",
        "top1_top2_margin": float(test_scores[order[0]] - test_scores[order[1]]) if len(order) > 1 else "",
        "predicted_count": len(selected),
    }


def exp84_score_pair(record):
    target_len = min(max(8, model.target_len_for_record(record, "actual_median")), 2048)
    X_train_raw = model.align_series_lengths(record["train_series"], target_len)
    X_test_raw = model.align_series_lengths(record["test_series"], target_len)
    X_train_z = model.z_normalize(X_train_raw).astype(np.float32)
    X_test_z = model.z_normalize(X_test_raw).astype(np.float32)
    X_train, X_test = model.prepare_series_pair_for_scale(
        CONFIG.get("series_scale", "per_series_z"), X_train_raw, X_test_raw, X_train_z, X_test_z
    )
    return model.score_pair_for_config(X_train, X_test, target_len, dict(CONFIG), record, feature_cache={})


def lane_diff(current, neutral, test_size):
    output = {}
    for lane in ("hard", "standard_review", "priority_review"):
        left = current[lane]
        right = neutral[lane]
        union = left | right
        output[f"{lane}_exact_match"] = int(left == right)
        output[f"{lane}_jaccard"] = len(left & right) / len(union) if union else 1.0
        for tolerance in (1, 3, 5):
            matched = b1.lane_jaccard(left, right) if tolerance == 0 else None
            if matched is None:
                from run_exp137_policy_validation_evidence import tolerance_match

                value = tolerance_match(left, right, tolerance)
                output[f"{lane}_tolerance_{tolerance}_matched"] = value["matched"]
    output["test_size"] = test_size
    return output


def candidate_diff_rows(results):
    rows = []
    lane_columns = {
        "hard": ("baseline_hard_indices", "hard_indices"),
        "standard_review": ("baseline_standard_indices", "standard_review_indices"),
        "priority_review": ("baseline_priority_indices", "priority_review_indices"),
    }
    for result in results:
        for lane, (baseline_column, new_column) in lane_columns.items():
            baseline = b1.parse_indices(result.get(baseline_column))
            new = b1.parse_indices(result.get(new_column))
            union = baseline | new
            row = {
                "dataset": result["dataset_name"],
                "family": result.get("family", ""),
                "lane": lane,
                "baseline_indices": format_indices(baseline),
                "b2_indices": format_indices(new),
                "added_indices": format_indices(new - baseline),
                "removed_indices": format_indices(baseline - new),
                "exact_match": int(baseline == new),
                "exact_jaccard": len(baseline & new) / len(union) if union else 1.0,
            }
            for tolerance in (1, 3, 5):
                from run_exp137_policy_validation_evidence import tolerance_match

                matched = tolerance_match(baseline, new, tolerance)
                row[f"tolerance_{tolerance}_matched"] = matched["matched"]
                row[f"tolerance_{tolerance}_baseline_only"] = matched["left_only"]
                row[f"tolerance_{tolerance}_b2_only"] = matched["right_only"]
            rows.append(row)
    return rows


def lane_transition_rows(results):
    rows = []
    for result in results:
        baseline_lanes = {
            "hard": b1.parse_indices(result.get("baseline_hard_indices")),
            "standard_review": b1.parse_indices(result.get("baseline_standard_indices")),
            "priority_review": b1.parse_indices(result.get("baseline_priority_indices")),
        }
        new_lanes = {
            "hard": b1.parse_indices(result.get("hard_indices")),
            "standard_review": b1.parse_indices(result.get("standard_review_indices")),
            "priority_review": b1.parse_indices(result.get("priority_review_indices")),
        }
        all_indices = set().union(*baseline_lanes.values(), *new_lanes.values())
        for index in sorted(all_indices):
            before = next((lane for lane, indices in baseline_lanes.items() if index in indices), "no_alert")
            after = next((lane for lane, indices in new_lanes.items() if index in indices), "no_alert")
            if before != after:
                rows.append(
                    {
                        "dataset": result["dataset_name"],
                        "family": result.get("family", ""),
                        "candidate_index": index,
                        "baseline_lane": before,
                        "b2_lane": after,
                        "transition": f"{before}->{after}",
                    }
                )
    return rows


def run_one(args):
    name, base_row, block_b, baseline = args
    record, y_test, bundles = b1.load_candidate_predictions(name, threshold_rates=b1.CALIBRATION_PROFILES["relaxed_15pct"])
    train_scores, test_scores = exp84_score_pair(record)
    cap2 = source_row_from_scores(name, record["family"], train_scores, test_scores, 0.02, "count_cap_2pct")
    cap3 = source_row_from_scores(name, record["family"], train_scores, test_scores, 0.03, "count_cap_3pct")
    sources = {(name, "family_guard_v1"): dict(cap2), (name, "count_cap_2pct"): cap2, (name, "count_cap_3pct"): cap3}
    X_train, X_test, _ = load_dataset_data(name)
    _, _, exp93_indices, lanes = b1.policy_result(name, X_train, X_test, y_test, record, bundles, base_row, sources, block_b)
    baseline_lanes = {
        "hard": b1.parse_indices(baseline.get("hard_alert_indices")),
        "standard_review": b1.parse_indices(baseline.get("standard_review_indices")),
        "priority_review": b1.parse_indices(baseline.get("priority_review_indices")),
    }
    metrics = b1.metrics_for_lanes(y_test, lanes)
    result = {
        "dataset_name": name,
        "family": record["family"],
        "exp93_indices": format_indices(exp93_indices),
        "hard_indices": format_indices(lanes["hard"]),
        "standard_review_indices": format_indices(lanes["standard_review"]),
        "priority_review_indices": format_indices(lanes["priority_review"]),
        **metrics,
        **lane_diff(baseline_lanes, lanes, len(y_test)),
        "baseline_hard_indices": baseline.get("hard_alert_indices", ""),
        "baseline_standard_indices": baseline.get("standard_review_indices", ""),
        "baseline_priority_indices": baseline.get("priority_review_indices", ""),
    }
    return [cap2, cap3], result


def summarize(rows):
    total = lambda key: int(sum(float(row.get(key, 0) or 0) for row in rows))
    mean = lambda key: sum(float(row.get(key, 0) or 0) for row in rows) / max(1, len(rows))
    tp, fp = total("hard_tp"), total("hard_fp")
    standard_tp, standard_fp = total("standard_review_tp"), total("standard_review_fp")
    priority_tp, priority_fp = total("priority_review_tp"), total("priority_review_fp")
    return {
        "experiment_id": EXPERIMENT_ID,
        "datasets": len(rows),
        "hard_alerts": total("hard_alert_count"),
        "hard_tp": tp,
        "hard_fp": fp,
        "hard_precision": tp / max(1, tp + fp),
        "mean_hard_f1": mean("hard_f1"),
        "standard_review_candidates": total("standard_review_count"),
        "standard_review_tp": standard_tp,
        "standard_review_fp": standard_fp,
        "standard_review_precision": standard_tp / max(1, standard_tp + standard_fp),
        "priority_review_candidates": total("priority_review_count"),
        "priority_review_tp": priority_tp,
        "priority_review_fp": priority_fp,
        "priority_review_precision": priority_tp / max(1, priority_tp + priority_fp),
        "hard_changed_datasets": sum(not int(row["hard_exact_match"]) for row in rows),
        "standard_changed_datasets": sum(not int(row["standard_review_exact_match"]) for row in rows),
        "priority_changed_datasets": sum(not int(row["priority_review_exact_match"]) for row in rows),
        "retrospective_only": True,
        "combined_metric_scope": "human_assisted_diagnostic_only",
    }


def b1_common_support_comparison(results):
    if not B1_PATH.exists():
        return {"available": False, "reason": f"missing B1 result: {B1_PATH}"}
    b1_rows = {row["dataset_name"]: row for row in read_rows(B1_PATH)}
    b2_rows = {row["dataset_name"]: row for row in results}
    common = sorted(set(b1_rows) & set(b2_rows))
    exact = 0
    mismatches = []
    for name in common:
        b1_row = b1_rows[name]
        b2_row = b2_rows[name]
        matches = all(
            str(b1_row.get(f"neutral_{old}", "")).strip() == str(b2_row.get(new, "")).strip()
            for old, new in (
                ("hard_indices", "hard_indices"),
                ("standard_indices", "standard_review_indices"),
                ("priority_indices", "priority_review_indices"),
            )
        )
        exact += int(matches)
        if not matches:
            mismatches.append(name)
    return {
        "available": True,
        "common_support_datasets": len(common),
        "exact_lane_match_datasets": exact,
        "mismatch_datasets": mismatches,
    }


def write_summary_markdown(path, summary, b1_comparison, errors):
    lines = [
        "# Experiment 141 B2 Full-Coverage Family-Neutral Results",
        "",
        "Status: retrospective counterfactual ablation. This is not prospective validation and not end-to-end strict TRAIN-only validation.",
        "",
        "## Fixed policy",
        "",
        "- Exp84 feature/score configuration: `aeon_mrh_mr1024_hk4_g32_prune1024_stable_tail_local_gap_knn3`, seed `20260717`.",
        "- Every baseline dataset receives a newly calculated Exp84 source with the universal `count_cap_2pct` threshold.",
        "- Family and dataset names are not used to decide whether the Exp84 source is calculated.",
        "- Existing TEST-length budget behavior remains unchanged in B2; C/D1 are separately blocked by the unresolved operating budget contract.",
        "- Priority review remains review-only. Any combined metric is human-assisted diagnostic-only.",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in (
        "datasets", "hard_alerts", "hard_tp", "hard_fp", "hard_precision", "mean_hard_f1",
        "standard_review_candidates", "standard_review_tp", "standard_review_fp", "standard_review_precision",
        "priority_review_candidates", "priority_review_tp", "priority_review_fp", "priority_review_precision",
        "hard_changed_datasets", "standard_changed_datasets", "priority_changed_datasets",
    ):
        lines.append(f"| {key} | {summary.get(key, '')} |")
    lines.extend(["", "## B1 common-support comparison", "", "```json", json.dumps(b1_comparison, indent=2, sort_keys=True), "```", "", "## Coverage status", "", f"- Source calculation errors: {len(errors)}", "- A complete 1,117-dataset result is valid only when errors are zero and `datasets` is 1,117."])
    Path(path).write_text("\n".join(lines) + "\n")


def run(dataset_limit=None, output_dir=OUTPUT_DIR):
    base_primary, _ = b1.exp89.load_base_rows()
    _, block_rows = exp133.load_maps()
    baseline = {row["dataset_name"]: row for row in read_rows(BASELINE_PATH)}
    names = sorted(baseline)
    if dataset_limit:
        names = names[:dataset_limit]
    source_rows, results, errors = [], [], []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(run_one, (name, base_primary[name], b1.parse_indices(block_rows[name].get("selected_indices")), baseline[name])): name for name in names}
        for done, future in enumerate(as_completed(futures), 1):
            name = futures[future]
            try:
                source, result = future.result()
                source_rows.extend(source)
                results.append(result)
            except Exception as exc:
                errors.append((name, repr(exc)))
            if done % 10 == 0 or done == len(names):
                print(f"Progress: [{done:4d}/{len(names):4d}] rows={len(results)} errors={len(errors)}", flush=True)
    source_rows.sort(key=lambda row: (row["dataset_name"], row["threshold_method"]))
    results.sort(key=lambda row: row["dataset_name"])
    output_dir = Path(output_dir)
    write_rows(output_dir / "16_b2_full_coverage_source_manifest.csv", source_rows)
    write_rows(output_dir / "17_b2_family_neutral_results.csv", results)
    candidate_rows = candidate_diff_rows(results)
    transitions = lane_transition_rows(results)
    write_rows(output_dir / "19_b2_candidate_level_diff.csv", candidate_rows)
    write_rows(output_dir / "20_b2_lane_transition.csv", transitions)
    summary = summarize(results)
    summary["errors"] = errors
    summary["b1_common_support_comparison"] = b1_common_support_comparison(results)
    (output_dir / "18_b2_family_neutral_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_summary_markdown(output_dir / "18_b2_family_neutral_summary.md", summary, summary["b1_common_support_comparison"], errors)
    if errors or len(results) != len(names):
        raise SystemExit(f"B2 coverage failure rows={len(results)}/{len(names)} errors={errors[:3]}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_limit, args.output_dir), indent=2, sort_keys=True))
