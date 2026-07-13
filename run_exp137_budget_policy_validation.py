from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import run_experiment_133_block_b_confidence_tiers as exp133
import run_experiment_139_family_neutral_common_support as policy
from run_experiment_60_62_rocket_imaging_selector_variants import (
    CALIBRATION_PROFILES,
    cap_indices_count,
    indices_at_least,
    is_large_data_case,
    large_data_budget,
    load_candidate_predictions,
    top_score_indices,
)
from run_rank_ensemble_calibration import load_dataset_data


ROOT = Path(__file__).resolve().parent
DATA_DIR = Path("/Users/minho/Documents/Dataset")
OUTPUT_DIR = ROOT / "outputs/exp137_policy_train_only_validation"
BASELINE_PATH = DATA_DIR / "experiment_137_operational_triage_results.csv"
B2_SOURCE_PATH = OUTPUT_DIR / "b2_full" / "16_b2_full_coverage_source_manifest.csv"
WORKERS = int(os.environ.get("RANK_EXPERIMENT_WORKERS", "7"))
TRAIN_RUN_GRAIN_VERIFIED = False
FIXED_KS = (1, 2, 3, 4, 5, 6, 8)


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


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def fmt(indices):
    return " ".join(str(int(index)) for index in sorted(indices))


def parse(value):
    return policy.parse_indices(value)


def conformal_budget_status():
    return "available" if TRAIN_RUN_GRAIN_VERIFIED else "blocked_unverified_train_run_grain"


def append_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_checkpoints(path):
    path = Path(path)
    if not path.exists():
        return {}
    restored = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            restored[payload["dataset"]] = payload
    return restored


def source_bundle(row):
    return policy.exp89.exp84_bundle_from_row(row, 10**9)


def threshold_only_candidates(bundles, source):
    """Use only sets already selected by TRAIN-normal calibrated thresholds."""
    rocket = set(bundles["rocket_exp40"]["indices"])
    exp55 = set(bundles["exp55_best"]["indices"])
    exp56 = set(bundles["exp56_best"]["indices"])
    base = indices_at_least(2, [rocket, exp55, exp56])
    source_indices = set((source or {}).get("indices", set()))
    if not source_indices:
        return base
    return base | indices_at_least(2, [rocket, exp55, exp56, source_indices])


def apply_fixed_budget(indices, scores, fixed_k):
    count = max(0, int(fixed_k))
    if count == 0:
        return set()
    return cap_indices_count(set(indices), {"test_scores": np.asarray(scores)}, count)


def prepare_review_cache(x_train, x_test):
    x_train = policy.z_normalize(x_train).astype(np.float32)
    x_test = policy.z_normalize(x_test).astype(np.float32)
    a_train, a_test, b_train, b_test, _, _ = policy.block_base.block_scores(x_train, x_test)
    a_threshold, _, _ = policy.count_cap_threshold(a_train, 0.015)
    b_threshold, _, _ = policy.count_cap_threshold(b_train, 0.01)
    review_candidates = set(np.flatnonzero(a_test > a_threshold).astype(int).tolist())
    review_candidates &= set(np.flatnonzero(b_test > b_threshold).astype(int).tolist())
    return {
        "train_count": len(x_train),
        "review_candidates": review_candidates,
        "b_test": b_test,
        "c_indices": None,
        "x_train": x_train,
        "x_test": x_test,
    }


def review_and_priority_cached(cache, candidate_indices, high, standard):
    candidates = set(cache["review_candidates"]) - set(candidate_indices)
    review = set()
    if cache["train_count"] > 10 and candidates:
        review = {max(candidates, key=lambda index: (float(cache["b_test"][index]), -index))}
    priority = set()
    if review and not high and standard:
        if cache.get("c_indices") is None:
            cache["c_indices"], _, _, _, _, _ = policy.exp135.block_c_candidates(cache["x_train"], cache["x_test"])
        priority = review & set(cache["c_indices"])
    return review, priority


def lanes_from_candidates(review_cache, candidates, block_b, test_size):
    high = set(candidates) & set(block_b)
    standard = set(candidates) - set(block_b)
    review, priority = review_and_priority_cached(review_cache, candidates, high, standard)
    return policy.build_tiers(high, standard, priority, test_size)


def predict_lanes(review_cache, bundles, exp84_source, block_b, test_size, fixed_k=None):
    """Prediction-only policy: no labels, family branching, or TEST-size budget."""
    candidates = threshold_only_candidates(bundles, source_bundle(exp84_source))
    if fixed_k is not None:
        candidates = apply_fixed_budget(candidates, bundles["rocket_exp40"]["test_scores"], fixed_k)
    return lanes_from_candidates(review_cache, candidates, block_b, test_size), candidates


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def lane_record(name, family, method, lanes, candidates, y_test, baseline):
    metrics = policy.metrics_for_lanes(y_test, lanes)
    current = {
        "hard": parse(baseline.get("hard_alert_indices")),
        "standard_review": parse(baseline.get("standard_review_indices")),
        "priority_review": parse(baseline.get("priority_review_indices")),
    }
    output = {
        "dataset": name,
        "family": family,
        "policy": method,
        "candidate_indices": fmt(candidates),
        "hard_indices": fmt(lanes["hard"]),
        "standard_review_indices": fmt(lanes["standard_review"]),
        "priority_review_indices": fmt(lanes["priority_review"]),
        "candidate_count": len(candidates),
        **metrics,
    }
    for lane in ("hard", "standard_review", "priority_review"):
        new = lanes[lane]
        old = current[lane]
        union = old | new
        output[f"baseline_{lane}_indices"] = fmt(old)
        output[f"{lane}_exact_jaccard"] = len(old & new) / len(union) if union else 1.0
    return output


def audit_record(name, record, y_test, bundles, sources, base_row, block_b, baseline, review_cache):
    n_test = len(y_test)
    n_train = len(record["train_series"])
    large = is_large_data_case(record, y_test)
    budget2 = large_data_budget(y_test, rate=0.01, minimum=2, maximum=5)
    budget3 = large_data_budget(y_test, rate=0.02, minimum=3, maximum=8)
    fg = source_bundle(sources.get((name, "family_guard_v1")))
    four_agreement = indices_at_least(
        2,
        [
            bundles["rocket_exp40"]["indices"],
            bundles["exp55_best"]["indices"],
            bundles["exp56_best"]["indices"],
            fg["indices"],
        ],
    )
    guard_count = max(budget3, budget2 * 2)
    rocket_guard = top_score_indices(bundles["rocket_exp40"], guard_count)
    pre_cap = four_agreement & rocket_guard
    post_cap = cap_indices_count(pre_cap, bundles["rocket_exp40"], budget3)
    removed = pre_cap - post_cap
    exp89 = policy.select_exp89_best(record, y_test, bundles, base_row, sources, name)
    exp90 = policy.select_exp90_operational(record, y_test, bundles, exp89, sources, name)
    exp93 = policy.select_exp93_validated(record, bundles, exp90, sources, name)
    lanes = lanes_from_candidates(review_cache, exp93, block_b, len(y_test))
    scores = bundles["rocket_exp40"]["test_scores"]
    return {
        "dataset": name,
        "family": record.get("family", ""),
        "n_train_normal": n_train,
        "n_test": n_test,
        "is_large_case_current": int(large),
        "current_budget_2pct": budget2,
        "current_budget_3pct": budget3,
        "candidate_count_before_budget": len(pre_cap),
        "candidate_count_after_budget": len(post_cap),
        "budget_is_binding": int(len(pre_cap) > budget3),
        "num_candidates_removed_by_budget": len(removed),
        "removed_candidate_indices": fmt(removed),
        "removed_candidate_scores": " ".join(f"{float(scores[index]):.10g}" for index in sorted(removed, key=lambda index: -scores[index])),
        "kept_candidate_indices": fmt(post_cap),
        "kept_candidate_scores": " ".join(f"{float(scores[index]):.10g}" for index in sorted(post_cap, key=lambda index: -scores[index])),
        "downstream_exp89_indices": fmt(exp89),
        "downstream_exp90_indices": fmt(exp90),
        "downstream_exp137_hard_indices": fmt(lanes["hard"]),
        "downstream_exp137_standard_indices": fmt(lanes["standard_review"]),
        "baseline_hard_fp": safe_float(baseline.get("hard_fp"), 0.0),
    }


def run_one(args):
    name, base_row, block_b, baseline, current_sources, b2_sources = args
    record, y_test, bundles = load_candidate_predictions(name, threshold_rates=CALIBRATION_PROFILES["relaxed_15pct"])
    x_train, x_test, _ = load_dataset_data(name)
    review_cache = prepare_review_cache(x_train, x_test)
    audit = audit_record(name, record, y_test, bundles, current_sources, base_row, block_b, baseline, review_cache)
    current_lanes, current_candidates = predict_lanes(review_cache, bundles, current_sources.get((name, "family_guard_v1")), block_b, len(y_test))
    b2_lanes, b2_candidates = predict_lanes(review_cache, bundles, b2_sources.get((name, "count_cap_2pct")), block_b, len(y_test))
    c0 = lane_record(name, record["family"], "C0_train_threshold_only_current_source", current_lanes, current_candidates, y_test, baseline)
    d1a = lane_record(name, record["family"], "D1a_policy_level_train_only_no_budget", b2_lanes, b2_candidates, y_test, baseline)
    fixed = []
    for fixed_k in FIXED_KS:
        lanes, candidates = predict_lanes(review_cache, bundles, current_sources.get((name, "family_guard_v1")), block_b, len(y_test), fixed_k=fixed_k)
        fixed.append(lane_record(name, record["family"], f"fixed_k_{fixed_k}", lanes, candidates, y_test, baseline))
    return audit, c0, d1a, fixed


def aggregate(rows):
    total = lambda key: int(sum(safe_float(row.get(key), 0.0) for row in rows))
    mean = lambda key: float(np.mean([safe_float(row.get(key), 0.0) for row in rows])) if rows else 0.0
    candidates = np.asarray([safe_float(row.get("candidate_count"), 0.0) for row in rows], dtype=float)
    hard_tp, hard_fp = total("hard_tp"), total("hard_fp")
    return {
        "datasets": len(rows),
        "hard_alerts": total("hard_alert_count"),
        "hard_tp": hard_tp,
        "hard_fp": hard_fp,
        "hard_precision": hard_tp / max(1, hard_tp + hard_fp),
        "hard_recall_mean": mean("hard_recall"),
        "mean_hard_f1": mean("hard_f1"),
        "standard_review_count": total("standard_review_count"),
        "standard_review_tp": total("standard_review_tp"),
        "standard_review_fp": total("standard_review_fp"),
        "priority_review_count": total("priority_review_count"),
        "priority_review_tp": total("priority_review_tp"),
        "priority_review_fp": total("priority_review_fp"),
        "candidate_mean": float(np.mean(candidates)),
        "candidate_median": float(np.median(candidates)),
        "candidate_p90": float(np.percentile(candidates, 90)),
        "candidate_p95": float(np.percentile(candidates, 95)),
        "candidate_max": int(np.max(candidates)),
        "candidate_zero_datasets": int(np.sum(candidates == 0)),
        "candidate_ge5_datasets": int(np.sum(candidates >= 5)),
        "candidate_ge10_datasets": int(np.sum(candidates >= 10)),
    }


def run(output_dir=OUTPUT_DIR, dataset_limit=None):
    output_dir = Path(output_dir)
    b2_dir = output_dir / "b2_full"
    baseline = {row["dataset_name"]: row for row in read_rows(BASELINE_PATH)}
    base_primary, _ = policy.exp89.load_base_rows()
    _, block_rows = exp133.load_maps()
    current_sources = policy.source_maps()
    b2_sources = {(row["dataset_name"], row["threshold_method"]): row for row in read_rows(B2_SOURCE_PATH)}
    names = sorted(baseline)
    if dataset_limit:
        names = names[:dataset_limit]
    checkpoint_path = output_dir / "budget_policy_checkpoint.jsonl"
    restored = read_checkpoints(checkpoint_path)
    restored = {name: payload for name, payload in restored.items() if name in names}
    audits = [payload["audit"] for payload in restored.values()]
    c0_rows = [payload["c0"] for payload in restored.values()]
    d1a_rows = [payload["d1a"] for payload in restored.values()]
    fixed_rows = [row for payload in restored.values() for row in payload["fixed"]]
    errors = []
    pending_names = [name for name in names if name not in restored]
    if restored:
        print(f"Resume: restored={len(restored)} pending={len(pending_names)}", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(
                run_one,
                (
                    name,
                    base_primary[name],
                    parse(block_rows[name].get("selected_indices")),
                    baseline[name],
                    current_sources,
                    b2_sources,
                ),
            ): name
            for name in pending_names
        }
        for completed_now, future in enumerate(as_completed(futures), 1):
            name = futures[future]
            try:
                audit, c0, d1a, fixed = future.result()
                audits.append(audit)
                c0_rows.append(c0)
                d1a_rows.append(d1a)
                fixed_rows.extend(fixed)
                append_checkpoint(
                    checkpoint_path,
                    {"dataset": name, "audit": audit, "c0": c0, "d1a": d1a, "fixed": fixed},
                )
            except Exception as exc:
                errors.append((name, repr(exc)))
            done = len(restored) + completed_now
            if done % 10 == 0 or done == len(names):
                print(f"Progress: [{done:4d}/{len(names):4d}] rows={len(c0_rows)} errors={len(errors)}", flush=True)
    audits.sort(key=lambda row: row["dataset"])
    c0_rows.sort(key=lambda row: row["dataset"])
    d1a_rows.sort(key=lambda row: row["dataset"])
    fixed_rows.sort(key=lambda row: (row["policy"], row["dataset"]))
    write_rows(output_dir / "34_test_length_budget_binding_dataset.csv", audits)
    removed = [row for row in audits if row["budget_is_binding"]]
    write_rows(output_dir / "35_test_length_budget_removed_candidates.csv", removed)
    write_rows(output_dir / "38_c0_threshold_only_results.csv", c0_rows)
    write_rows(output_dir / "54_d1a_policy_train_only_no_budget_results.csv", d1a_rows)
    write_rows(output_dir / "51_fixed_k_sensitivity_metrics.csv", [{"fixed_k": k, **aggregate([row for row in fixed_rows if row["policy"] == f"fixed_k_{k}"])} for k in FIXED_KS])
    summaries = {"C0": aggregate(c0_rows), "D1a": aggregate(d1a_rows), "fixed_k": {str(k): aggregate([row for row in fixed_rows if row["policy"] == f"fixed_k_{k}"]) for k in FIXED_KS}}
    write_text(output_dir / "39_c0_threshold_only_summary.md", "# C0 Threshold-only\n\n```json\n" + json.dumps(summaries["C0"], indent=2, sort_keys=True) + "\n```\n\nRetrospective counterfactual. No TEST-length gate or top-N budget is used by the prediction policy.")
    write_text(output_dir / "55_d1a_policy_train_only_no_budget_summary.md", "# D1a B2 + Threshold-only\n\n```json\n" + json.dumps(summaries["D1a"], indent=2, sort_keys=True) + "\n```\n\nPolicy-level train-only candidate policy only; not end-to-end strict or prospective validation.")
    binding_summary = {
        "datasets": len(audits),
        "binding_datasets": sum(row["budget_is_binding"] for row in audits),
        "removed_candidates": sum(row["num_candidates_removed_by_budget"] for row in audits),
        "large_cases": sum(row["is_large_case_current"] for row in audits),
        "errors": errors,
    }
    write_text(output_dir / "36_test_length_budget_binding_summary.md", "# Exp143 TEST-length Budget Binding Audit\n\n```json\n" + json.dumps(binding_summary, indent=2, sort_keys=True) + "\n```\n\nCandidate counts before/after are the Exp89 four-source agreement pool after its current TEST-length top-score guard and before/after its final cap. This audit is diagnostic, not causal proof.")
    write_text(output_dir / "52_fixed_k_workload_frontier.md", "# Fixed-K Workload Sensitivity\n\n```json\n" + json.dumps(summaries["fixed_k"], indent=2, sort_keys=True) + "\n```\n\nNo K is selected as an operating policy. These are retrospective workload-performance sensitivity points.")
    if errors or len(c0_rows) != len(names):
        raise SystemExit(f"coverage failure rows={len(c0_rows)}/{len(names)} errors={errors[:3]}")
    return {"binding": binding_summary, **summaries}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dataset-limit", type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.dataset_limit), indent=2, sort_keys=True))
