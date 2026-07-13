from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import run_experiment_132_block_b_review_integration as block_base
import run_experiment_133_block_b_confidence_tiers as exp133
import run_experiment_135_block_c_review_confirmation as exp135
import run_experiment_137_operational_triage as exp137
import run_experiment_89_74d_with_exp84_candidate as exp89
import run_experiment_90_zero_f1_repair_selector as exp90
import run_experiment_93_nonpos_candidate_reranker as exp93
from run_experiment_40_original_score_normalization_sweep import count_cap_threshold
from run_experiment_60_62_rocket_imaging_selector_variants import (
    CALIBRATION_PROFILES,
    cap_indices_count,
    indices_at_least,
    is_large_data_case,
    large_data_budget,
    load_candidate_predictions,
    top_score_indices,
)
from run_rank_ensemble_calibration import load_dataset_data, z_normalize


DATA_DIR = Path("/Users/minho/Documents/Dataset")
OUTPUT_DIR = Path("outputs/exp137_strict_train_only_execution")
EXPERIMENT_ID = "experiment_139_family_neutral_common_support"
EXP87_PATH = DATA_DIR / "experiment_87_exp84_index_diagnostics_results.csv"
EXP89_PATH = DATA_DIR / "experiment_89_74d_with_exp84_candidate_results.csv"
EXP90_PATH = DATA_DIR / "experiment_90_zero_f1_repair_selector_results.csv"
EXP119A_PATH = DATA_DIR / "experiment_119a_exp93_rank_order_validation_results.csv"
EXP137_PATH = DATA_DIR / "experiment_137_operational_triage_results.csv"
EXP87_CONFIG = exp89.EXP87_CONFIG
EXP89_SELECTOR = exp90.EXP89_BEST_SELECTOR
EXP90_SELECTOR = exp93.EXP90_OPERATIONAL_SELECTOR
EXP93_SELECTOR = "nonpos_weak_alert_replace"
EXP119A_SELECTOR = "exp93_rank_order_validated"
WORKERS = int(os.environ.get("RANK_EXPERIMENT_WORKERS", "6"))


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
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


def parse_indices(value):
    return exp89.parse_indices(value)


def format_indices(indices):
    return exp89.format_indices(indices)


def replace_family_guard_with_cap2(sources, dataset_name):
    neutral = dict(sources)
    cap2 = sources.get((dataset_name, "count_cap_2pct"))
    if cap2 is None:
        raise ValueError(f"missing count_cap_2pct source for {dataset_name}")
    neutral[(dataset_name, "family_guard_v1")] = dict(cap2)
    return neutral


def build_tiers(high, standard, priority, test_size):
    hard = set(high)
    standard_review = set(standard) - hard
    priority_review = set(priority) - hard - standard_review
    all_indices = hard | standard_review | priority_review
    invalid = sorted(index for index in all_indices if index < 0 or index >= int(test_size))
    if invalid:
        raise ValueError(f"out-of-bounds indices: {invalid[:10]}")
    return {
        "hard": hard,
        "standard_review": standard_review,
        "priority_review": priority_review,
        "no_alert": set(range(int(test_size))) - all_indices,
    }


def source_maps():
    rows = read_rows(EXP87_PATH)
    out = {}
    for row in rows:
        if row.get("config_name") != EXP87_CONFIG:
            continue
        method = row.get("threshold_method")
        if method in {"family_guard_v1", "count_cap_2pct", "count_cap_3pct"}:
            out[(row["dataset_name"], method)] = row
    return out


def common_support_names(sources):
    return sorted(
        name
        for name in {key[0] for key in sources}
        if all((name, method) in sources for method in ("family_guard_v1", "count_cap_2pct", "count_cap_3pct"))
    )


def select_exp89_best(record, y_test, bundles, base_row, sources, dataset_name):
    n = len(y_test)
    base_indices = parse_indices(base_row.get("selected_indices"))
    large_case = is_large_data_case(record, y_test)
    rocket = bundles["rocket_exp40"]["indices"]
    exp55_indices = bundles["exp55_best"]["indices"]
    exp56_indices = bundles["exp56_best"]["indices"]
    fg = exp89.exp84_bundle_from_row(sources.get((dataset_name, "family_guard_v1")), n)
    cap3 = exp89.exp84_bundle_from_row(sources.get((dataset_name, "count_cap_3pct")), n)
    budget2 = large_data_budget(y_test, rate=0.01, minimum=2, maximum=5)
    budget3 = large_data_budget(y_test, rate=0.02, minimum=3, maximum=8)
    rocket_guard = top_score_indices(bundles["rocket_exp40"], max(budget3, budget2 * 2))
    primary_fg = cap_indices_count(
        indices_at_least(2, [rocket, exp55_indices, exp56_indices, fg["indices"]]) & rocket_guard,
        bundles["rocket_exp40"],
        budget3,
    )
    top1 = fg["top1"]
    train_safe = fg["train_exceed_rate"] <= 0.015
    margin_safe = fg["top1_threshold_margin"] >= 0.0 and fg["top1_top2_margin"] >= 0.0
    noalert_repair = {top1} if not base_indices and top1 is not None and train_safe and margin_safe else set()
    selected = (base_indices if not large_case else (base_indices | primary_fg)) | noalert_repair
    return selected


def select_exp90_operational(record, y_test, bundles, exp89_indices, sources, dataset_name):
    n = len(y_test)
    tiny_train = len(record["train_series"]) <= 10
    fg = exp89.exp84_bundle_from_row(sources.get((dataset_name, "family_guard_v1")), n)
    rocket = bundles["rocket_exp40"]
    no_alert = not exp89_indices
    repair = set()
    if no_alert and not tiny_train:
        if fg["top1"] is not None and fg["train_exceed_rate"] <= 0.015 and fg["top1_threshold_margin"] >= 0.0:
            repair.add(fg["top1"])
        elif rocket.get("train_exceed_rate", 1.0) <= 0.02:
            top = exp90.first_top_index_from_bundle(rocket)
            if top is not None:
                repair.add(top)
    return set(exp89_indices) | repair


def select_exp93_validated(record, bundles, exp90_indices, sources, dataset_name):
    train_normal_count = len(record["train_series"])
    tiny_train = train_normal_count <= 10
    sparse_alert = len(exp90_indices) <= 1
    rocket_order = exp93.bundle_order(bundles["rocket_exp40"])
    exp55_order = exp93.bundle_order(bundles["exp55_best"])
    exp56_order = exp93.bundle_order(bundles["exp56_best"])
    fg_row = sources.get((dataset_name, "family_guard_v1"))
    cap3_row = sources.get((dataset_name, "count_cap_3pct"))
    fg_order = exp93.exp84_order(fg_row)
    cap3_order = exp93.exp84_order(cap3_row)
    if exp89.as_float(fg_row.get("train_exceed_rate") if fg_row else None, 1.0) > 0.015:
        fg_order = []
    if exp89.as_float(cap3_row.get("train_exceed_rate") if cap3_row else None, 1.0) > 0.02:
        cap3_order = []
    rank_maps = {
        "rocket": exp93.rank_map(rocket_order),
        "exp55": exp93.rank_map(exp55_order),
        "exp56": exp93.rank_map(exp56_order),
        "exp84_fg": exp93.rank_map(fg_order),
        "exp84_cap3": exp93.rank_map(cap3_order),
    }
    weights = {"rocket": 0.36, "exp55": 0.22, "exp56": 0.22, "exp84_fg": 0.12, "exp84_cap3": 0.08}
    pool = exp93.candidate_pool(rocket_order[:8], exp55_order[:8], exp56_order[:8], fg_order[:8], cap3_order[:8])
    scored = exp93.score_candidates(pool | set(exp90_indices), rank_maps, weights)
    top_candidate = exp93.top_non_base(scored, set(exp90_indices))
    weak_base = exp93.weak_base_indices(set(exp90_indices), scored)
    top_info = scored.get(top_candidate, {}) if top_candidate is not None else {}
    top_support = int(top_info.get("support", 0))
    top_best_rank = int(top_info.get("best_rank", 99))
    base_best = max([scored.get(index, {}).get("score", 0.0) for index in exp90_indices] or [0.0])
    score_gain = float(top_info.get("score", 0.0)) - base_best
    selected = set(exp90_indices)
    if not tiny_train and sparse_alert and weak_base and top_candidate is not None:
        if top_support >= 2 and top_best_rank <= 3 and score_gain >= 0.04:
            selected = (selected - weak_base) | {top_candidate}
    return selected


def review_and_priority(X_train, X_test, exp93_indices, high, standard):
    X_train = z_normalize(X_train).astype(np.float32)
    X_test = z_normalize(X_test).astype(np.float32)
    a_train, a_test, b_train, b_test, _, _ = block_base.block_scores(X_train, X_test)
    a_threshold, _, _ = count_cap_threshold(a_train, 0.015)
    b_threshold, _, _ = count_cap_threshold(b_train, 0.01)
    candidates = set(np.flatnonzero(a_test > a_threshold).astype(int).tolist())
    candidates &= set(np.flatnonzero(b_test > b_threshold).astype(int).tolist())
    candidates -= exp93_indices
    review = set()
    if len(X_train) > 10 and candidates:
        review = {max(candidates, key=lambda index: (float(b_test[index]), -index))}
    priority = set()
    if review and not high and standard:
        c_indices, _, _, _, _, _ = exp135.block_c_candidates(X_train, X_test)
        priority = review & c_indices
    return review, priority


def metrics_for_lanes(y_test, lanes):
    out = {}
    for label, indices in (("hard", lanes["hard"]), ("standard_review", lanes["standard_review"]), ("priority_review", lanes["priority_review"])):
        metrics = exp137.tier_metrics(y_test, indices)
        out.update({f"{label}_{key}": value for key, value in metrics.items()})
    combined = exp137.tier_metrics(y_test, lanes["hard"] | lanes["standard_review"] | lanes["priority_review"])
    out.update({f"combined_{key}": value for key, value in combined.items()})
    out.update(
        {
            "hard_alert_count": len(lanes["hard"]),
            "standard_review_count": len(lanes["standard_review"]),
            "priority_review_count": len(lanes["priority_review"]),
        }
    )
    return out


def lane_jaccard(first, second):
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def policy_result(dataset_name, X_train, X_test, y_test, record, bundles, base_row, sources, block_b):
    exp89_indices = select_exp89_best(record, y_test, bundles, base_row, sources, dataset_name)
    exp90_indices = select_exp90_operational(record, y_test, bundles, exp89_indices, sources, dataset_name)
    exp93_indices = select_exp93_validated(record, bundles, exp90_indices, sources, dataset_name)
    high = exp93_indices & block_b
    standard = exp93_indices - block_b
    review, priority = review_and_priority(X_train, X_test, exp93_indices, high, standard)
    lanes = build_tiers(high, standard, priority, len(y_test))
    return exp89_indices, exp90_indices, exp93_indices, lanes


def run_one(args):
    dataset_name, base_row, sources, block_b, baseline = args
    record, y_test, bundles = load_candidate_predictions(dataset_name, threshold_rates=CALIBRATION_PROFILES["relaxed_15pct"])
    X_train, X_test, _ = load_dataset_data(dataset_name)
    current = policy_result(dataset_name, X_train, X_test, y_test, record, bundles, base_row, sources, block_b)
    neutral_sources = replace_family_guard_with_cap2(sources, dataset_name)
    neutral = policy_result(dataset_name, X_train, X_test, y_test, record, bundles, base_row, neutral_sources, block_b)
    current_exp89, current_exp90, current_exp93, current_lanes = current
    neutral_exp89, neutral_exp90, neutral_exp93, neutral_lanes = neutral
    expected_hard = parse_indices(baseline.get("hard_alert_indices"))
    expected_standard = parse_indices(baseline.get("standard_review_indices"))
    expected_priority = parse_indices(baseline.get("priority_review_indices"))
    return {
        "dataset_name": dataset_name,
        "family": record.get("family", ""),
        "test_size": len(y_test),
        "current_exp89_indices": format_indices(current_exp89),
        "neutral_exp89_indices": format_indices(neutral_exp89),
        "current_exp90_indices": format_indices(current_exp90),
        "neutral_exp90_indices": format_indices(neutral_exp90),
        "current_exp93_indices": format_indices(current_exp93),
        "neutral_exp93_indices": format_indices(neutral_exp93),
        "current_hard_indices": format_indices(current_lanes["hard"]),
        "neutral_hard_indices": format_indices(neutral_lanes["hard"]),
        "current_standard_indices": format_indices(current_lanes["standard_review"]),
        "neutral_standard_indices": format_indices(neutral_lanes["standard_review"]),
        "current_priority_indices": format_indices(current_lanes["priority_review"]),
        "neutral_priority_indices": format_indices(neutral_lanes["priority_review"]),
        "current_matches_exp137": int(
            current_lanes["hard"] == expected_hard
            and current_lanes["standard_review"] == expected_standard
            and current_lanes["priority_review"] == expected_priority
        ),
        "hard_exact_match": int(current_lanes["hard"] == neutral_lanes["hard"]),
        "standard_exact_match": int(current_lanes["standard_review"] == neutral_lanes["standard_review"]),
        "priority_exact_match": int(current_lanes["priority_review"] == neutral_lanes["priority_review"]),
        "hard_jaccard": lane_jaccard(current_lanes["hard"], neutral_lanes["hard"]),
        "standard_jaccard": lane_jaccard(current_lanes["standard_review"], neutral_lanes["standard_review"]),
        "priority_jaccard": lane_jaccard(current_lanes["priority_review"], neutral_lanes["priority_review"]),
        **{f"current_{key}": value for key, value in metrics_for_lanes(y_test, current_lanes).items()},
        **{f"neutral_{key}": value for key, value in metrics_for_lanes(y_test, neutral_lanes).items()},
    }


def summarize(rows):
    totals = lambda prefix, key: int(sum(float(row[f"{prefix}_{key}"]) for row in rows))
    mean = lambda prefix, key: sum(float(row[f"{prefix}_{key}"]) for row in rows) / max(1, len(rows))
    out = {"experiment_id": EXPERIMENT_ID, "datasets": len(rows), "scope": "B1 common-support retrospective counterfactual"}
    for prefix in ("current", "neutral"):
        tp = totals(prefix, "hard_tp")
        fp = totals(prefix, "hard_fp")
        out.update(
            {
                f"{prefix}_hard_alerts": totals(prefix, "hard_alert_count"),
                f"{prefix}_hard_tp": tp,
                f"{prefix}_hard_fp": fp,
                f"{prefix}_hard_precision": tp / max(1, tp + fp),
                f"{prefix}_mean_hard_f1": mean(prefix, "hard_f1"),
                f"{prefix}_standard_review_candidates": totals(prefix, "standard_review_count"),
                f"{prefix}_priority_review_candidates": totals(prefix, "priority_review_count"),
            }
        )
    out.update(
        {
            "current_exp137_match_datasets": sum(int(row["current_matches_exp137"]) for row in rows),
            "hard_changed_datasets": sum(not int(row["hard_exact_match"]) for row in rows),
            "standard_changed_datasets": sum(not int(row["standard_exact_match"]) for row in rows),
            "priority_changed_datasets": sum(not int(row["priority_exact_match"]) for row in rows),
        }
    )
    return out


def run(dataset_limit=None, output_dir=OUTPUT_DIR):
    sources = source_maps()
    names = common_support_names(sources)
    if dataset_limit:
        names = names[:dataset_limit]
    base_primary, _ = exp89.load_base_rows()
    _, block_b_rows = exp133.load_maps()
    baseline = {row["dataset_name"]: row for row in read_rows(EXP137_PATH)}
    rows, errors = [], []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(run_one, (name, base_primary[name], sources, parse_indices(block_b_rows[name].get("selected_indices")), baseline[name])): name
            for name in names
        }
        for completed, future in enumerate(as_completed(futures), 1):
            name = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                errors.append((name, repr(exc)))
            if completed % 10 == 0 or completed == len(names):
                print(f"Progress: [{completed:3d}/{len(names):3d}] rows={len(rows)} errors={len(errors)}", flush=True)
    rows.sort(key=lambda row: row["dataset_name"])
    output_dir = Path(output_dir)
    write_rows(output_dir / "b1_family_neutral_common_support_results.csv", rows)
    summary = summarize(rows)
    summary["errors"] = errors
    (output_dir / "b1_family_neutral_common_support_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if errors or len(rows) != len(names):
        raise SystemExit(f"B1 coverage failure rows={len(rows)}/{len(names)} errors={errors[:3]}")
    if summary["current_exp137_match_datasets"] != len(rows):
        raise SystemExit("B1 current-policy replay does not match stored Exp137 baseline")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_limit, args.output_dir), indent=2, sort_keys=True))
