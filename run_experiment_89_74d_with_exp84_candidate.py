from __future__ import annotations

import csv
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Mapping, Optional, Set, Tuple

import numpy as np

from run_experiment_60_62_rocket_imaging_selector_variants import (
    CALIBRATION_PROFILES,
    cap_indices_count,
    evaluate_indices,
    indices_at_least,
    is_large_data_case,
    large_data_budget,
    load_candidate_predictions,
    results_path,
    summary_path,
    top_score_indices,
)


DATA_DIR = Path("/Users/minho/Documents/Dataset")
EXPERIMENT_ID = "experiment_89_74d_with_exp84_candidate"
BASE_PATH = DATA_DIR / "experiment_74d_large_rank_review_tier_split_results.csv"
EXP87_PATH = DATA_DIR / "experiment_87_exp84_index_diagnostics_results.csv"
STDOUT_LOG = DATA_DIR / f"{EXPERIMENT_ID}_stdout.log"
BASE_PRIMARY_SELECTOR = "large_primary_rocket_guard_only"
BASE_REVIEW_SELECTOR = "large_primary_plus_review_limited"
EXP87_CONFIG = "aeon_mrh_mr1024_hk4_g32_prune1024_stable_tail_local_gap_knn3"
WORKERS = int(__import__("os").environ.get("EXP89_WORKERS", "4"))


def parse_indices(value) -> Set[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return set()
    return {int(float(item)) for item in text.split()}


def format_indices(indices) -> str:
    return " ".join(str(int(idx)) for idx in sorted(indices))


def as_float(value, default=0.0):
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def first_top_index(row: Optional[Mapping]) -> Optional[int]:
    if row is None:
        return None
    text = str(row.get("top_score_indices", "")).strip()
    if not text or text.lower() == "nan":
        return None
    return int(float(text.split()[0]))


def read_dict_rows(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_base_rows():
    rows = read_dict_rows(BASE_PATH)
    primary = {row["dataset_name"]: row for row in rows if row.get("selector_name") == BASE_PRIMARY_SELECTOR}
    review = {row["dataset_name"]: row for row in rows if row.get("selector_name") == BASE_REVIEW_SELECTOR}
    if set(primary) != set(review):
        raise SystemExit("Exp74d primary/review coverage mismatch")
    return primary, review


def load_exp87_rows():
    rows = read_dict_rows(EXP87_PATH)
    out = {}
    for row in rows:
        if row.get("config_name") != EXP87_CONFIG:
            continue
        out[(row["dataset_name"], row["threshold_method"])] = row
    return out


def passthrough_row(source_row, selector_name, source_name, reason, extras=None):
    out = dict(source_row)
    out["experiment_id"] = EXPERIMENT_ID
    out["selector_name"] = selector_name
    out["config_name"] = selector_name
    out["selector_reason"] = reason
    out["threshold_method"] = "selector"
    out["score_source_name"] = source_name
    out["selected_source_experiment_id"] = source_row.get("experiment_id", "")
    out["selected_source_selector_name"] = source_row.get("selector_name", "")
    out["used_exp84_candidate"] = 0
    out["exp84_only_promoted_count"] = 0
    out["exp84_overlap_base_count"] = 0
    out["review_candidate_count"] = 0
    if extras:
        out.update(extras)
    return out


def computed_row(dataset_name, record, y_test, bundles, base_row, selector_name, indices, source_name, reason, extras=None):
    source_bundle = bundles.get(source_name, bundles["rocket_exp40"])
    metrics = evaluate_indices(y_test, source_bundle["test_scores"], indices)
    out = dict(base_row)
    out.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "dataset_name": dataset_name,
            "family": record["family"],
            "config_name": selector_name,
            "selector_name": selector_name,
            "selector_reason": reason,
            "score_source_name": source_name,
            "threshold_method": "selector",
            "score_family": "exp84_candidate_selector",
            "sequence_length": len(record["test_series"][0]) if len(record["test_series"]) else "",
            "test_size": len(y_test),
            "anomaly_count": int(np.sum(y_test)),
            "selected_indices": format_indices(indices),
            "predicted_count": metrics["predicted_count"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "auc_roc": metrics["auc_roc"],
            "auc_pr": metrics["auc_pr"],
            "f1": metrics["f1"],
            "oracle_f1": metrics["oracle_f1"],
            "train_exceed_rate": source_bundle.get("train_exceed_rate", base_row.get("train_exceed_rate", "")),
            "rocket_predicted_count": len(bundles["rocket_exp40"]["indices"]),
            "exp55_predicted_count": len(bundles["exp55_best"]["indices"]),
            "exp56_predicted_count": len(bundles["exp56_best"]["indices"]),
        }
    )
    if extras:
        out.update(extras)
    return out


def exp84_bundle_from_row(row, n):
    if row is None:
        return {
            "indices": set(),
            "top": set(),
            "top1": None,
            "train_exceed_rate": 1.0,
            "top1_threshold_margin": -1.0,
            "top1_top2_margin": 0.0,
        }
    selected = parse_indices(row.get("selected_indices"))
    top_text = str(row.get("top_score_indices", "")).strip()
    top_order = [int(float(item)) for item in top_text.split()] if top_text and top_text.lower() != "nan" else []
    return {
        "indices": selected,
        "top": set(top_order[: max(1, min(8, n))]),
        "top1": top_order[0] if top_order else None,
        "train_exceed_rate": as_float(row.get("train_exceed_rate"), 1.0),
        "top1_threshold_margin": as_float(row.get("top1_threshold_margin"), -1.0),
        "top1_top2_margin": as_float(row.get("top1_top2_margin"), 0.0),
    }


def choose_rows_for_dataset(args):
    dataset_name, base_row, review_row, exp87_rows = args
    record, y_test, bundles = load_candidate_predictions(dataset_name, threshold_rates=CALIBRATION_PROFILES["relaxed_15pct"])
    n = len(y_test)
    large_case = is_large_data_case(record, y_test)
    base_indices = parse_indices(base_row.get("selected_indices"))
    review_indices = parse_indices(review_row.get("selected_indices"))
    rocket = bundles["rocket_exp40"]["indices"]
    exp55 = bundles["exp55_best"]["indices"]
    exp56 = bundles["exp56_best"]["indices"]
    exp84_fg = exp84_bundle_from_row(exp87_rows.get((dataset_name, "family_guard_v1")), n)
    exp84_cap3 = exp84_bundle_from_row(exp87_rows.get((dataset_name, "count_cap_3pct")), n)
    budget2 = large_data_budget(y_test, rate=0.01, minimum=2, maximum=5)
    budget3 = large_data_budget(y_test, rate=0.02, minimum=3, maximum=8)
    rocket_guard = top_score_indices(bundles["rocket_exp40"], max(budget3, budget2 * 2))
    four_sets_cap3 = [rocket, exp55, exp56, exp84_cap3["indices"]]
    four_sets_fg = [rocket, exp55, exp56, exp84_fg["indices"]]
    two_of_four_cap3 = indices_at_least(2, four_sets_cap3)
    two_of_four_fg = indices_at_least(2, four_sets_fg)
    three_of_four_fg = indices_at_least(3, four_sets_fg)
    exp84_only = exp84_fg["indices"] - (rocket | exp55 | exp56)
    exp84_overlap_base = exp84_fg["indices"] & base_indices

    primary_cap3 = cap_indices_count(two_of_four_cap3 & rocket_guard, bundles["rocket_exp40"], budget3)
    primary_fg = cap_indices_count(two_of_four_fg & rocket_guard, bundles["rocket_exp40"], budget3)
    primary_three = cap_indices_count(three_of_four_fg & rocket_guard, bundles["rocket_exp40"], budget3)
    review_pool = two_of_four_fg - primary_fg - base_indices
    review_limited = cap_indices_count(review_pool, bundles["rocket_exp40"], max(1, budget2)) if review_pool else set()

    rows = [
        passthrough_row(
            base_row,
            "baseline_74d_primary",
            "exp74d_primary",
            "control",
            {
                "used_exp84_candidate": 0,
                "exp84_overlap_base_count": len(exp84_overlap_base),
                "exp84_only_candidate_count": len(exp84_only),
            },
        ),
        passthrough_row(
            base_row,
            "exp84_confidence_boost_only",
            "exp74d_primary",
            "same alerts as Exp74d; Exp84 only annotates confidence",
            {
                "used_exp84_candidate": int(bool(exp84_overlap_base)),
                "exp84_overlap_base_count": len(exp84_overlap_base),
                "exp84_only_candidate_count": len(exp84_only),
            },
        ),
    ]

    strategies = {
        "exp84_four_model_cap3_rocket_guard": (
            base_indices if not large_case else (base_indices | primary_cap3),
            "rocket_exp40",
            f"large_{large_case}_two_of_four_cap3_rocket_guard_budget_{budget3}",
            primary_cap3,
        ),
        "exp84_four_model_fg_rocket_guard": (
            base_indices if not large_case else (base_indices | primary_fg),
            "rocket_exp40",
            f"large_{large_case}_two_of_four_fg_rocket_guard_budget_{budget3}",
            primary_fg,
        ),
        "exp84_four_model_fg_three_of_four": (
            base_indices if not large_case else (base_indices | primary_three),
            "rocket_exp40",
            f"large_{large_case}_three_of_four_fg_rocket_guard_budget_{budget3}",
            primary_three,
        ),
        "exp84_review_tier_limited": (
            base_indices if not large_case else (base_indices | review_limited),
            "rocket_exp40",
            f"large_{large_case}_exp84_review_tier_budget_{budget2}",
            review_limited,
        ),
    }
    top1 = exp84_fg["top1"]
    train_safe = exp84_fg["train_exceed_rate"] <= 0.015
    margin_safe = exp84_fg["top1_threshold_margin"] >= 0.0 and exp84_fg["top1_top2_margin"] >= 0.0
    noalert_repair = {top1} if not base_indices and top1 is not None and train_safe and margin_safe else set()
    strategies["exp84_top1_noalert_repair"] = (
        base_indices | noalert_repair,
        "rocket_exp40",
        "Exp84 top1 allowed only when Exp74d has no alert and Exp84 is train-safe",
        noalert_repair,
    )
    strategies["exp84_four_model_fg_plus_noalert_repair"] = (
        (base_indices if not large_case else (base_indices | primary_fg)) | noalert_repair,
        "rocket_exp40",
        f"large_{large_case}_two_of_four_fg_plus_noalert_repair",
        primary_fg | noalert_repair,
    )

    for selector_name, (indices, source, reason, added) in strategies.items():
        rows.append(
            computed_row(
                dataset_name,
                record,
                y_test,
                bundles,
                base_row,
                selector_name,
                set(indices),
                source,
                reason,
                {
                    "large_data_case": int(large_case),
                    "large_data_budget_1pct": budget2,
                    "large_data_budget_2pct": budget3,
                    "used_exp84_candidate": int(bool(added & exp84_fg["indices"]) or bool(added & exp84_cap3["indices"]) or bool(noalert_repair)),
                    "exp84_only_candidate_count": len(exp84_only),
                    "exp84_overlap_base_count": len(exp84_overlap_base),
                    "exp84_only_promoted_count": len(set(indices) & exp84_only),
                    "review_candidate_count": len(review_limited),
                    "exp84_fg_predicted_count": len(exp84_fg["indices"]),
                    "exp84_cap3_predicted_count": len(exp84_cap3["indices"]),
                },
            )
        )
    return rows


def summarize(rows):
    out = []
    for selector in sorted({row["selector_name"] for row in rows}):
        subset = [row for row in rows if row["selector_name"] == selector]
        vals = lambda key: [as_float(row.get(key)) for row in subset]
        f1s = vals("f1")
        by_family = {}
        for row in subset:
            by_family.setdefault(row["family"], []).append(as_float(row.get("f1")))
        out.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "selector_name": selector,
                "config_name": selector,
                "threshold_method": "selector",
                "num_datasets": len(subset),
                "num_families": len(by_family),
                "mean_auc_roc": float(np.mean(vals("auc_roc"))),
                "mean_auc_pr": float(np.mean(vals("auc_pr"))),
                "mean_f1": float(np.mean(f1s)),
                "median_f1": float(np.median(f1s)),
                "p25_f1": float(np.percentile(f1s, 25)),
                "zero_f1_count": sum(1 for value in f1s if value == 0.0),
                "ge_0_5_count": sum(1 for value in f1s if value >= 0.5),
                "family_macro_f1": float(np.mean([np.mean(v) for v in by_family.values()])),
                "mean_predicted_count": float(np.mean(vals("predicted_count"))),
                "mean_anomaly_count": float(np.mean(vals("anomaly_count"))),
                "mean_tp": float(np.mean(vals("tp"))),
                "mean_fp": float(np.mean(vals("fp"))),
                "mean_fn": float(np.mean(vals("fn"))),
                "mean_train_exceed_rate": float(np.mean(vals("train_exceed_rate"))),
                "mean_oracle_f1": float(np.mean(vals("oracle_f1"))),
                "exp84_used_datasets": sum(1 for row in subset if as_float(row.get("used_exp84_candidate")) > 0),
                "mean_exp84_only_promoted_count": float(np.mean(vals("exp84_only_promoted_count"))),
            }
        )
    return sorted(out, key=lambda row: (row["mean_f1"], -row["mean_fp"]), reverse=True)


def write_csv(path: Path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(dataset_limit=None):
    primary, review = load_base_rows()
    exp87 = load_exp87_rows()
    datasets = sorted(primary)
    if dataset_limit:
        datasets = datasets[: int(dataset_limit)]
    tasks = [(name, primary[name], review[name], exp87) for name in datasets]
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(choose_rows_for_dataset, task): task[0] for task in tasks}
        for idx, future in enumerate(as_completed(futures), 1):
            dataset_name = futures[future]
            rows.extend(future.result())
            if idx % 25 == 0 or idx == len(tasks):
                print(f"{EXPERIMENT_ID} progress {idx}/{len(tasks)} last={dataset_name}", flush=True)
    write_csv(results_path(EXPERIMENT_ID), rows)
    summary = summarize(rows)
    write_csv(summary_path(EXPERIMENT_ID), summary)
    STDOUT_LOG.write_text(f"{EXPERIMENT_ID} finished rows={len(rows)} datasets={len(datasets)}\n{summary[0] if summary else ''}\n")
    print(f"{EXPERIMENT_ID} finished rows={len(rows)} datasets={len(datasets)}")
    if summary:
        print(summary[0])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-limit", type=int, default=None)
    args = parser.parse_args()
    run_experiment(args.dataset_limit)
