from __future__ import annotations

import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from run_experiment_60_62_rocket_imaging_selector_variants import (
    CALIBRATION_PROFILES,
    evaluate_indices,
    load_candidate_predictions,
    results_path,
    summary_path,
)
from run_experiment_89_74d_with_exp84_candidate import as_float, format_indices, parse_indices
from run_experiment_93_nonpos_candidate_reranker import (
    bundle_order,
    candidate_pool,
    exp84_order,
    load_exp87_rows,
    rank_map,
    score_candidates,
    sorted_candidates,
)


DATA_DIR = Path("/Users/minho/Documents/Dataset")
EXPERIMENT_ID = "experiment_95_topk_review_tier"
EXP93_PATH = DATA_DIR / "experiment_93_nonpos_candidate_reranker_results.csv"
EXP93_OPERATING_SELECTOR = "nonpos_weak_alert_replace"
STDOUT_LOG = DATA_DIR / f"{EXPERIMENT_ID}_stdout.log"
WORKERS = int(os.environ.get("EXP95_WORKERS", "4"))


def read_dict_rows(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_exp93_operating_rows():
    rows = read_dict_rows(EXP93_PATH)
    out = {row["dataset_name"]: row for row in rows if row.get("selector_name") == EXP93_OPERATING_SELECTOR}
    if not out:
        raise SystemExit("Exp93 operating rows not found")
    return out


def review_metrics(y_test, hard_indices, review_indices):
    hard = set(hard_indices)
    review = set(review_indices)
    combined = hard | review
    true = {idx for idx, value in enumerate(y_test) if int(value) == 1}
    review_tp = len(review & true)
    review_fp = len(review - true)
    combined_tp = len(combined & true)
    combined_fp = len(combined - true)
    combined_fn = len(true - combined)
    denom = 2 * combined_tp + combined_fp + combined_fn
    combined_f1 = (2 * combined_tp / denom) if denom else 0.0
    return {
        "review_candidate_count": len(review),
        "review_tp": review_tp,
        "review_fp": review_fp,
        "combined_tp": combined_tp,
        "combined_fp": combined_fp,
        "combined_fn": combined_fn,
        "combined_f1": combined_f1,
        "review_hit": int(review_tp > 0),
        "combined_zero_f1": int(combined_f1 == 0.0),
    }


def candidate_context(dataset_name, exp87_rows):
    record, y_test, bundles = load_candidate_predictions(
        dataset_name,
        threshold_rates=CALIBRATION_PROFILES["relaxed_15pct"],
    )
    rocket_order = bundle_order(bundles["rocket_exp40"], 16)
    exp55_order = bundle_order(bundles["exp55_best"], 16)
    exp56_order = bundle_order(bundles["exp56_best"], 16)
    exp84_fg_row = exp87_rows.get((dataset_name, "family_guard_v1"))
    exp84_cap3_row = exp87_rows.get((dataset_name, "count_cap_3pct"))
    exp84_fg_order = exp84_order(exp84_fg_row, 16)
    exp84_cap3_order = exp84_order(exp84_cap3_row, 16)
    if as_float(exp84_fg_row.get("train_exceed_rate") if exp84_fg_row else None, 1.0) > 0.015:
        exp84_fg_order = []
    if as_float(exp84_cap3_row.get("train_exceed_rate") if exp84_cap3_row else None, 1.0) > 0.02:
        exp84_cap3_order = []
    rank_maps = {
        "rocket": rank_map(rocket_order),
        "exp55": rank_map(exp55_order),
        "exp56": rank_map(exp56_order),
        "exp84_fg": rank_map(exp84_fg_order),
        "exp84_cap3": rank_map(exp84_cap3_order),
    }
    weights = {
        "rocket": 0.34,
        "exp55": 0.22,
        "exp56": 0.22,
        "exp84_fg": 0.14,
        "exp84_cap3": 0.08,
    }
    pool = candidate_pool(
        rocket_order[:12],
        exp55_order[:12],
        exp56_order[:12],
        exp84_fg_order[:12],
        exp84_cap3_order[:12],
    )
    scored = score_candidates(pool, rank_maps, weights, max_rank=16)
    return record, y_test, bundles, scored


def pick_review(scored, hard_indices, limit, min_support, max_best_rank, min_score):
    out = []
    for idx in sorted_candidates(scored):
        if idx in hard_indices:
            continue
        info = scored[idx]
        if info["support"] >= min_support and info["best_rank"] <= max_best_rank and info["score"] >= min_score:
            out.append(idx)
        if len(out) >= limit:
            break
    return set(out)


def row_with_review(dataset_name, record, y_test, bundles, base_row, selector_name, review_indices, reason, extras=None):
    hard_indices = parse_indices(base_row.get("selected_indices"))
    hard_metrics = evaluate_indices(y_test, bundles["rocket_exp40"]["test_scores"], hard_indices)
    rm = review_metrics(y_test, hard_indices, review_indices)
    out = dict(base_row)
    out.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "dataset_name": dataset_name,
            "family": record["family"],
            "config_name": selector_name,
            "selector_name": selector_name,
            "selector_reason": reason,
            "score_source_name": "rocket_exp40",
            "threshold_method": "selector",
            "score_family": "topk_review_tier",
            "sequence_length": len(record["test_series"][0]) if len(record["test_series"]) else "",
            "test_size": len(y_test),
            "anomaly_count": int(np.sum(y_test)),
            "selected_indices": format_indices(hard_indices),
            "review_candidate_indices": format_indices(review_indices),
            "predicted_count": hard_metrics["predicted_count"],
            "tp": hard_metrics["tp"],
            "fp": hard_metrics["fp"],
            "fn": hard_metrics["fn"],
            "auc_roc": hard_metrics["auc_roc"],
            "auc_pr": hard_metrics["auc_pr"],
            "f1": hard_metrics["f1"],
            "oracle_f1": hard_metrics["oracle_f1"],
            "train_exceed_rate": bundles["rocket_exp40"].get("train_exceed_rate", base_row.get("train_exceed_rate", "")),
        }
    )
    out.update(rm)
    if extras:
        out.update(extras)
    return out


def choose_rows_for_dataset(args):
    dataset_name, base_row, exp87_rows = args
    record, y_test, bundles, scored = candidate_context(dataset_name, exp87_rows)
    hard_indices = parse_indices(base_row.get("selected_indices"))
    train_normal_count = int(as_float(base_row.get("train_normal_count"), len(record["train_series"])))
    tiny_train = train_normal_count <= 10
    rows = []
    configs = [
        ("review_top1_strict", 1, 3, 3, 0.45),
        ("review_top2_balanced", 2, 2, 5, 0.35),
        ("review_top3_broad", 3, 2, 8, 0.25),
        ("review_top5_diagnostic", 5, 1, 10, 0.15),
    ]
    for name, limit, min_support, max_rank, min_score in configs:
        review = set()
        if not tiny_train:
            review = pick_review(scored, hard_indices, limit, min_support, max_rank, min_score)
        rows.append(
            row_with_review(
                dataset_name,
                record,
                y_test,
                bundles,
                base_row,
                name,
                review,
                "review-tier candidates only; hard alert remains Exp93 operating default",
                {"train_normal_count": train_normal_count, "tiny_train": int(tiny_train)},
            )
        )
    return rows


def summarize(rows):
    out = []
    for selector in sorted({row["selector_name"] for row in rows}):
        subset = [row for row in rows if row["selector_name"] == selector]
        vals = lambda key: [as_float(row.get(key)) for row in subset]
        f1s = vals("f1")
        combined_f1s = vals("combined_f1")
        by_family = {}
        for row in subset:
            by_family.setdefault(row["family"], []).append(as_float(row.get("combined_f1")))
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
                "zero_f1_count": sum(1 for value in f1s if value == 0.0),
                "mean_combined_f1": float(np.mean(combined_f1s)),
                "combined_zero_f1_count": sum(1 for value in combined_f1s if value == 0.0),
                "family_macro_combined_f1": float(np.mean([np.mean(v) for v in by_family.values()])),
                "mean_predicted_count": float(np.mean(vals("predicted_count"))),
                "mean_review_candidate_count": float(np.mean(vals("review_candidate_count"))),
                "review_hit_datasets": sum(1 for row in subset if as_float(row.get("review_hit")) > 0),
                "mean_tp": float(np.mean(vals("tp"))),
                "mean_fp": float(np.mean(vals("fp"))),
                "mean_review_tp": float(np.mean(vals("review_tp"))),
                "mean_review_fp": float(np.mean(vals("review_fp"))),
                "mean_combined_tp": float(np.mean(vals("combined_tp"))),
                "mean_combined_fp": float(np.mean(vals("combined_fp"))),
                "mean_oracle_f1": float(np.mean(vals("oracle_f1"))),
                "tiny_train_datasets": sum(1 for row in subset if as_float(row.get("tiny_train")) > 0),
            }
        )
    return sorted(out, key=lambda row: (row["mean_combined_f1"], -row["mean_review_fp"]), reverse=True)


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
    base_rows = load_exp93_operating_rows()
    exp87 = load_exp87_rows()
    datasets = sorted(base_rows)
    if dataset_limit:
        datasets = datasets[: int(dataset_limit)]
    tasks = [(name, base_rows[name], exp87) for name in datasets]
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
