from __future__ import annotations

import argparse
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
from run_experiment_93_nonpos_candidate_reranker import candidate_pool, exp84_order, rank_map, score_candidates, sorted_candidates
from run_experiment_118_rocket512_knn3_exp93_source_probe import (
    deterministic_order,
    guarded_exp84_orders,
    make_rocket512_bundle,
    read_dict_rows,
)


DATA_DIR = Path("/Users/minho/Documents/Dataset")
EXPERIMENT_ID = "experiment_128_rocket512_only_review_selector"
VALIDATED_EXP93_PATH = DATA_DIR / "experiment_119a_exp93_rank_order_validation_results.csv"
EXP87_PATH = DATA_DIR / "experiment_87_exp84_index_diagnostics_results.csv"
STDOUT_LOG = DATA_DIR / f"{EXPERIMENT_ID}_stdout.log"
WORKERS = int(os.environ.get("RANK_EXPERIMENT_WORKERS", "6"))
MAX_RANK = 12
WEIGHTS = {"rocket512": 0.36, "exp55": 0.22, "exp56": 0.22, "exp84_fg": 0.12, "exp84_cap3": 0.08}


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_maps():
    hard_rows = {
        row["dataset_name"]: row
        for row in read_dict_rows(VALIDATED_EXP93_PATH)
        if row.get("selector_name") == "exp93_rank_order_validated"
    }
    exp87 = {}
    for row in read_dict_rows(EXP87_PATH):
        if row.get("config_name") in {"aeon_mrh_mr1024_hk4_g32_prune1024_stable_tail_local_gap_knn3"}:
            exp87[(row["dataset_name"], row["threshold_method"])] = row
    if len(hard_rows) != 1117 or not exp87:
        raise SystemExit(f"validated hard coverage={len(hard_rows)} exp87={len(exp87)}")
    return hard_rows, exp87


def review_metrics(y_test, hard_indices, review_indices):
    hard, review = set(hard_indices), set(review_indices)
    combined = hard | review
    true = {idx for idx, value in enumerate(y_test) if int(value) == 1}
    review_only = review - hard
    review_tp = len(review_only & true)
    review_fp = len(review_only - true)
    combined_tp = len(combined & true)
    combined_fp = len(combined - true)
    combined_fn = len(true - combined)
    denominator = 2 * combined_tp + combined_fp + combined_fn
    return {
        "review_candidate_count": len(review_only),
        "review_tp": review_tp,
        "review_fp": review_fp,
        "combined_tp": combined_tp,
        "combined_fp": combined_fp,
        "combined_fn": combined_fn,
        "combined_f1": (2 * combined_tp / denominator) if denominator else 0.0,
        "review_hit": int(review_tp > 0),
        "combined_zero_f1": int((2 * combined_tp / denominator) == 0.0) if denominator else 1,
    }


def other_agreement(index, maps, max_other_rank):
    names = ("exp55", "exp56", "exp84_fg", "exp84_cap3")
    agreeing = [name for name in names if maps[name].get(index, MAX_RANK + 1) <= max_other_rank]
    return agreeing


def pick_review(scored, maps, hard, rocket512_train_exceed, *, max_rocket_rank, max_other_rank, min_other_support):
    if rocket512_train_exceed > 0.015:
        return set(), {"reason": "blocked: rocket512 train-normal exceed rate > 1.5%", "candidate": "", "support": 0, "other_support": 0}
    for index in sorted_candidates(scored):
        if index in hard or maps["rocket512"].get(index, MAX_RANK + 1) > max_rocket_rank:
            continue
        agreeing = other_agreement(index, maps, max_other_rank)
        if len(agreeing) < min_other_support:
            continue
        info = scored[index]
        return {index}, {
            "reason": "512-only review candidate: ROCKET-512 plus independent source agreement",
            "candidate": index,
            "support": int(info["support"]),
            "other_support": len(agreeing),
            "other_sources": ",".join(agreeing),
            "rocket512_rank": int(maps["rocket512"][index]),
            "candidate_score": float(info["score"]),
        }
    return set(), {"reason": "no 512-only candidate passed agreement guard", "candidate": "", "support": 0, "other_support": 0}


def run_one(args):
    dataset_name, hard_row, exp87 = args
    record, y_test, bundles = load_candidate_predictions(dataset_name, threshold_rates=CALIBRATION_PROFILES["relaxed_15pct"])
    rocket512 = make_rocket512_bundle(dataset_name, y_test)
    rocket512_order = deterministic_order(rocket512, MAX_RANK)
    exp55_order = deterministic_order(bundles["exp55_best"], MAX_RANK)
    exp56_order = deterministic_order(bundles["exp56_best"], MAX_RANK)
    exp84_fg_order, exp84_cap3_order = guarded_exp84_orders(dataset_name, exp87)
    maps = {
        "rocket512": rank_map(rocket512_order),
        "exp55": rank_map(exp55_order),
        "exp56": rank_map(exp56_order),
        "exp84_fg": rank_map(exp84_fg_order[:MAX_RANK]),
        "exp84_cap3": rank_map(exp84_cap3_order[:MAX_RANK]),
    }
    pool = candidate_pool(rocket512_order, exp55_order, exp56_order, exp84_fg_order[:MAX_RANK], exp84_cap3_order[:MAX_RANK])
    scored = score_candidates(pool, maps, WEIGHTS, max_rank=MAX_RANK)
    hard = parse_indices(hard_row.get("selected_indices"))
    tiny_train = len(record["train_series"]) <= 10
    train_exceed = as_float(rocket512.get("train_exceed_rate"), 1.0)
    configs = [
        ("review_512_top1_agree_top3", 1, 3, 1),
        ("review_512_top3_agree_top3", 3, 3, 1),
        ("review_512_top3_two_source_agree", 3, 3, 2),
    ]
    rows = []
    for selector, max_rocket_rank, max_other_rank, min_other_support in configs:
        if tiny_train:
            review, diag = set(), {"reason": "blocked: tiny train-normal set", "candidate": "", "support": 0, "other_support": 0}
        else:
            review, diag = pick_review(
                scored, maps, hard, train_exceed,
                max_rocket_rank=max_rocket_rank,
                max_other_rank=max_other_rank,
                min_other_support=min_other_support,
            )
        hard_metrics = evaluate_indices(y_test, rocket512["test_scores"], hard)
        row = dict(hard_row)
        row.update({
            "experiment_id": EXPERIMENT_ID,
            "dataset_name": dataset_name,
            "family": record["family"],
            "config_name": selector,
            "selector_name": selector,
            "selector_reason": diag.pop("reason"),
            "score_source_name": "rocket_512_knn3_local_gap",
            "threshold_method": "review_lane",
            "score_family": "rocket512_only_review_selector",
            "sequence_length": len(record["test_series"][0]) if len(record["test_series"]) else "",
            "test_size": len(y_test),
            "anomaly_count": int(np.sum(y_test)),
            "selected_indices": format_indices(hard),
            "review_candidate_indices": format_indices(review),
            "predicted_count": hard_metrics["predicted_count"],
            "tp": hard_metrics["tp"],
            "fp": hard_metrics["fp"],
            "fn": hard_metrics["fn"],
            "f1": hard_metrics["f1"],
            "auc_roc": hard_metrics["auc_roc"],
            "auc_pr": hard_metrics["auc_pr"],
            "oracle_f1": hard_metrics["oracle_f1"],
            "train_exceed_rate": train_exceed,
            "train_normal_count": len(record["train_series"]),
            "tiny_train": int(tiny_train),
            "uses_rocket256": 0,
            "rocket512_top12": format_indices(rocket512_order),
            **diag,
            **review_metrics(y_test, hard, review),
        })
        rows.append(row)
    return rows


def summarize(rows):
    output = []
    for selector in sorted({row["selector_name"] for row in rows}):
        subset = [row for row in rows if row["selector_name"] == selector]
        values = lambda key: [as_float(row.get(key)) for row in subset]
        combined = values("combined_f1")
        output.append({
            "experiment_id": EXPERIMENT_ID,
            "selector_name": selector,
            "config_name": selector,
            "threshold_method": "review_lane",
            "num_datasets": len(subset),
            "mean_hard_f1": float(np.mean(values("f1"))),
            "hard_zero_f1_count": sum(value == 0.0 for value in values("f1")),
            "mean_hard_fp": float(np.mean(values("fp"))),
            "mean_combined_f1": float(np.mean(combined)),
            "combined_zero_f1_count": sum(value == 0.0 for value in combined),
            "mean_review_candidate_count": float(np.mean(values("review_candidate_count"))),
            "review_candidate_datasets": sum(as_float(row.get("review_candidate_count")) > 0 for row in subset),
            "review_hit_datasets": sum(as_float(row.get("review_hit")) > 0 for row in subset),
            "mean_review_tp": float(np.mean(values("review_tp"))),
            "mean_review_fp": float(np.mean(values("review_fp"))),
            "mean_combined_fp": float(np.mean(values("combined_fp"))),
            "tiny_train_datasets": sum(as_float(row.get("tiny_train")) > 0 for row in subset),
            "rocket256_used_rows": sum(as_float(row.get("uses_rocket256")) > 0 for row in subset),
        })
    return sorted(output, key=lambda row: (row["mean_combined_f1"], -row["mean_review_fp"]), reverse=True)


def run_experiment(dataset_limit=None):
    hard_rows, exp87 = load_maps()
    names = sorted(hard_rows)
    if dataset_limit:
        names = names[:dataset_limit]
    rows, errors = [], []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(run_one, (name, hard_rows[name], exp87)): name for name in names}
        for done, future in enumerate(as_completed(futures), 1):
            name = futures[future]
            try:
                rows.extend(future.result())
            except Exception as exc:
                errors.append((name, repr(exc)))
                print(f"ERROR dataset={name} error={exc!r}", flush=True)
            if done % 25 == 0 or done == len(names):
                print(f"Progress: [{done:4d}/{len(names):4d}] rows={len(rows)} last={name} errors={len(errors)}", flush=True)
    expected = len(names) * 3
    if errors or len(rows) != expected:
        raise SystemExit(f"coverage failure {len(rows)}/{expected} errors={errors[:5]}")
    write_csv(results_path(EXPERIMENT_ID), rows)
    summary = summarize(rows)
    write_csv(summary_path(EXPERIMENT_ID), summary)
    print(f"{EXPERIMENT_ID} finished rows={len(rows)} datasets={len(names)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-limit", type=int)
    run_experiment(parser.parse_args().dataset_limit)
