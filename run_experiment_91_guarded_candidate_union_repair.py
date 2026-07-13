from __future__ import annotations

import csv
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Set

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
from run_experiment_89_74d_with_exp84_candidate import (
    EXP87_CONFIG,
    as_float,
    exp84_bundle_from_row,
    format_indices,
    parse_indices,
)


DATA_DIR = Path("/Users/minho/Documents/Dataset")
EXPERIMENT_ID = "experiment_91_guarded_candidate_union_repair"
EXP89_PATH = DATA_DIR / "experiment_89_74d_with_exp84_candidate_results.csv"
EXP90_PATH = DATA_DIR / "experiment_90_zero_f1_repair_selector_results.csv"
EXP87_PATH = DATA_DIR / "experiment_87_exp84_index_diagnostics_results.csv"
STDOUT_LOG = DATA_DIR / f"{EXPERIMENT_ID}_stdout.log"
EXP89_BEST_SELECTOR = "exp84_four_model_fg_plus_noalert_repair"
EXP90_SAFE_SELECTOR = "noalert_top1_train_safe_repair"
EXP90_UNION_SELECTOR = "candidate_union_rerank_repair"
WORKERS = int(__import__("os").environ.get("EXP91_WORKERS", "4"))


def read_dict_rows(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_reference_rows():
    exp89_rows = read_dict_rows(EXP89_PATH)
    exp90_rows = read_dict_rows(EXP90_PATH)
    exp89 = {row["dataset_name"]: row for row in exp89_rows if row.get("selector_name") == EXP89_BEST_SELECTOR}
    exp90_safe = {row["dataset_name"]: row for row in exp90_rows if row.get("selector_name") == EXP90_SAFE_SELECTOR}
    exp90_union = {row["dataset_name"]: row for row in exp90_rows if row.get("selector_name") == EXP90_UNION_SELECTOR}
    if set(exp89) != set(exp90_safe) or set(exp89) != set(exp90_union):
        raise SystemExit("Reference coverage mismatch")
    return exp89, exp90_safe, exp90_union


def load_exp87_rows():
    rows = read_dict_rows(EXP87_PATH)
    out = {}
    for row in rows:
        if row.get("config_name") != EXP87_CONFIG:
            continue
        out[(row["dataset_name"], row["threshold_method"])] = row
    return out


def top_indices_from_bundle(bundle, count: int) -> Set[int]:
    return set(top_score_indices(bundle, max(1, int(count))))


def top_indices_from_exp84(exp84_bundle, count: int) -> Set[int]:
    top = set(exp84_bundle.get("top", set()))
    if not top:
        one = exp84_bundle.get("top1")
        return {one} if one is not None else set()
    return set(sorted(top)[: max(1, int(count))])


def first_top_index_from_bundle(bundle) -> Optional[int]:
    top = top_score_indices(bundle, 1)
    if not top:
        return None
    scores = bundle["test_scores"]
    return sorted(top, key=lambda idx: scores[idx], reverse=True)[0]


def cap_union(candidates, score_bundle, budget: int) -> Set[int]:
    pool = set().union(*(set(s) for s in candidates))
    return cap_indices_count(pool, score_bundle, budget) if pool else set()


def tail_indices(n: int, rate=0.02, minimum=1, maximum=5) -> Set[int]:
    count = min(maximum, max(minimum, int(math.ceil(n * rate))))
    return set(range(max(0, n - count), n))


def passthrough_row(source_row, selector_name, reason, extras=None):
    out = dict(source_row)
    out["experiment_id"] = EXPERIMENT_ID
    out["selector_name"] = selector_name
    out["config_name"] = selector_name
    out["threshold_method"] = "selector"
    out["selector_reason"] = reason
    if extras:
        out.update(extras)
    return out


def row_with_metrics(dataset_name, record, y_test, bundles, base_row, selector_name, indices, reason, extras=None):
    source = bundles["rocket_exp40"]
    metrics = evaluate_indices(y_test, source["test_scores"], indices)
    out = dict(base_row)
    out.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "dataset_name": dataset_name,
            "family": record["family"],
            "config_name": selector_name,
            "selector_name": selector_name,
            "threshold_method": "selector",
            "selector_reason": reason,
            "score_source_name": "rocket_exp40",
            "score_family": "guarded_candidate_union_repair",
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
            "train_exceed_rate": source.get("train_exceed_rate", base_row.get("train_exceed_rate", "")),
            "rocket_predicted_count": len(bundles["rocket_exp40"]["indices"]),
            "exp55_predicted_count": len(bundles["exp55_best"]["indices"]),
            "exp56_predicted_count": len(bundles["exp56_best"]["indices"]),
        }
    )
    if extras:
        out.update(extras)
    return out


def choose_rows_for_dataset(args):
    dataset_name, exp89_row, exp90_safe_row, exp90_union_row, exp87_rows = args
    record, y_test, bundles = load_candidate_predictions(dataset_name, threshold_rates=CALIBRATION_PROFILES["relaxed_15pct"])
    n = len(y_test)
    base_indices = parse_indices(exp89_row.get("selected_indices"))
    tiny_train = int(as_float(exp89_row.get("train_normal_count"), len(record["train_series"]))) <= 10
    large_case = is_large_data_case(record, y_test)
    no_alert = not base_indices
    sparse_alert = len(base_indices) == 1
    budget1 = large_data_budget(y_test, rate=0.01, minimum=1, maximum=3)
    budget2 = large_data_budget(y_test, rate=0.02, minimum=2, maximum=5)
    rocket = bundles["rocket_exp40"]
    exp55 = bundles["exp55_best"]
    exp56 = bundles["exp56_best"]
    exp84_fg = exp84_bundle_from_row(exp87_rows.get((dataset_name, "family_guard_v1")), n)
    exp84_cap3 = exp84_bundle_from_row(exp87_rows.get((dataset_name, "count_cap_3pct")), n)
    exp84_train_safe = exp84_fg["train_exceed_rate"] <= 0.015
    exp84_margin_safe = exp84_fg["top1_threshold_margin"] >= 0.0
    rocket_train_safe = rocket.get("train_exceed_rate", 1.0) <= 0.02
    tail = tail_indices(n)

    top_union_candidates = [
        top_indices_from_bundle(rocket, budget2),
        top_indices_from_bundle(exp55, budget2),
        top_indices_from_bundle(exp56, budget2),
        top_indices_from_exp84(exp84_fg, budget2) if exp84_train_safe else set(),
        top_indices_from_exp84(exp84_cap3, budget2) if exp84_cap3["train_exceed_rate"] <= 0.02 else set(),
    ]
    agreed = indices_at_least(2, top_union_candidates)
    union_budget = budget1 if not large_case else budget2
    union_repair = cap_union([agreed], rocket, union_budget)
    if no_alert and not union_repair:
        union_repair = cap_union(top_union_candidates, rocket, 1)
    union_repair = union_repair - base_indices
    tail_repair = union_repair & tail

    rocket_top1 = first_top_index_from_bundle(rocket)
    exp84_top1 = exp84_fg.get("top1")
    top1_repair = set()
    if no_alert and not tiny_train:
        if exp84_top1 is not None and exp84_train_safe and exp84_margin_safe:
            top1_repair.add(exp84_top1)
        elif rocket_top1 is not None and rocket_train_safe:
            top1_repair.add(rocket_top1)

    noalert_union = set()
    if no_alert and not tiny_train:
        noalert_union = union_repair if union_repair else top1_repair

    sparse_tail_add = set()
    if sparse_alert and not tiny_train:
        sparse_tail_add = tail_repair

    sparse_tail_replace = set(base_indices)
    replace_added = set()
    replaced_count = 0
    if sparse_alert and not tiny_train and tail_repair:
        current = next(iter(base_indices))
        candidate = sorted(tail_repair)[-1]
        if current not in tail and candidate in tail:
            sparse_tail_replace = {candidate}
            replace_added = {candidate}
            replaced_count = 1

    noalert_plus_sparse_tail = base_indices | noalert_union | sparse_tail_add
    noalert_plus_replace = (base_indices | noalert_union) if not replaced_count else sparse_tail_replace

    rows = [
        passthrough_row(
            exp89_row,
            "reference_exp89_best",
            "reference: Exp89 best",
            {"repair_added_count": 0, "tiny_train": int(tiny_train), "large_data_case": int(large_case)},
        ),
        passthrough_row(
            exp90_safe_row,
            "reference_exp90_noalert_top1",
            "reference: Exp90 no-alert top1 train-safe repair",
            {"repair_added_count": as_float(exp90_safe_row.get("repair_added_count")), "tiny_train": int(tiny_train), "large_data_case": int(large_case)},
        ),
        passthrough_row(
            exp90_union_row,
            "reference_exp90_candidate_union",
            "reference: Exp90 candidate union rerank repair",
            {"repair_added_count": as_float(exp90_union_row.get("repair_added_count")), "tiny_train": int(tiny_train), "large_data_case": int(large_case)},
        ),
        row_with_metrics(
            dataset_name,
            record,
            y_test,
            bundles,
            exp89_row,
            "noalert_union_only",
            base_indices | noalert_union,
            "candidate-union/top1 repair only when Exp89 emits no alert",
            {"repair_added_count": len(noalert_union), "tiny_train": int(tiny_train), "large_data_case": int(large_case), "tail_repair_count": len(tail_repair), "replaced_count": 0},
        ),
        row_with_metrics(
            dataset_name,
            record,
            y_test,
            bundles,
            exp89_row,
            "noalert_plus_sparse_tail_add",
            noalert_plus_sparse_tail,
            "no-alert repair plus sparse-alert extension only when added candidate is in tail window",
            {"repair_added_count": len((noalert_union | sparse_tail_add) - base_indices), "tiny_train": int(tiny_train), "large_data_case": int(large_case), "tail_repair_count": len(tail_repair), "replaced_count": 0},
        ),
        row_with_metrics(
            dataset_name,
            record,
            y_test,
            bundles,
            exp89_row,
            "noalert_plus_sparse_tail_replace",
            noalert_plus_replace,
            "no-alert repair plus sparse-alert tail replacement when current alert is not in tail",
            {"repair_added_count": len(noalert_union | replace_added), "tiny_train": int(tiny_train), "large_data_case": int(large_case), "tail_repair_count": len(tail_repair), "replaced_count": replaced_count},
        ),
    ]
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
                "mean_oracle_f1": float(np.mean(vals("oracle_f1"))),
                "mean_repair_added_count": float(np.mean(vals("repair_added_count"))),
                "repair_used_datasets": sum(1 for row in subset if as_float(row.get("repair_added_count")) > 0),
                "mean_replaced_count": float(np.mean(vals("replaced_count"))),
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
    exp89, exp90_safe, exp90_union = load_reference_rows()
    exp87 = load_exp87_rows()
    datasets = sorted(exp89)
    if dataset_limit:
        datasets = datasets[: int(dataset_limit)]
    tasks = [(name, exp89[name], exp90_safe[name], exp90_union[name], exp87) for name in datasets]
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
