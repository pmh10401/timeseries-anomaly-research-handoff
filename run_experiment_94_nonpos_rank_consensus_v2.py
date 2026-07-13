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
from run_experiment_89_74d_with_exp84_candidate import (
    EXP87_CONFIG,
    as_float,
    format_indices,
    parse_indices,
)
from run_experiment_93_nonpos_candidate_reranker import (
    candidate_pool,
    exp84_order,
    load_exp87_rows,
    rank_map,
    score_candidates,
    sorted_candidates,
    bundle_order,
)


DATA_DIR = Path("/Users/minho/Documents/Dataset")
EXPERIMENT_ID = "experiment_94_nonpos_rank_consensus_v2"
EXP93_PATH = DATA_DIR / "experiment_93_nonpos_candidate_reranker_results.csv"
EXP93_OPERATING_SELECTOR = "nonpos_weak_alert_replace"
STDOUT_LOG = DATA_DIR / f"{EXPERIMENT_ID}_stdout.log"
WORKERS = int(os.environ.get("EXP94_WORKERS", "4"))


def read_dict_rows(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_exp93_operating_rows():
    rows = read_dict_rows(EXP93_PATH)
    out = {row["dataset_name"]: row for row in rows if row.get("selector_name") == EXP93_OPERATING_SELECTOR}
    if not out:
        raise SystemExit("Exp93 operating rows not found")
    return out


def passthrough_row(source_row, selector_name, reason, extras=None):
    out = dict(source_row)
    out["experiment_id"] = EXPERIMENT_ID
    out["selector_name"] = selector_name
    out["config_name"] = selector_name
    out["selector_reason"] = reason
    out["threshold_method"] = "selector"
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
            "selector_reason": reason,
            "score_source_name": "rocket_exp40",
            "threshold_method": "selector",
            "score_family": "nonpos_rank_consensus_v2",
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
        rocket_order[:10],
        exp55_order[:10],
        exp56_order[:10],
        exp84_fg_order[:10],
        exp84_cap3_order[:10],
    )
    scored = score_candidates(pool, rank_maps, weights, max_rank=16)
    return record, y_test, bundles, scored


def top_candidate(scored, base_indices, min_support, max_best_rank, min_gain):
    base_best = max([scored.get(idx, {}).get("score", 0.0) for idx in base_indices] or [0.0])
    for idx in sorted_candidates(scored):
        if idx in base_indices:
            continue
        info = scored[idx]
        gain = info["score"] - base_best
        if info["support"] >= min_support and info["best_rank"] <= max_best_rank and gain >= min_gain:
            return idx, gain
    return None, 0.0


def weak_indices(base_indices, scored, max_support, min_best_rank, max_score):
    out = set()
    for idx in base_indices:
        info = scored.get(idx, {"score": 0.0, "support": 0.0, "best_rank": 99.0})
        if info["support"] <= max_support and info["best_rank"] >= min_best_rank and info["score"] <= max_score:
            out.add(idx)
    return out


def choose_rows_for_dataset(args):
    dataset_name, base_row, exp87_rows = args
    record, y_test, bundles, scored = candidate_context(dataset_name, exp87_rows)
    base_indices = parse_indices(base_row.get("selected_indices"))
    train_normal_count = int(as_float(base_row.get("train_normal_count"), len(record["train_series"])))
    tiny_train = train_normal_count <= 10
    sparse = len(base_indices) <= 1

    rows = [
        passthrough_row(
            base_row,
            "baseline_exp93_operating",
            "control: Exp93 operating default",
            {"train_normal_count": train_normal_count, "tiny_train": int(tiny_train), "changed_count": 0},
        )
    ]

    configs = [
        ("replace_v2_strict", 3, 3, 0.10, 1, 6, 0.35),
        ("replace_v2_balanced", 2, 3, 0.12, 1, 6, 0.30),
        ("replace_v2_margin", 2, 2, 0.20, 1, 5, 0.40),
        ("replace_or_add_v2_strict", 4, 2, 0.18, 1, 6, 0.35),
    ]
    for name, min_support, max_rank, min_gain, weak_support, weak_rank, weak_score in configs:
        chosen = set(base_indices)
        changed = 0
        cand, gain = top_candidate(scored, base_indices, min_support, max_rank, min_gain)
        weak = weak_indices(base_indices, scored, weak_support, weak_rank, weak_score)
        if not tiny_train and sparse and cand is not None:
            if weak:
                chosen = (chosen - weak) | {cand}
                changed = len(weak) + int(cand not in base_indices)
            elif name.startswith("replace_or_add") and not base_indices:
                chosen.add(cand)
                changed = int(cand not in base_indices)
        rows.append(
            row_with_metrics(
                dataset_name,
                record,
                y_test,
                bundles,
                base_row,
                name,
                chosen,
                "non-position rank consensus v2; replace weak sparse alert only",
                {
                    "train_normal_count": train_normal_count,
                    "tiny_train": int(tiny_train),
                    "changed_count": changed,
                    "candidate_index": cand if cand is not None else "",
                    "candidate_gain": gain,
                    "candidate_support": scored.get(cand, {}).get("support", "") if cand is not None else "",
                    "candidate_best_rank": scored.get(cand, {}).get("best_rank", "") if cand is not None else "",
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
                "mean_oracle_f1": float(np.mean(vals("oracle_f1"))),
                "changed_datasets": sum(1 for row in subset if as_float(row.get("changed_count")) > 0),
                "tiny_train_datasets": sum(1 for row in subset if as_float(row.get("tiny_train")) > 0),
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
