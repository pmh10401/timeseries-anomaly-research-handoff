from __future__ import annotations

import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Set

import numpy as np

from run_experiment_60_62_rocket_imaging_selector_variants import (
    CALIBRATION_PROFILES,
    evaluate_indices,
    load_candidate_predictions,
    results_path,
    summary_path,
    top_score_order,
)
from run_experiment_89_74d_with_exp84_candidate import (
    EXP87_CONFIG,
    as_float,
    format_indices,
    parse_indices,
)


DATA_DIR = Path("/Users/minho/Documents/Dataset")
EXPERIMENT_ID = "experiment_93_nonpos_candidate_reranker"
EXP90_PATH = DATA_DIR / "experiment_90_zero_f1_repair_selector_results.csv"
EXP87_PATH = DATA_DIR / "experiment_87_exp84_index_diagnostics_results.csv"
STDOUT_LOG = DATA_DIR / f"{EXPERIMENT_ID}_stdout.log"
EXP90_OPERATIONAL_SELECTOR = "noalert_top1_train_safe_repair"
EXP90_RESEARCH_SELECTOR = "candidate_union_rerank_repair"
WORKERS = int(os.environ.get("RANK_EXPERIMENT_WORKERS", os.environ.get("EXP93_WORKERS", "6")))
MAX_TOP = 12


def read_dict_rows(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_exp90_rows():
    rows = read_dict_rows(EXP90_PATH)
    operational = {
        row["dataset_name"]: row
        for row in rows
        if row.get("selector_name") == EXP90_OPERATIONAL_SELECTOR
    }
    research = {
        row["dataset_name"]: row
        for row in rows
        if row.get("selector_name") == EXP90_RESEARCH_SELECTOR
    }
    if set(operational) != set(research):
        raise SystemExit("Exp90 operational/research coverage mismatch")
    return operational, research


def load_exp87_rows():
    rows = read_dict_rows(EXP87_PATH)
    out = {}
    for row in rows:
        if row.get("config_name") != EXP87_CONFIG:
            continue
        out[(row["dataset_name"], row["threshold_method"])] = row
    return out


def parse_order(value) -> list[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [int(float(item)) for item in text.split()]


def bundle_order(bundle, count: int = MAX_TOP) -> list[int]:
    return top_score_order(bundle, count)


def exp84_order(row: Optional[Mapping], count: int = MAX_TOP) -> list[int]:
    if row is None:
        return []
    return parse_order(row.get("top_score_indices"))[:count]


def rank_map(order: Sequence[int]) -> dict[int, int]:
    return {int(idx): rank for rank, idx in enumerate(order, 1)}


def candidate_pool(*orders: Sequence[int]) -> Set[int]:
    pool: Set[int] = set()
    for order in orders:
        pool.update(int(idx) for idx in order)
    return pool


def score_candidates(
    candidates: Iterable[int],
    rank_maps: Mapping[str, Mapping[int, int]],
    weights: Mapping[str, float],
    max_rank: int = MAX_TOP,
) -> dict[int, dict[str, float]]:
    out = {}
    for idx in candidates:
        support = 0
        weighted = 0.0
        best_rank = max_rank + 1
        rank_sum = 0.0
        for name, ranks in rank_maps.items():
            rank = ranks.get(idx)
            if rank is None or rank > max_rank:
                continue
            support += 1
            best_rank = min(best_rank, rank)
            rank_sum += rank
            weighted += weights.get(name, 0.0) * ((max_rank + 1 - rank) / max_rank)
        consensus_bonus = 0.08 * max(0, support - 1)
        out[idx] = {
            "score": weighted + consensus_bonus,
            "support": float(support),
            "best_rank": float(best_rank),
            "rank_sum": float(rank_sum if support else 999.0),
        }
    return out


def sorted_candidates(scored: Mapping[int, Mapping[str, float]]) -> list[int]:
    return sorted(
        scored,
        key=lambda idx: (
            scored[idx]["score"],
            scored[idx]["support"],
            -scored[idx]["best_rank"],
            -scored[idx]["rank_sum"],
            -idx,
        ),
        reverse=True,
    )


def top_non_base(scored: Mapping[int, Mapping[str, float]], base_indices: Set[int]) -> Optional[int]:
    for idx in sorted_candidates(scored):
        if idx not in base_indices:
            return idx
    return None


def weak_base_indices(base_indices: Set[int], scored: Mapping[int, Mapping[str, float]]) -> Set[int]:
    weak = set()
    for idx in base_indices:
        info = scored.get(idx, {"score": 0.0, "support": 0.0, "best_rank": 99.0})
        if info["support"] <= 1 and info["best_rank"] > 5:
            weak.add(idx)
    return weak


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
            "score_family": "nonpos_candidate_reranker",
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


def choose_rows_for_dataset(args):
    dataset_name, exp90_operational_row, exp90_research_row, exp87_rows = args
    record, y_test, bundles = load_candidate_predictions(
        dataset_name,
        threshold_rates=CALIBRATION_PROFILES["relaxed_15pct"],
    )
    base_indices = parse_indices(exp90_operational_row.get("selected_indices"))
    research_indices = parse_indices(exp90_research_row.get("selected_indices"))
    train_normal_count = int(as_float(exp90_operational_row.get("train_normal_count"), len(record["train_series"])))
    tiny_train = train_normal_count <= 10
    sparse_alert = len(base_indices) <= 1

    rocket_order = bundle_order(bundles["rocket_exp40"])
    exp55_order = bundle_order(bundles["exp55_best"])
    exp56_order = bundle_order(bundles["exp56_best"])
    exp84_fg_row = exp87_rows.get((dataset_name, "family_guard_v1"))
    exp84_cap3_row = exp87_rows.get((dataset_name, "count_cap_3pct"))
    exp84_fg_order = exp84_order(exp84_fg_row)
    exp84_cap3_order = exp84_order(exp84_cap3_row)
    exp84_fg_train_safe = as_float(exp84_fg_row.get("train_exceed_rate") if exp84_fg_row else None, 1.0) <= 0.015
    exp84_cap3_train_safe = as_float(exp84_cap3_row.get("train_exceed_rate") if exp84_cap3_row else None, 1.0) <= 0.02
    if not exp84_fg_train_safe:
        exp84_fg_order = []
    if not exp84_cap3_train_safe:
        exp84_cap3_order = []

    rank_maps = {
        "rocket": rank_map(rocket_order),
        "exp55": rank_map(exp55_order),
        "exp56": rank_map(exp56_order),
        "exp84_fg": rank_map(exp84_fg_order),
        "exp84_cap3": rank_map(exp84_cap3_order),
    }
    weights = {
        "rocket": 0.36,
        "exp55": 0.22,
        "exp56": 0.22,
        "exp84_fg": 0.12,
        "exp84_cap3": 0.08,
    }
    pool = candidate_pool(rocket_order[:8], exp55_order[:8], exp56_order[:8], exp84_fg_order[:8], exp84_cap3_order[:8])
    scored = score_candidates(pool | base_indices, rank_maps, weights)
    top_candidate = top_non_base(scored, base_indices)
    weak_base = weak_base_indices(base_indices, scored)
    top_info = scored.get(top_candidate, {}) if top_candidate is not None else {}
    top_support = int(top_info.get("support", 0))
    top_score = float(top_info.get("score", 0.0))
    top_best_rank = int(top_info.get("best_rank", 99))
    base_best_score = max([scored.get(idx, {}).get("score", 0.0) for idx in base_indices] or [0.0])
    score_gain = top_score - base_best_score

    rows = [
        passthrough_row(
            exp90_operational_row,
            "baseline_exp90_noalert_top1",
            "control: Exp90 operating default",
            {
                "train_normal_count": train_normal_count,
                "tiny_train": int(tiny_train),
                "rerank_added_count": 0,
                "rerank_replaced_count": 0,
                "top_candidate": top_candidate if top_candidate is not None else "",
                "top_candidate_support": top_support,
                "top_candidate_score": top_score,
                "top_candidate_best_rank": top_best_rank,
                "top_candidate_score_gain": score_gain,
            },
        ),
        passthrough_row(
            exp90_research_row,
            "reference_exp90_candidate_union",
            "reference only: Exp90 candidate-union research selector",
            {
                "train_normal_count": train_normal_count,
                "tiny_train": int(tiny_train),
                "rerank_added_count": max(0, len(research_indices - base_indices)),
                "rerank_replaced_count": 0,
                "top_candidate": top_candidate if top_candidate is not None else "",
                "top_candidate_support": top_support,
                "top_candidate_score": top_score,
                "top_candidate_best_rank": top_best_rank,
                "top_candidate_score_gain": score_gain,
            },
        ),
    ]

    conservative = set(base_indices)
    conservative_added = 0
    if (not tiny_train) and sparse_alert and top_candidate is not None:
        if top_support >= 3 and top_best_rank <= 3 and score_gain >= 0.08:
            conservative.add(top_candidate)
            conservative_added = int(top_candidate not in base_indices)
    rows.append(
        row_with_metrics(
            dataset_name,
            record,
            y_test,
            bundles,
            exp90_operational_row,
            "nonpos_consensus_add_cap1",
            conservative,
            "add one non-position candidate only with support>=3, best_rank<=3, and score gain",
            {
                "train_normal_count": train_normal_count,
                "tiny_train": int(tiny_train),
                "rerank_added_count": conservative_added,
                "rerank_replaced_count": 0,
                "top_candidate": top_candidate if top_candidate is not None else "",
                "top_candidate_support": top_support,
                "top_candidate_score": top_score,
                "top_candidate_best_rank": top_best_rank,
                "top_candidate_score_gain": score_gain,
            },
        )
    )

    replace = set(base_indices)
    replaced_count = 0
    if (not tiny_train) and sparse_alert and weak_base and top_candidate is not None:
        if top_support >= 2 and top_best_rank <= 3 and score_gain >= 0.04:
            replace = (replace - weak_base) | {top_candidate}
            replaced_count = len(weak_base)
    rows.append(
        row_with_metrics(
            dataset_name,
            record,
            y_test,
            bundles,
            exp90_operational_row,
            "nonpos_weak_alert_replace",
            replace,
            "replace weak sparse alert using non-position rank consensus only",
            {
                "train_normal_count": train_normal_count,
                "tiny_train": int(tiny_train),
                "rerank_added_count": int(top_candidate not in base_indices) if replaced_count else 0,
                "rerank_replaced_count": replaced_count,
                "top_candidate": top_candidate if top_candidate is not None else "",
                "top_candidate_support": top_support,
                "top_candidate_score": top_score,
                "top_candidate_best_rank": top_best_rank,
                "top_candidate_score_gain": score_gain,
            },
        )
    )

    hybrid = set(replace)
    hybrid_added = 0
    if (not tiny_train) and sparse_alert and top_candidate is not None:
        if not replaced_count and top_support >= 4 and top_best_rank <= 2 and score_gain >= 0.12:
            hybrid.add(top_candidate)
            hybrid_added = int(top_candidate not in replace)
    rows.append(
        row_with_metrics(
            dataset_name,
            record,
            y_test,
            bundles,
            exp90_operational_row,
            "nonpos_replace_else_strict_add",
            hybrid,
            "replace weak alert first; otherwise add only very strong non-position consensus",
            {
                "train_normal_count": train_normal_count,
                "tiny_train": int(tiny_train),
                "rerank_added_count": hybrid_added + (int(top_candidate not in base_indices) if replaced_count else 0),
                "rerank_replaced_count": replaced_count,
                "top_candidate": top_candidate if top_candidate is not None else "",
                "top_candidate_support": top_support,
                "top_candidate_score": top_score,
                "top_candidate_best_rank": top_best_rank,
                "top_candidate_score_gain": score_gain,
            },
        )
    )

    review_candidates = set(base_indices)
    if (not tiny_train) and top_candidate is not None and top_support >= 2 and top_best_rank <= 5:
        review_candidates.add(top_candidate)
    rows.append(
        row_with_metrics(
            dataset_name,
            record,
            y_test,
            bundles,
            exp90_operational_row,
            "review_tier_nonpos_top_candidate",
            review_candidates,
            "diagnostic review-tier: add plausible top candidate for review, not recommended as hard alert by default",
            {
                "train_normal_count": train_normal_count,
                "tiny_train": int(tiny_train),
                "rerank_added_count": len(review_candidates - base_indices),
                "rerank_replaced_count": 0,
                "top_candidate": top_candidate if top_candidate is not None else "",
                "top_candidate_support": top_support,
                "top_candidate_score": top_score,
                "top_candidate_best_rank": top_best_rank,
                "top_candidate_score_gain": score_gain,
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
                "mean_rerank_added_count": float(np.mean(vals("rerank_added_count"))),
                "mean_rerank_replaced_count": float(np.mean(vals("rerank_replaced_count"))),
                "rerank_used_datasets": sum(
                    1
                    for row in subset
                    if as_float(row.get("rerank_added_count")) > 0
                    or as_float(row.get("rerank_replaced_count")) > 0
                ),
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
    exp90_operational, exp90_research = load_exp90_rows()
    exp87 = load_exp87_rows()
    datasets = sorted(exp90_operational)
    if dataset_limit:
        datasets = datasets[: int(dataset_limit)]
    tasks = [(name, exp90_operational[name], exp90_research[name], exp87) for name in datasets]
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
    STDOUT_LOG.write_text(
        f"{EXPERIMENT_ID} finished rows={len(rows)} datasets={len(datasets)}\n"
        f"{summary[0] if summary else ''}\n"
    )
    print(f"{EXPERIMENT_ID} finished rows={len(rows)} datasets={len(datasets)}")
    if summary:
        print(summary[0])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-limit", type=int, default=None)
    args = parser.parse_args()
    run_experiment(args.dataset_limit)
