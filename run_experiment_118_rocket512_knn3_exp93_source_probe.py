from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from run_experiment_40_original_score_normalization_sweep import count_cap_threshold, score_pair_for_config
from run_experiment_60_62_rocket_imaging_selector_variants import (
    CALIBRATION_PROFILES,
    evaluate_indices,
    load_candidate_predictions,
    prediction_bundle,
    results_path,
    summary_path,
)
from run_experiment_89_74d_with_exp84_candidate import EXP87_CONFIG, as_float, format_indices, parse_indices
from run_experiment_93_nonpos_candidate_reranker import (
    EXP90_OPERATIONAL_SELECTOR,
    EXP90_PATH,
    MAX_TOP,
    candidate_pool,
    exp84_order,
    rank_map,
    read_dict_rows,
    score_candidates,
    sorted_candidates,
    weak_base_indices,
)
from run_rank_ensemble_calibration import load_dataset_data, z_normalize


DATA_DIR = Path("/Users/minho/Documents/Dataset")
EXPERIMENT_ID = "experiment_118_rocket512_knn3_exp93_source_probe"
EXP93_PATH = DATA_DIR / "experiment_93_nonpos_candidate_reranker_results.csv"
EXP93_SELECTOR = "nonpos_weak_alert_replace"
EXP87_PATH = DATA_DIR / "experiment_87_exp84_index_diagnostics_results.csv"
WORKERS = int(os.environ.get("RANK_EXPERIMENT_WORKERS", "6"))
TOP_COUNT = MAX_TOP
WEIGHTS = {"rocket": 0.36, "exp55": 0.22, "exp56": 0.22, "exp84_fg": 0.12, "exp84_cap3": 0.08}
ROCKET_512_CONFIG = {"name": "rocket_512_knn3_local_gap", "kind": "density_knn", "num_kernels": 512, "neighbors": 3, "mode": "local_gap"}


def write_csv(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def deterministic_order(bundle, count=TOP_COUNT):
    scores = np.asarray(bundle["test_scores"], dtype=np.float64)
    return [int(idx) for idx in np.argsort(-scores, kind="stable")[: min(count, len(scores))]]


def top_jaccard(left, right):
    left, right = set(left), set(right)
    return len(left & right) / len(left | right) if left | right else 1.0


def make_rocket512_bundle(dataset_name, y_test):
    x_train, x_test, _ = load_dataset_data(dataset_name)
    seq_len = x_train.shape[1]
    train_scores, test_scores = score_pair_for_config(
        z_normalize(x_train).astype(np.float32),
        z_normalize(x_test).astype(np.float32),
        seq_len,
        ROCKET_512_CONFIG,
        {},
    )
    rate = CALIBRATION_PROFILES["relaxed_15pct"]["rocket_exp40"]
    threshold, q_effective, cap_target = count_cap_threshold(train_scores, rate)
    return prediction_bundle("rocket_512_knn3_local_gap", y_test, train_scores, test_scores, threshold, q_effective, cap_target)


def load_maps():
    exp90 = {row["dataset_name"]: row for row in read_dict_rows(EXP90_PATH) if row.get("selector_name") == EXP90_OPERATIONAL_SELECTOR}
    exp93 = {row["dataset_name"]: row for row in read_dict_rows(EXP93_PATH) if row.get("selector_name") == EXP93_SELECTOR}
    exp87 = {}
    for row in read_dict_rows(EXP87_PATH):
        if row.get("config_name") == EXP87_CONFIG:
            exp87[(row["dataset_name"], row["threshold_method"])] = row
    if len(exp90) != 1117 or set(exp90) != set(exp93):
        raise SystemExit("Exp90/Exp93 baseline coverage mismatch")
    return exp90, exp93, exp87


def guarded_exp84_orders(name, exp87):
    fg_row = exp87.get((name, "family_guard_v1"))
    cap3_row = exp87.get((name, "count_cap_3pct"))
    fg = exp84_order(fg_row)
    cap3 = exp84_order(cap3_row)
    if as_float(fg_row.get("train_exceed_rate") if fg_row else None, 1.0) > 0.015:
        fg = []
    if as_float(cap3_row.get("train_exceed_rate") if cap3_row else None, 1.0) > 0.02:
        cap3 = []
    return fg, cap3


def choose_candidate(base, maps, pool, tiny_train):
    scored = score_candidates(pool | base, maps, WEIGHTS)
    top = next((idx for idx in sorted_candidates(scored) if idx not in base), None)
    weak = weak_base_indices(base, scored)
    info = scored.get(top, {}) if top is not None else {}
    base_score = max([scored.get(idx, {}).get("score", 0.0) for idx in base] or [0.0])
    gain = float(info.get("score", 0.0)) - base_score
    selected, replaced = set(base), 0
    if not tiny_train and len(base) <= 1 and weak and top is not None:
        if int(info.get("support", 0)) >= 2 and int(info.get("best_rank", 99)) <= 3 and gain >= 0.04:
            selected = (selected - weak) | {top}
            replaced = len(weak)
    return selected, {
        "rerank_added_count": int(replaced and top not in base),
        "rerank_replaced_count": replaced,
        "top_candidate": "" if top is None else top,
        "top_candidate_support": int(info.get("support", 0)),
        "top_candidate_best_rank": int(info.get("best_rank", 99)),
        "top_candidate_score_gain": gain,
    }


def make_row(name, record, y_test, reference, config, selected, score_bundle, diagnostics, extra):
    metrics = evaluate_indices(y_test, score_bundle["test_scores"], selected)
    return {
        **reference,
        "experiment_id": EXPERIMENT_ID,
        "dataset_name": name,
        "family": record["family"],
        "config_name": config,
        "selector_name": config,
        "selector_reason": extra.pop("selector_reason"),
        "threshold_method": "selector",
        "score_family": "rocket512_exp93_source_probe",
        "selected_indices": format_indices(selected),
        "predicted_count": metrics["predicted_count"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "f1": metrics["f1"],
        "auc_roc": metrics["auc_roc"],
        "auc_pr": metrics["auc_pr"],
        "oracle_f1": metrics["oracle_f1"],
        "train_normal_count": len(record["train_series"]),
        "tiny_train": int(len(record["train_series"]) <= 10),
        **diagnostics,
        **extra,
    }


def run_one(args):
    name, exp90_row, exp93_row, exp87 = args
    record, y_test, bundles = load_candidate_predictions(name, threshold_rates=CALIBRATION_PROFILES["relaxed_15pct"])
    rocket256 = bundles["rocket_exp40"]
    rocket512 = make_rocket512_bundle(name, y_test)
    order256 = deterministic_order(rocket256)
    order512 = deterministic_order(rocket512)
    exp55 = deterministic_order(bundles["exp55_best"])
    exp56 = deterministic_order(bundles["exp56_best"])
    exp84_fg, exp84_cap3 = guarded_exp84_orders(name, exp87)
    base = parse_indices(exp90_row.get("selected_indices"))
    tiny_train = len(record["train_series"]) <= 10
    quality256 = evaluate_indices(y_test, rocket256["test_scores"], set())
    quality512 = evaluate_indices(y_test, rocket512["test_scores"], set())
    diagnostics = {
        "rocket256_auc_pr": quality256["auc_pr"],
        "rocket512_auc_pr": quality512["auc_pr"],
        "rocket_auc_pr_delta_512_minus_256": quality512["auc_pr"] - quality256["auc_pr"],
        "rocket256_oracle_f1": quality256["oracle_f1"],
        "rocket512_oracle_f1": quality512["oracle_f1"],
        "rocket_oracle_f1_delta_512_minus_256": quality512["oracle_f1"] - quality256["oracle_f1"],
        "rocket_top12_jaccard": top_jaccard(order256, order512),
        "rocket_top1_same": int(order256[0] == order512[0]),
        "rocket256_top12": format_indices(order256),
        "rocket512_top12": format_indices(order512),
    }

    def source_maps(rocket_ranks):
        return {"rocket": rocket_ranks, "exp55": rank_map(exp55), "exp56": rank_map(exp56), "exp84_fg": rank_map(exp84_fg), "exp84_cap3": rank_map(exp84_cap3)}

    other_pool = candidate_pool(exp55[:8], exp56[:8], exp84_fg[:8], exp84_cap3[:8])
    deterministic, deterministic_extra = choose_candidate(base, source_maps(rank_map(order256)), other_pool | set(order256[:8]), tiny_train)
    replace512, replace512_extra = choose_candidate(base, source_maps(rank_map(order512)), other_pool | set(order512[:8]), tiny_train)
    ranks256, ranks512 = rank_map(order256), rank_map(order512)
    family_rank = {idx: min(ranks256.get(idx, TOP_COUNT + 1), ranks512.get(idx, TOP_COUNT + 1)) for idx in set(order256) | set(order512)}
    tiebreak, tiebreak_extra = choose_candidate(base, source_maps(family_rank), other_pool | set(order256[:8]) | set(order512[:8]), tiny_train)

    historical = dict(exp93_row)
    historical.update({
        "experiment_id": EXPERIMENT_ID,
        "config_name": "baseline_exp93_historical",
        "selector_name": "baseline_exp93_historical",
        "selector_reason": "control: recorded Exp93 operating output",
        "threshold_method": "selector",
        "rocket_source_variant": "historical_256_unordered_top_set",
        **diagnostics,
    })
    return [
        historical,
        make_row(name, record, y_test, exp93_row, "exp93_deterministic_rocket256_control", deterministic, rocket256, diagnostics, {**deterministic_extra, "rocket_source_variant": "rocket256_deterministic_rank", "selector_reason": "control: Exp93 rules with deterministic 256-kernel ROCKET ranking"}),
        make_row(name, record, y_test, exp93_row, "exp93_rocket512_rank_replace", replace512, rocket512, diagnostics, {**replace512_extra, "rocket_source_variant": "rocket512_deterministic_rank", "selector_reason": "replace only the Exp93 ROCKET candidate ranking with 512-kernel KNN-3 local-gap"}),
        make_row(name, record, y_test, exp93_row, "exp93_rocket256_512_family_tiebreak", tiebreak, rocket256, diagnostics, {**tiebreak_extra, "rocket_source_variant": "rocket256_512_single_family_best_rank", "selector_reason": "one ROCKET-family vote using each candidate's better 256/512 rank; no double-counted support"}),
    ]


def summarize(rows):
    out = []
    for config in sorted({row["config_name"] for row in rows}):
        subset = [row for row in rows if row["config_name"] == config]
        values = lambda key: [as_float(row.get(key)) for row in subset]
        f1s = values("f1")
        out.append({
            "experiment_id": EXPERIMENT_ID,
            "config_name": config,
            "selector_name": config,
            "threshold_method": "selector",
            "num_datasets": len(subset),
            "mean_f1": float(np.mean(f1s)),
            "median_f1": float(np.median(f1s)),
            "zero_f1_count": sum(value == 0.0 for value in f1s),
            "mean_fp": float(np.mean(values("fp"))),
            "mean_tp": float(np.mean(values("tp"))),
            "mean_fn": float(np.mean(values("fn"))),
            "mean_auc_pr": float(np.mean(values("auc_pr"))),
            "mean_oracle_f1": float(np.mean(values("oracle_f1"))),
            "mean_rocket256_auc_pr": float(np.mean(values("rocket256_auc_pr"))),
            "mean_rocket512_auc_pr": float(np.mean(values("rocket512_auc_pr"))),
            "mean_rocket_auc_pr_delta_512_minus_256": float(np.mean(values("rocket_auc_pr_delta_512_minus_256"))),
            "mean_rocket256_oracle_f1": float(np.mean(values("rocket256_oracle_f1"))),
            "mean_rocket512_oracle_f1": float(np.mean(values("rocket512_oracle_f1"))),
            "mean_rocket_oracle_f1_delta_512_minus_256": float(np.mean(values("rocket_oracle_f1_delta_512_minus_256"))),
            "mean_rocket_top12_jaccard": float(np.mean(values("rocket_top12_jaccard"))),
            "top1_same_count": int(sum(values("rocket_top1_same"))),
            "rerank_used_datasets": int(sum(as_float(row.get("rerank_replaced_count")) > 0 for row in subset)),
        })
    return sorted(out, key=lambda row: (row["mean_f1"], -row["mean_fp"]), reverse=True)


def main(limit=None):
    exp90, exp93, exp87 = load_maps()
    names = sorted(exp90)[:limit] if limit else sorted(exp90)
    rows, errors = [], []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(run_one, (name, exp90[name], exp93[name], exp87)): name for name in names}
        for done, future in enumerate(as_completed(futures), 1):
            name = futures[future]
            try:
                rows.extend(future.result())
            except Exception as exc:
                errors.append((name, repr(exc)))
                print(f"ERROR dataset={name} error={exc!r}", flush=True)
            if done % 25 == 0 or done == len(names):
                print(f"Progress: [{done:4d}/{len(names):4d}] rows={len(rows)} last={name} errors={len(errors)}", flush=True)
    if errors or len(rows) != len(names) * 4:
        raise SystemExit(f"coverage failure {len(rows)}/{len(names) * 4} {errors[:5]}")
    write_csv(results_path(EXPERIMENT_ID), rows)
    write_csv(summary_path(EXPERIMENT_ID), summarize(rows))
    print(f"{EXPERIMENT_ID} finished rows={len(rows)} datasets={len(names)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-limit", type=int)
    main(parser.parse_args().dataset_limit)
