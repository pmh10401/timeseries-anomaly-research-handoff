from __future__ import annotations

import csv
import os
from collections import defaultdict
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


DATA_DIR = Path("/Users/minho/Documents/Dataset")
EXPERIMENT_ID = "experiment_98_tiny_train_normal_pooling"
EXP93_PATH = DATA_DIR / "experiment_93_nonpos_candidate_reranker_results.csv"
EXP97_PATH = DATA_DIR / "experiment_97_zero_f1_feature_need_diagnosis_results.csv"
EXP93_SELECTOR = "nonpos_weak_alert_replace"
STDOUT_LOG = DATA_DIR / f"{EXPERIMENT_ID}_stdout.log"
WORKERS = int(os.environ.get("EXP98_WORKERS", "4"))


def read_rows(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


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


def load_exp93_rows():
    rows = read_rows(EXP93_PATH)
    out = {row["dataset_name"]: row for row in rows if row.get("selector_name") == EXP93_SELECTOR}
    if not out:
        raise SystemExit("Exp93 operating rows not found")
    return out


def load_target_names():
    rows = read_rows(EXP97_PATH)
    return {
        row["dataset_name"]
        for row in rows
        if row.get("feature_need_primary") == "A_tiny_train_normal_pooling"
    }


def load_family_pools(exp93_rows, target_names):
    target_families = {exp93_rows[name]["family"] for name in target_names if name in exp93_rows}
    family_names = defaultdict(list)
    for name, row in exp93_rows.items():
        if row["family"] in target_families:
            family_names[row["family"]].append(name)
    pools = defaultdict(list)
    for family, names in family_names.items():
        for name in sorted(names):
            try:
                _, _, bundles = load_candidate_predictions(
                    name,
                    threshold_rates=CALIBRATION_PROFILES["relaxed_15pct"],
                )
                pools[(family, "rocket_exp40")].extend(bundles["rocket_exp40"]["train_scores"])
                pools[(family, "exp55_best")].extend(bundles["exp55_best"]["train_scores"])
                pools[(family, "exp56_best")].extend(bundles["exp56_best"]["train_scores"])
            except Exception:
                continue
    return {key: np.asarray(values, dtype=float) for key, values in pools.items()}


def select_by_pool(scores, threshold, cap):
    scores = np.asarray(scores, dtype=float)
    candidates = np.flatnonzero(scores > threshold)
    if not len(candidates):
        return set()
    order = sorted(candidates, key=lambda idx: scores[idx], reverse=True)
    return set(int(idx) for idx in order[:cap])


def pooled_indices(bundle, pool_scores, q, cap, own_weight=0.0):
    train_scores = np.asarray(bundle["train_scores"], dtype=float)
    if pool_scores is None or len(pool_scores) < 8:
        ref = train_scores
    elif own_weight > 0 and len(train_scores):
        ref = np.concatenate([pool_scores, np.repeat(train_scores, max(1, int(len(pool_scores) * own_weight / len(train_scores))))])
    else:
        ref = pool_scores
    threshold = float(np.quantile(ref, q))
    return select_by_pool(bundle["test_scores"], threshold, cap), threshold


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
            "score_family": "tiny_train_normal_pooling",
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
        }
    )
    if extras:
        out.update(extras)
    return out


def passthrough(base_row, selector_name, reason, extras=None):
    out = dict(base_row)
    out["experiment_id"] = EXPERIMENT_ID
    out["selector_name"] = selector_name
    out["config_name"] = selector_name
    out["selector_reason"] = reason
    out["threshold_method"] = "selector"
    if extras:
        out.update(extras)
    return out


def choose_dataset(args):
    name, base_row, is_target, pool_payload = args
    if not is_target:
        return [
            passthrough(base_row, selector, "non-target passthrough from Exp93", {"pooling_target": 0})
            for selector in [
                "baseline_exp93_operating",
                "rocket_family_pool_q99_cap1",
                "rocket_family_pool_q98_cap1",
                "multi_source_pool_union_q99_cap1",
                "multi_source_pool_union_q98_cap2",
            ]
        ]

    family = base_row["family"]
    record, y_test, bundles = load_candidate_predictions(name, threshold_rates=CALIBRATION_PROFILES["relaxed_15pct"])
    base_indices = parse_indices(base_row.get("selected_indices"))
    pools = {
        key: np.asarray(values, dtype=float)
        for key, values in pool_payload.items()
        if key[0] == family
    }
    rows = [
        row_with_metrics(
            name,
            record,
            y_test,
            bundles,
            base_row,
            "baseline_exp93_operating",
            base_indices,
            "control: Exp93 operating default",
            {"pooling_target": 1, "pool_source": "none"},
        )
    ]
    configs = [
        ("rocket_family_pool_q99_cap1", [("rocket_exp40", 0.99, 1)], 1),
        ("rocket_family_pool_q98_cap1", [("rocket_exp40", 0.98, 1)], 1),
        (
            "multi_source_pool_union_q99_cap1",
            [("rocket_exp40", 0.99, 1), ("exp55_best", 0.99, 1), ("exp56_best", 0.99, 1)],
            1,
        ),
        (
            "multi_source_pool_union_q98_cap2",
            [("rocket_exp40", 0.98, 2), ("exp55_best", 0.98, 2), ("exp56_best", 0.98, 2)],
            2,
        ),
    ]
    for selector, sources, cap in configs:
        selected = set()
        thresholds = []
        for source_name, q, source_cap in sources:
            idxs, threshold = pooled_indices(
                bundles[source_name],
                pools.get((family, source_name)),
                q=q,
                cap=source_cap,
            )
            selected |= idxs
            thresholds.append(f"{source_name}:q{q}:{threshold:.6g}")
        if len(selected) > cap:
            # Final hard-alert cap is ranked by ROCKET score for a stable operating surface.
            scores = bundles["rocket_exp40"]["test_scores"]
            selected = set(sorted(selected, key=lambda idx: scores[idx], reverse=True)[:cap])
        rows.append(
            row_with_metrics(
                name,
                record,
                y_test,
                bundles,
                base_row,
                selector,
                selected,
                "replace tiny-train target alert with family-pooled normal threshold candidates",
                {
                    "pooling_target": 1,
                    "pool_source": "family_train_scores",
                    "pool_thresholds": ";".join(thresholds),
                    "pool_family_train_score_count": len(pools.get((family, "rocket_exp40"), [])),
                },
            )
        )
    return rows


def summarize(rows):
    out = []
    for selector in sorted({row["selector_name"] for row in rows}):
        subset = [row for row in rows if row["selector_name"] == selector]
        target = [row for row in subset if as_float(row.get("pooling_target")) > 0]
        vals = lambda key, rows_: [as_float(row.get(key)) for row in rows_]
        f1s = vals("f1", subset)
        target_f1s = vals("f1", target)
        out.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "selector_name": selector,
                "config_name": selector,
                "threshold_method": "selector",
                "num_datasets": len(subset),
                "num_families": len({row["family"] for row in subset}),
                "mean_auc_roc": float(np.mean(vals("auc_roc", subset))),
                "mean_auc_pr": float(np.mean(vals("auc_pr", subset))),
                "mean_f1": float(np.mean(f1s)),
                "median_f1": float(np.median(f1s)),
                "p25_f1": float(np.percentile(f1s, 25)),
                "zero_f1_count": sum(1 for value in f1s if value == 0.0),
                "ge_0_5_count": sum(1 for value in f1s if value >= 0.5),
                "mean_predicted_count": float(np.mean(vals("predicted_count", subset))),
                "mean_anomaly_count": float(np.mean(vals("anomaly_count", subset))),
                "mean_tp": float(np.mean(vals("tp", subset))),
                "mean_fp": float(np.mean(vals("fp", subset))),
                "mean_fn": float(np.mean(vals("fn", subset))),
                "mean_oracle_f1": float(np.mean(vals("oracle_f1", subset))),
                "target_datasets": len(target),
                "target_mean_f1": float(np.mean(target_f1s)) if target_f1s else 0.0,
                "target_zero_f1_count": sum(1 for value in target_f1s if value == 0.0) if target_f1s else 0,
                "target_mean_fp": float(np.mean(vals("fp", target))) if target else 0.0,
                "target_mean_tp": float(np.mean(vals("tp", target))) if target else 0.0,
            }
        )
    return sorted(out, key=lambda row: (row["mean_f1"], -row["mean_fp"]), reverse=True)


def run_experiment(dataset_limit=None):
    exp93 = load_exp93_rows()
    targets = load_target_names()
    if dataset_limit:
        names = sorted(exp93)[: int(dataset_limit)]
        targets = targets & set(names)
    else:
        names = sorted(exp93)
    pools = load_family_pools(exp93, targets)
    pool_payload = {key: value.tolist() for key, value in pools.items()}
    tasks = [(name, exp93[name], name in targets, pool_payload) for name in names]
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(choose_dataset, task): task[0] for task in tasks}
        for idx, future in enumerate(as_completed(futures), 1):
            rows.extend(future.result())
            if idx % 50 == 0 or idx == len(tasks):
                print(f"{EXPERIMENT_ID} progress {idx}/{len(tasks)}", flush=True)
    write_csv(results_path(EXPERIMENT_ID), rows)
    summary = summarize(rows)
    write_csv(summary_path(EXPERIMENT_ID), summary)
    STDOUT_LOG.write_text(f"{EXPERIMENT_ID} finished rows={len(rows)} datasets={len(names)} targets={len(targets)}\n{summary[0] if summary else ''}\n")
    print(f"{EXPERIMENT_ID} finished rows={len(rows)} datasets={len(names)} targets={len(targets)}")
    if summary:
        print(summary[0])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-limit", type=int, default=None)
    args = parser.parse_args()
    run_experiment(args.dataset_limit)
