from __future__ import annotations

import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.preprocessing import RobustScaler

from run_experiment_60_62_rocket_imaging_selector_variants import (
    evaluate_indices,
    results_path,
    summary_path,
)
from run_experiment_89_74d_with_exp84_candidate import as_float, format_indices, parse_indices
from run_rank_ensemble_calibration import load_dataset_data, z_normalize


DATA_DIR = Path("/Users/minho/Documents/Dataset")
EXPERIMENT_ID = "experiment_99_spectral_derivative_feature_score"
EXP93_PATH = DATA_DIR / "experiment_93_nonpos_candidate_reranker_results.csv"
EXP97_PATH = DATA_DIR / "experiment_97_zero_f1_feature_need_diagnosis_results.csv"
EXP93_SELECTOR = "nonpos_weak_alert_replace"
STDOUT_LOG = DATA_DIR / f"{EXPERIMENT_ID}_stdout.log"
WORKERS = int(os.environ.get("EXP99_WORKERS", "4"))

SPECTRAL_FAMILIES = {
    "Phoneme",
    "CricketX",
    "CricketY",
    "CricketZ",
    "GestureMidAirD1",
    "GestureMidAirD2",
    "GestureMidAirD3",
    "AllGestureWiimoteX",
    "AllGestureWiimoteY",
    "AllGestureWiimoteZ",
    "EOGHorizontalSignal",
    "EOGVerticalSignal",
    "InlineSkate",
    "EthanolLevel",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "UWaveGestureLibraryAll",
    "Haptics",
}


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


def load_research_targets():
    rows = read_rows(EXP97_PATH)
    return {
        row["dataset_name"]
        for row in rows
        if row.get("feature_need_primary") == "C_spectral_derivative_feature"
    }


def spectral_features(X):
    X = np.asarray(X, dtype=np.float64)
    d1 = np.diff(X, axis=1, prepend=X[:, :1])
    d2 = np.diff(d1, axis=1, prepend=d1[:, :1])
    fft = np.abs(np.fft.rfft(X, axis=1))
    if fft.shape[1] > 1:
        fft[:, 0] = 0.0
    total = fft.sum(axis=1, keepdims=True) + 1e-9
    norm_fft = fft / total
    n_bins = fft.shape[1]
    edges = np.linspace(0, n_bins, 9, dtype=int)
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        hi = max(hi, lo + 1)
        bands.append(norm_fft[:, lo:hi].sum(axis=1))
    bands = np.vstack(bands).T
    freqs = np.arange(n_bins, dtype=np.float64)
    centroid = (norm_fft * freqs).sum(axis=1) / max(1, n_bins - 1)
    entropy = -(norm_fft * np.log(norm_fft + 1e-12)).sum(axis=1) / np.log(max(2, n_bins))
    peak = norm_fft.max(axis=1)
    def stats(A):
        return np.vstack([
            A.mean(axis=1),
            A.std(axis=1),
            np.median(A, axis=1),
            np.percentile(A, 90, axis=1) - np.percentile(A, 10, axis=1),
            np.max(np.abs(A), axis=1),
        ]).T
    return np.hstack([stats(X), stats(d1), stats(d2), bands, centroid[:, None], entropy[:, None], peak[:, None]])


def robust_score_pair(train_features, test_features, top_k=None):
    scaler = RobustScaler(quantile_range=(10, 90))
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    center = np.median(train_scaled, axis=0)
    spread = np.median(np.abs(train_scaled - center), axis=0) + 1e-6
    train_dev = np.abs(train_scaled - center) / spread
    test_dev = np.abs(test_scaled - center) / spread
    if top_k is None:
        return train_dev.mean(axis=1), test_dev.mean(axis=1)
    top_k = max(1, min(int(top_k), train_dev.shape[1]))
    split_at = train_dev.shape[1] - top_k
    train_scores = np.mean(np.partition(train_dev, split_at, axis=1)[:, -top_k:], axis=1)
    test_scores = np.mean(np.partition(test_dev, split_at, axis=1)[:, -top_k:], axis=1)
    return train_scores, test_scores


def select(scores, train_scores, q, cap):
    threshold = float(np.quantile(train_scores, q))
    idxs = np.flatnonzero(np.asarray(scores) > threshold)
    order = sorted(idxs, key=lambda idx: scores[idx], reverse=True)
    return set(int(i) for i in order[:cap]), threshold


def row_with_metrics(name, family, y_test, base_row, selector, indices, scores, reason, extras=None):
    metrics = evaluate_indices(y_test, scores, indices)
    out = dict(base_row)
    out.update({
        "experiment_id": EXPERIMENT_ID,
        "dataset_name": name,
        "family": family,
        "config_name": selector,
        "selector_name": selector,
        "selector_reason": reason,
        "score_source_name": "spectral_derivative",
        "threshold_method": "selector",
        "score_family": "spectral_derivative_feature_score",
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
    })
    if extras:
        out.update(extras)
    return out


def passthrough(base_row, selector, reason, extras=None):
    out = dict(base_row)
    out["experiment_id"] = EXPERIMENT_ID
    out["selector_name"] = selector
    out["config_name"] = selector
    out["selector_reason"] = reason
    out["threshold_method"] = "selector"
    if extras:
        out.update(extras)
    return out


def run_dataset(args):
    name, base_row, research_target = args
    family = base_row["family"]
    family_target = family in SPECTRAL_FAMILIES
    selectors = [
        "baseline_exp93_operating",
        "research_spectral_q98_cap2",
        "train_family_spectral_q99_cap1",
        "train_family_spectral_q98_cap2",
        "train_family_spectral_topk4_q98_cap2",
    ]
    if not (research_target or family_target):
        return [passthrough(base_row, s, "non-target passthrough", {"spectral_target": 0}) for s in selectors]
    X_train, X_test, y_test = load_dataset_data(name)
    X_train = z_normalize(X_train).astype(np.float64)
    X_test = z_normalize(X_test).astype(np.float64)
    train_feat = spectral_features(X_train)
    test_feat = spectral_features(X_test)
    train_mean, test_mean = robust_score_pair(train_feat, test_feat, None)
    train_top4, test_top4 = robust_score_pair(train_feat, test_feat, 4)
    base_indices = parse_indices(base_row.get("selected_indices"))
    rows = [
        passthrough(
            base_row,
            "baseline_exp93_operating",
            "control: Exp93 operating default",
            {"spectral_target": int(family_target), "research_target": int(research_target)},
        )
    ]
    configs = [
        ("research_spectral_q98_cap2", research_target, train_mean, test_mean, 0.98, 2),
        ("train_family_spectral_q99_cap1", family_target, train_mean, test_mean, 0.99, 1),
        ("train_family_spectral_q98_cap2", family_target, train_mean, test_mean, 0.98, 2),
        ("train_family_spectral_topk4_q98_cap2", family_target, train_top4, test_top4, 0.98, 2),
    ]
    for selector, enabled, train_scores, test_scores, q, cap in configs:
        if enabled:
            idxs, threshold = select(test_scores, train_scores, q, cap)
        else:
            idxs, threshold = base_indices, float("nan")
        rows.append(row_with_metrics(
            name, family, y_test, base_row, selector, idxs, test_scores,
            "spectral/derivative robust score candidate",
            {
                "spectral_target": int(family_target),
                "research_target": int(research_target),
                "spectral_threshold": threshold,
            },
        ))
    return rows


def summarize(rows):
    out = []
    for selector in sorted({r["selector_name"] for r in rows}):
        subset = [r for r in rows if r["selector_name"] == selector]
        target = [r for r in subset if as_float(r.get("spectral_target")) > 0]
        research_target = [r for r in subset if as_float(r.get("research_target")) > 0]
        vals = lambda k, rs: [as_float(r.get(k)) for r in rs]
        f1s = vals("f1", subset)
        target_f1s = vals("f1", target)
        research_f1s = vals("f1", research_target)
        out.append({
            "experiment_id": EXPERIMENT_ID,
            "selector_name": selector,
            "config_name": selector,
            "threshold_method": "selector",
            "num_datasets": len(subset),
            "num_families": len({r["family"] for r in subset}),
            "mean_auc_roc": float(np.mean(vals("auc_roc", subset))),
            "mean_auc_pr": float(np.mean(vals("auc_pr", subset))),
            "mean_f1": float(np.mean(f1s)),
            "median_f1": float(np.median(f1s)),
            "p25_f1": float(np.percentile(f1s, 25)),
            "zero_f1_count": sum(1 for v in f1s if v == 0.0),
            "ge_0_5_count": sum(1 for v in f1s if v >= 0.5),
            "mean_predicted_count": float(np.mean(vals("predicted_count", subset))),
            "mean_tp": float(np.mean(vals("tp", subset))),
            "mean_fp": float(np.mean(vals("fp", subset))),
            "mean_fn": float(np.mean(vals("fn", subset))),
            "mean_oracle_f1": float(np.mean(vals("oracle_f1", subset))),
            "target_datasets": len(target),
            "target_mean_f1": float(np.mean(target_f1s)) if target_f1s else 0.0,
            "target_zero_f1_count": sum(1 for v in target_f1s if v == 0.0) if target_f1s else 0,
            "target_mean_fp": float(np.mean(vals("fp", target))) if target else 0.0,
            "target_mean_tp": float(np.mean(vals("tp", target))) if target else 0.0,
            "research_target_datasets": len(research_target),
            "research_target_mean_f1": float(np.mean(research_f1s)) if research_f1s else 0.0,
            "research_target_zero_f1_count": sum(1 for v in research_f1s if v == 0.0) if research_f1s else 0,
            "research_target_mean_fp": float(np.mean(vals("fp", research_target))) if research_target else 0.0,
            "research_target_mean_tp": float(np.mean(vals("tp", research_target))) if research_target else 0.0,
        })
    return sorted(out, key=lambda r: (r["mean_f1"], -r["mean_fp"]), reverse=True)


def run_experiment(dataset_limit=None):
    exp93 = {r["dataset_name"]: r for r in read_rows(EXP93_PATH) if r.get("selector_name") == EXP93_SELECTOR}
    research_targets = load_research_targets()
    names = sorted(exp93)
    if dataset_limit:
        names = names[: int(dataset_limit)]
    tasks = [(n, exp93[n], n in research_targets) for n in names]
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(run_dataset, t): t[0] for t in tasks}
        for idx, fut in enumerate(as_completed(futures), 1):
            rows.extend(fut.result())
            if idx % 50 == 0 or idx == len(tasks):
                print(f"{EXPERIMENT_ID} progress {idx}/{len(tasks)}", flush=True)
    write_csv(results_path(EXPERIMENT_ID), rows)
    summary = summarize(rows)
    write_csv(summary_path(EXPERIMENT_ID), summary)
    STDOUT_LOG.write_text(f"{EXPERIMENT_ID} finished rows={len(rows)} datasets={len(names)}\n{summary[0] if summary else ''}\n")
    print(f"{EXPERIMENT_ID} finished rows={len(rows)} datasets={len(names)}")
    if summary:
        print(summary[0])


def load_research_targets():
    rows = read_rows(EXP97_PATH)
    return {r["dataset_name"] for r in rows if r.get("feature_need_primary") == "C_spectral_derivative_feature"}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-limit", type=int, default=None)
    args = p.parse_args()
    run_experiment(args.dataset_limit)
