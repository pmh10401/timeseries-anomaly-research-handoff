from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from run_experiment_60_62_rocket_imaging_selector_variants import (
    CALIBRATION_PROFILES,
    evaluate_indices,
    load_candidate_predictions,
    results_path,
    summary_path,
)
from run_experiment_89_74d_with_exp84_candidate import as_float, format_indices


DATA_DIR = Path("/Users/minho/Documents/Dataset")
EXPERIMENT_ID = "experiment_113_train_normal_conformal_fusion"
EXP93_PATH = DATA_DIR / "experiment_93_nonpos_candidate_reranker_results.csv"
EXP93_SELECTOR = "nonpos_weak_alert_replace"
STDOUT_LOG = DATA_DIR / f"{EXPERIMENT_ID}_stdout.log"
WORKERS = int(os.environ.get("EXP113_WORKERS", "4"))
SOURCE_NAMES = ("rocket_exp40", "exp55_best", "exp56_best")
ALPHAS = (0.005, 0.01)


def read_rows(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_rows(path: Path, rows, fieldnames):
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def leave_self_out_tail_pvalues(train_scores, test_scores):
    """Convert high-is-anomalous scores to empirical train-normal tail p-values."""
    train_scores = np.nan_to_num(np.asarray(train_scores, dtype=np.float64), nan=0.0, posinf=1e12, neginf=-1e12)
    test_scores = np.nan_to_num(np.asarray(test_scores, dtype=np.float64), nan=0.0, posinf=1e12, neginf=-1e12)
    reference = np.sort(train_scores)
    n = max(1, len(reference))

    # Each normal calibration point excludes itself. Test points use all normals.
    train_ge = n - np.searchsorted(reference, train_scores, side="left") - 1
    test_ge = n - np.searchsorted(reference, test_scores, side="left")
    train_p = (1.0 + np.maximum(train_ge, 0)) / (n + 1.0)
    test_p = (1.0 + np.maximum(test_ge, 0)) / (n + 1.0)
    return np.clip(train_p, 1.0 / (n + 1.0), 1.0), np.clip(test_p, 1.0 / (n + 1.0), 1.0)


def cauchy_combined_pvalue(p_matrix):
    p_matrix = np.clip(np.asarray(p_matrix, dtype=np.float64), 1e-10, 1.0 - 1e-10)
    statistic = np.mean(np.tan((0.5 - p_matrix) * np.pi), axis=0)
    return np.clip(0.5 - np.arctan(statistic) / np.pi, 1e-10, 1.0)


def fusion_specs(source_pvalues):
    test_p = np.vstack([source_pvalues[name][1] for name in SOURCE_NAMES])
    min_p = np.min(test_p, axis=0)
    bonferroni_p = np.minimum(1.0, len(SOURCE_NAMES) * min_p)
    fisher_statistic = -2.0 * np.sum(np.log(np.clip(test_p, 1e-10, 1.0)), axis=0)
    fisher_p = chi2.sf(fisher_statistic, df=2 * len(SOURCE_NAMES))
    return {
        "conformal_bonferroni_minp": {
            "test_scores": -np.log(np.clip(bonferroni_p, 1e-10, 1.0)),
            "test_p": bonferroni_p,
            "selection": "combined_pvalue",
        },
        "conformal_cauchy": {
            "test_scores": -np.log(cauchy_combined_pvalue(test_p)),
            "test_p": cauchy_combined_pvalue(test_p),
            "selection": "combined_pvalue",
        },
        "conformal_fisher_independence": {
            "test_scores": -np.log(np.clip(fisher_p, 1e-10, 1.0)),
            "test_p": fisher_p,
            "selection": "combined_pvalue",
        },
        "conformal_2of3": {
            "test_scores": np.sum(test_p <= 0.01, axis=0).astype(np.float64),
            "selection": "source_agreement",
            "test_p": test_p,
        },
    }


def source_pvalues(bundles):
    return {
        name: leave_self_out_tail_pvalues(bundles[name]["train_scores"], bundles[name]["test_scores"])
        for name in SOURCE_NAMES
    }


def threshold_indices(spec, alpha):
    alpha = float(alpha)
    test_scores = np.asarray(spec["test_scores"], dtype=np.float64)
    selection = spec["selection"]
    if selection == "source_agreement":
        source_test_p = spec["test_p"]
        indices = np.flatnonzero(np.sum(source_test_p <= alpha, axis=0) >= 2)
        threshold = 2.0
    else:
        indices = np.flatnonzero(spec["test_p"] <= alpha)
        threshold = -float(np.log(alpha))
    return set(indices.astype(int).tolist()), threshold


def baseline_row(base_row):
    out = dict(base_row)
    out.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "config_name": "baseline_exp93_nonpos_weak_alert_replace",
            "selector_name": "baseline_exp93_nonpos_weak_alert_replace",
            "score_source_name": "exp93",
            "score_family": "baseline",
            "threshold_method": "selector",
            "threshold_family": "exp93_existing_policy",
            "calibration_kind": "existing_selector",
        }
    )
    return out


def result_row(dataset_name, record, y_test, name, spec, alpha, pvalues):
    indices, threshold = threshold_indices(spec, alpha)
    metrics = evaluate_indices(y_test, spec["test_scores"], indices)
    source_train_exceed = {
        source: float(np.mean(values[0] <= alpha))
        for source, values in pvalues.items()
    }
    return {
        "experiment_id": EXPERIMENT_ID,
        "dataset_name": dataset_name,
        "family": record["family"],
        "config_name": name,
        "selector_name": name,
        "score_source_name": "+".join(SOURCE_NAMES),
        "score_family": "train_normal_conformal_fusion",
        "threshold_method": f"train_normal_alpha_{alpha:g}",
        "threshold_family": "leave_self_out_train_normal_tail",
        "calibration_kind": spec["selection"],
        "source_names": ";".join(SOURCE_NAMES),
        "alpha": alpha,
        "sequence_length": len(record["test_series"][0]) if len(record["test_series"]) else "",
        "train_normal_count": min(len(values[0]) for values in pvalues.values()),
        "test_size": len(y_test),
        "anomaly_count": int(np.sum(y_test)),
        "threshold": threshold,
        "train_exceed_count": "",
        "train_exceed_rate": "",
        "normal_fpr_target": alpha,
        "train_fusion_rows_available": 0,
        "rocket_train_p_exceed_rate": source_train_exceed["rocket_exp40"],
        "spectrogram_train_p_exceed_rate": source_train_exceed["exp55_best"],
        "glcm_rp_train_p_exceed_rate": source_train_exceed["exp56_best"],
        "selected_indices": format_indices(indices),
        **metrics,
    }


def run_dataset(args):
    dataset_name, base_row = args
    record, y_test, bundles = load_candidate_predictions(
        dataset_name,
        threshold_rates=CALIBRATION_PROFILES["relaxed_15pct"],
    )
    pvalues = source_pvalues(bundles)
    specs = fusion_specs(pvalues)
    rows = [baseline_row(base_row)]
    for name, spec in specs.items():
        for alpha in ALPHAS:
            rows.append(result_row(dataset_name, record, y_test, name, spec, alpha, pvalues))
    return rows


def summarize(rows):
    out = []
    keys = sorted({(row["config_name"], row["threshold_method"]) for row in rows})
    for config_name, threshold_method in keys:
        subset = [row for row in rows if row["config_name"] == config_name and row["threshold_method"] == threshold_method]
        f1s = [as_float(row.get("f1")) for row in subset]
        families = {}
        for row in subset:
            families.setdefault(row["family"], []).append(as_float(row.get("f1")))
        out.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "config_name": config_name,
                "threshold_method": threshold_method,
                "calibration_kind": subset[0].get("calibration_kind", ""),
                "num_datasets": len(subset),
                "mean_f1": float(np.mean(f1s)),
                "median_f1": float(np.median(f1s)),
                "zero_f1_count": sum(value == 0.0 for value in f1s),
                "mean_fp": float(np.mean([as_float(row.get("fp")) for row in subset])),
                "mean_tp": float(np.mean([as_float(row.get("tp")) for row in subset])),
                "mean_fn": float(np.mean([as_float(row.get("fn")) for row in subset])),
                "mean_auc_pr": float(np.mean([as_float(row.get("auc_pr")) for row in subset])),
                "mean_oracle_f1": float(np.mean([as_float(row.get("oracle_f1")) for row in subset])),
                "normal_fpr_target": subset[0].get("normal_fpr_target", ""),
                "family_macro_f1": float(np.mean([np.mean(values) for values in families.values()])),
            }
        )
    return sorted(out, key=lambda row: (row["mean_f1"], -row["mean_fp"]), reverse=True)


def run_experiment(dataset_limit=None):
    base_rows = {
        row["dataset_name"]: row
        for row in read_rows(EXP93_PATH)
        if row.get("selector_name") == EXP93_SELECTOR
    }
    if len(base_rows) != 1117:
        raise SystemExit(f"Expected 1117 Exp93 baseline rows, got {len(base_rows)}")
    datasets = sorted(base_rows)
    if dataset_limit:
        datasets = datasets[: int(dataset_limit)]
    detail_path = results_path(EXPERIMENT_ID)
    summary_file = summary_path(EXPERIMENT_ID)
    for path in (detail_path, summary_file):
        if path.exists():
            path.unlink()

    rows = []
    fieldnames = None
    errors = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(run_dataset, (name, base_rows[name])): name for name in datasets}
        for completed, future in enumerate(as_completed(futures), 1):
            dataset_name = futures[future]
            try:
                dataset_rows = future.result()
            except Exception as exc:  # Keep the remaining queue work moving, then fail coverage clearly.
                errors.append((dataset_name, repr(exc)))
                print(f"ERROR dataset={dataset_name} error={exc!r}", flush=True)
                continue
            rows.extend(dataset_rows)
            if fieldnames is None:
                fieldnames = []
                for row in dataset_rows:
                    for key in row:
                        if key not in fieldnames:
                            fieldnames.append(key)
            append_rows(detail_path, dataset_rows, fieldnames)
            if completed % 25 == 0 or completed == len(datasets):
                write_csv(summary_file, summarize(rows))
                print(f"Progress: [{completed:4d}/{len(datasets):4d}] rows={len(rows)} last={dataset_name} errors={len(errors)}", flush=True)

    expected_rows = len(datasets) * (1 + len(ALPHAS) * 4)
    if errors or len(rows) != expected_rows:
        raise SystemExit(f"Coverage failure: rows={len(rows)}/{expected_rows} errors={errors[:10]}")
    write_csv(summary_file, summarize(rows))
    print(f"{EXPERIMENT_ID} finished rows={len(rows)} datasets={len(datasets)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 113 train-normal conformal score fusion")
    parser.add_argument("--dataset-limit", type=int, default=None)
    args = parser.parse_args()
    run_experiment(args.dataset_limit)
