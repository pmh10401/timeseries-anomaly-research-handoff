"""Pre-registered Exp149/150 virtual wafer-run conformal policy.

Prediction is built only from TRAIN-normal instances and frozen source
configuration. TEST labels are loaded only by the post-hoc evaluation step.
The existing Exp137/Exp145/Exp148 files are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

import run_experiment_132_block_b_review_integration as block_base
import run_experiment_133_block_b_confidence_tiers as exp133
import run_experiment_135_block_c_review_confirmation as exp135
import run_experiment_137_operational_triage as exp137
import run_experiment_40_original_score_normalization_sweep as exp40
import run_experiment_60_62_rocket_imaging_selector_variants as selector
import run_experiment_89_74d_with_exp84_candidate as exp89
import run_model_hard_research_experiments as model_hard
from run_experiment_40_original_score_normalization_sweep import count_cap_threshold
from run_original_improvement_experiment import DB_PATH, load_original_record, target_len_for_record
from run_rank_ensemble_calibration import align_series_lengths, load_dataset_data, z_normalize


ROOT = Path(__file__).resolve().parent
DATA_DIR = Path("/Users/minho/Documents/Dataset")
OUTPUT_DIR = ROOT / "outputs/exp137_policy_train_only_validation/virtual_run_conformal"
BASELINE_PATH = DATA_DIR / "experiment_137_operational_triage_results.csv"
EXP87_PATH = DATA_DIR / "experiment_87_exp84_index_diagnostics_results.csv"
B2_MANIFEST_PATH = ROOT / "outputs/exp137_policy_train_only_validation/b2_full/16_b2_full_coverage_source_manifest.csv"
EXTERNAL_LIVE_PATH = DATA_DIR / "rank_dashboard_external_live_run.json"
SEED = 20260717
ALPHAS = (0.005, 0.01, 0.02, 0.05)
WORKERS = int(os.environ.get("RANK_EXPERIMENT_WORKERS", "7"))

SOURCE_CONFIGS = {
    "rocket_exp40": {
        "kind": "rocket",
        "config": next(c for c in exp40.CONFIGS if c["name"] == "rocket_256_knn3_local_gap"),
    },
    "exp55_best": {
        "kind": "imaging",
        "config": selector.IMAGING_CONFIGS["exp55_best"],
    },
    "exp56_best": {
        "kind": "imaging",
        "config": selector.IMAGING_CONFIGS["exp56_best"],
    },
    "exp84": {
        "kind": "aeon",
        "config": next(c for c in model_hard.EXPERIMENT_SPECS["experiment_87_exp84_index_diagnostics"]["configs"]),
    },
}


def read_rows(path: Path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update_external_live_run(status, output_dir, done, expected, rows, started_at, error=None, result_summary=None):
    """Publish progress for the read-only dashboard without queue mutation."""
    payload = {
        "status": status,
        "id": "experiment_149_150_virtual_run_conformal_policy",
        "label": "Exp149/150 · virtual wafer-run TRAIN-only conformal policy",
        "pid": os.getpid(),
        "child_pid": os.getpid(),
        "expected_datasets": int(expected),
        "datasets_done": int(done),
        "progress_percent": float(done / expected * 100) if expected else 0.0,
        "rows": int(rows),
        "started_at": started_at,
        "detail_csv": str(Path(output_dir) / "03_exp149_current_source_results.csv"),
        "stdout_log": str(Path(output_dir) / "full_run.log"),
        "scope": "retrospective counterfactual virtual wafer-run benchmark",
        "autonomous_metric": "hard_alert_only",
        "review_metric": "human_assisted_diagnostic_only",
        "error": error,
    }
    if result_summary is not None:
        payload["result_summary"] = result_summary
        payload["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    temporary = EXTERNAL_LIVE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    temporary.replace(EXTERNAL_LIVE_PATH)


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_indices(value):
    return exp89.parse_indices(value)


def format_indices(indices):
    return " ".join(str(int(index)) for index in sorted(set(indices)))


def conformal_p_value(train_oof_scores, score):
    scores = np.asarray(train_oof_scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0 or not np.isfinite(score):
        return 1.0
    return float((1 + np.sum(scores >= float(score))) / (len(scores) + 1))


def fold_plan(n_train: int):
    n_train = int(n_train)
    if n_train < 2:
        return []
    if n_train < 20:
        return [(np.delete(np.arange(n_train), i), np.array([i])) for i in range(n_train)]
    splitter = KFold(n_splits=5, shuffle=True, random_state=SEED)
    return [(train.astype(int), test.astype(int)) for train, test in splitter.split(np.arange(n_train))]


def agreement_indices(p_values, alpha):
    names = sorted(p_values)
    if not names:
        return set(), np.zeros(0, dtype=int)
    matrix = np.vstack([np.asarray(p_values[name]) <= float(alpha) for name in names])
    counts = matrix.sum(axis=0)
    return set(np.flatnonzero(counts >= 2).astype(int).tolist()), counts


def route_lanes(candidates, hard_cross_check, priority_candidates, test_size):
    hard = set(candidates) & set(hard_cross_check)
    standard = set(candidates) - hard
    priority = set(priority_candidates) & standard
    selected = hard | standard | priority
    invalid = sorted(index for index in selected if index < 0 or index >= int(test_size))
    if invalid:
        raise ValueError(f"out-of-bounds candidate indices: {invalid[:10]}")
    return {
        "hard": hard,
        "standard_review": standard - priority,
        "priority_review": priority,
        "no_alert": set(range(int(test_size))) - selected,
    }


def tier_metrics(y_test, indices):
    return exp137.tier_metrics(np.asarray(y_test, dtype=int), set(indices))


def source_score_pair(source_name, x_fit, x_query, x_test, record):
    """Fit one frozen source on x_fit and score query/test without labels."""
    target_len = min(max(8, int(target_len_for_record(record, "actual_median"))), 2048)
    if source_name == "rocket_exp40":
        fit_z = z_normalize(x_fit).astype(np.float32)
        query_z = z_normalize(x_query).astype(np.float32)
        test_z = z_normalize(x_test).astype(np.float32)
        config = SOURCE_CONFIGS[source_name]["config"]
        fit_features, query_features = exp40.rocket_feature_pair(fit_z, query_z, fit_z.shape[1], config["num_kernels"])
        _, test_features = exp40.rocket_feature_pair(fit_z, test_z, fit_z.shape[1], config["num_kernels"])
        fit_for_query, query_features = model_hard.scale_feature_pair(fit_features, query_features)
        fit_for_test, test_features = model_hard.scale_feature_pair(fit_features, test_features)
        query_scores = exp40.density_knn_score_pair(fit_for_query, query_features, 3, "local_gap")[1]
        test_scores = exp40.density_knn_score_pair(fit_for_test, test_features, 3, "local_gap")[1]
        fit_scores = exp40.density_knn_score_pair(fit_for_query, fit_for_query, 3, "local_gap")[0]
        return fit_scores, query_scores, test_scores
    if source_name in {"exp55_best", "exp56_best"}:
        config = SOURCE_CONFIGS[source_name]["config"]
        # Match the frozen Exp55/56 preprocessing exactly, including their
        # train-normal scaling mode, image transform, PCA and KNN score.
        fit_raw = align_series_lengths([row for row in x_fit], target_len)
        query_raw = align_series_lengths([row for row in x_query], target_len)
        test_raw = align_series_lengths([row for row in x_test], target_len)
        fit_z = z_normalize(fit_raw).astype(np.float32)
        query_z = z_normalize(query_raw).astype(np.float32)
        test_z = z_normalize(test_raw).astype(np.float32)
        fit_pre, query_pre = model_hard.prepare_series_pair_for_scale(
            config.get("series_scale", "per_series_z"), fit_raw, query_raw, fit_z, query_z
        )
        fit_pre_for_test, test_pre = model_hard.prepare_series_pair_for_scale(
            config.get("series_scale", "per_series_z"), fit_raw, test_raw, fit_z, test_z
        )
        fit_scores, query_scores = model_hard.score_pair_for_config(
            fit_pre, query_pre, target_len, config, dict(record), {}
        )
        _, test_scores = model_hard.score_pair_for_config(
            fit_pre_for_test, test_pre, target_len, config, dict(record), {}
        )
        return fit_scores, query_scores, test_scores
    if source_name == "exp84":
        fit_z = z_normalize(x_fit).astype(np.float32)
        query_z = z_normalize(x_query).astype(np.float32)
        test_z = z_normalize(x_test).astype(np.float32)
        config = SOURCE_CONFIGS[source_name]["config"]
        fit_scores, query_scores = model_hard.score_pair_for_config(fit_z, query_z, fit_z.shape[1], config, dict(record), {})
        _, test_scores = model_hard.score_pair_for_config(fit_z, test_z, fit_z.shape[1], config, dict(record), {})
        return np.asarray(fit_scores), np.asarray(query_scores), np.asarray(test_scores)
    raise ValueError(f"unknown source {source_name}")


def crossfit_source(source_name, x_train, x_test, record):
    n_train = len(x_train)
    if n_train < 5:
        return {
            "train_oof_scores": np.array([], dtype=float),
            "test_scores": np.array([], dtype=float),
            "method": "unsupported_n_train_lt_5",
            "minimum_attainable_p": 1.0,
        }
    oof = np.full(n_train, np.nan, dtype=float)
    for fit_idx, query_idx in fold_plan(n_train):
        _, query_scores, _ = source_score_pair(source_name, x_train[fit_idx], x_train[query_idx], x_test, record)
        oof[query_idx] = np.asarray(query_scores, dtype=float)
    # Reuse x_test as the ignored query in the full-fit pass because PCA and
    # feature transformers require at least one query row.
    _, _, test_scores = source_score_pair(source_name, x_train, x_test, x_test, record)
    valid = oof[np.isfinite(oof)]
    return {
        "train_oof_scores": valid,
        "test_scores": np.asarray(test_scores, dtype=float),
        "method": "deterministic_5fold" if n_train >= 20 else "leave_one_out",
        "minimum_attainable_p": 1.0 / (len(valid) + 1),
    }


def load_source_coverage():
    current = {(row["dataset_name"], row["threshold_method"]): row for row in read_rows(EXP87_PATH)}
    b2 = {(row["dataset_name"], row["threshold_method"]): row for row in read_rows(B2_MANIFEST_PATH)}
    return current, b2


def load_routing_maps():
    row133, row135 = exp133.load_maps()
    return {
        name: {
            "hard": parse_indices(row133[name].get("high_confidence_indices")),
            "priority": parse_indices(row135[name].get("review_candidate_indices")),
        }
        for name in row133
    }


def make_prediction_record(name, x_train, x_test, record, current_sources, b2_sources, routing):
    source_scores = {}
    for source_name in SOURCE_CONFIGS:
        source_scores[source_name] = crossfit_source(source_name, x_train, x_test, record)
    rows = []
    for variant, available_exp84 in (("exp149_current_source", (name, "family_guard_v1") in current_sources), ("exp150_b2_source", True)):
        available = ["rocket_exp40", "exp55_best", "exp56_best"] + (["exp84"] if available_exp84 else [])
        for alpha in ALPHAS:
            p_values = {
                source: np.asarray(
                    [conformal_p_value(source_scores[source]["train_oof_scores"], score) for score in source_scores[source]["test_scores"]]
                )
                for source in available
                if source_scores[source]["test_scores"].size
            }
            if len(x_train) < 5:
                # Pre-registered abstention: preserve the dataset in the
                # report, but do not create autonomous Hard candidates.
                candidates = set()
                agreement_count = np.zeros(len(x_test), dtype=int)
            else:
                candidates, agreement_count = agreement_indices(p_values, alpha)
            lanes = route_lanes(candidates, routing["hard"], routing["priority"], len(x_test))
            metrics = {}
            for lane, indices in lanes.items():
                if lane == "no_alert":
                    continue
                metrics.update({f"{lane}_{key}": value for key, value in tier_metrics(record["y_test"], indices).items()})
            combined = set(lanes["hard"]) | set(lanes["standard_review"]) | set(lanes["priority_review"])
            metrics.update({f"combined_{key}": value for key, value in tier_metrics(record["y_test"], combined).items()})
            row = {
                "experiment_id": "experiment_149_virtual_run_conformal_policy" if variant.startswith("exp149") else "experiment_150_virtual_run_conformal_policy",
                "dataset_name": name,
                "family": record.get("family", ""),
                "variant": variant,
                "alpha": alpha,
                "n_train": len(x_train),
                "n_test": len(x_test),
                "low_train_review_only": int(len(x_train) < 5),
                "crossfit_method": source_scores["rocket_exp40"]["method"],
                "available_sources": " ".join(available),
                "source_count": len(available),
                "candidate_indices": format_indices(candidates),
                "candidate_count": len(candidates),
                "agreement_count_distribution": json.dumps({str(k): int(np.sum(agreement_count == k)) for k in range(len(available) + 1)}, sort_keys=True),
                "hard_cross_check_indices": format_indices(routing["hard"]),
                "priority_rule_indices": format_indices(routing["priority"]),
                "resolution_limited_at_alpha": int(any(alpha < source_scores[s]["minimum_attainable_p"] for s in available)),
                "minimum_attainable_p": min(source_scores[s]["minimum_attainable_p"] for s in available),
                "routing_uses_test_labels": 0,
                "routing_uses_test_positions": 0,
                "routing_uses_family_performance": 0,
                "retrospective_counterfactual": 1,
                "prospective_validated": 0,
                "mean_combined_f1_scope": "human_assisted_diagnostic_only",
                **{f"{lane}_indices": format_indices(indices) for lane, indices in lanes.items() if lane != "no_alert"},
                **{f"{lane}_count": len(indices) for lane, indices in lanes.items() if lane != "no_alert"},
                **metrics,
            }
            for source in SOURCE_CONFIGS:
                row[f"{source}_p_min"] = float(np.min([conformal_p_value(source_scores[source]["train_oof_scores"], score) for score in source_scores[source]["test_scores"]])) if source_scores[source]["test_scores"].size else ""
            rows.append(row)
    return rows


def run_one(args):
    name, current_sources, b2_sources, routing = args
    x_train, x_test, y_test = load_dataset_data(name)
    record = load_original_record(name, DB_PATH)
    record["y_test"] = y_test
    return make_prediction_record(name, x_train, x_test, record, current_sources, b2_sources, routing)


def aggregate(rows):
    def total(key):
        return int(sum(as_float(row.get(key), 0.0) for row in rows))

    def mean(key):
        return float(np.mean([as_float(row.get(key), 0.0) for row in rows])) if rows else 0.0

    hard_tp = total("hard_tp")
    hard_fp = total("hard_fp")
    standard_tp = total("standard_review_tp")
    standard_fp = total("standard_review_fp")
    priority_tp = total("priority_review_tp")
    priority_fp = total("priority_review_fp")
    candidate_counts = np.asarray([as_float(row.get("candidate_count"), 0) for row in rows], dtype=float)
    return {
        "datasets": len(rows),
        "hard_alerts": total("hard_count"),
        "hard_tp": hard_tp,
        "hard_fp": hard_fp,
        "hard_precision": hard_tp / max(1, hard_tp + hard_fp),
        "mean_hard_recall": mean("hard_recall"),
        "mean_hard_f1": mean("hard_f1"),
        "standard_review_candidates": total("standard_review_count"),
        "standard_review_tp": standard_tp,
        "standard_review_fp": standard_fp,
        "standard_review_precision": standard_tp / max(1, standard_tp + standard_fp),
        "priority_review_candidates": total("priority_review_count"),
        "priority_review_tp": priority_tp,
        "priority_review_fp": priority_fp,
        "priority_review_precision": priority_tp / max(1, priority_tp + priority_fp),
        "mean_combined_f1": mean("combined_f1"),
        "candidate_mean": float(np.mean(candidate_counts)) if len(candidate_counts) else 0.0,
        "candidate_median": float(np.median(candidate_counts)) if len(candidate_counts) else 0.0,
        "candidate_p90": float(np.percentile(candidate_counts, 90)) if len(candidate_counts) else 0.0,
        "candidate_p95": float(np.percentile(candidate_counts, 95)) if len(candidate_counts) else 0.0,
        "candidate_max": int(np.max(candidate_counts)) if len(candidate_counts) else 0,
        "candidate_zero_datasets": int(np.sum(candidate_counts == 0)),
        "n_train_lt5_datasets": int(sum(int(as_float(row.get("n_train"), 0)) < 5 for row in rows)),
        "resolution_limited_datasets": int(sum(int(as_float(row.get("resolution_limited_at_alpha"), 0)) for row in rows)),
        "retrospective_counterfactual": 1,
        "mean_combined_f1_scope": "human_assisted_diagnostic_only",
    }


def grain_audit(names):
    conn = sqlite3.connect(str(DB_PATH))
    rows = []
    for name in names:
        dataset = conn.execute("SELECT * FROM datasets WHERE name = ?", (name,)).fetchone()
        cols = [item[1] for item in conn.execute("PRAGMA table_info(datasets)").fetchall()]
        meta = dict(zip(cols, dataset))
        labels_train = [row[0] for row in conn.execute("SELECT label FROM instances i JOIN datasets d ON i.dataset_id=d.id WHERE d.name=? AND i.split='TRAIN' ORDER BY i.instance_index", (name,))]
        labels_test = [row[0] for row in conn.execute("SELECT label FROM instances i JOIN datasets d ON i.dataset_id=d.id WHERE d.name=? AND i.split='TEST' ORDER BY i.instance_index", (name,))]
        rows.append({
            "dataset": name,
            "train_instance_count": len(labels_train),
            "test_instance_count": len(labels_test),
            "train_label_values": " ".join(sorted(set(labels_train))),
            "test_label_values": " ".join(sorted(set(labels_test))),
            "values_dtype": "float32",
            "labels_blob_presence": "not_used_for_instance_policy",
            "instance_level_binary_label_compatible": int(set(labels_train + labels_test) <= {"0", "1"}),
            "minimum_attainable_conformal_p": 1.0 / (len(labels_train) + 1) if len(labels_train) else 1.0,
            "low_train_review_only": int(len(labels_train) < 5),
            "dataset_train_normal_count_metadata": meta.get("train_normal_count", ""),
            "dataset_test_total_count_metadata": meta.get("test_total_count", ""),
        })
    conn.close()
    return rows


def write_summaries(output_dir, rows):
    for variant, exp_id in (("exp149_current_source", "experiment_149_virtual_run_conformal_policy"), ("exp150_b2_source", "experiment_150_virtual_run_conformal_policy")):
        for alpha in ALPHAS:
            subset = [row for row in rows if row["variant"] == variant and float(row["alpha"]) == alpha]
            summary = aggregate(subset)
            summary.update({"experiment_id": exp_id, "variant": variant, "alpha": alpha, "scope": "retrospective counterfactual virtual wafer-run benchmark"})
            write_rows(output_dir / ("04_exp149_current_source_summary.csv" if variant.startswith("exp149") else "07_exp150_b2_source_summary.csv"), [summary] if alpha == ALPHAS[0] else read_rows(output_dir / ("04_exp149_current_source_summary.csv" if variant.startswith("exp149") else "07_exp150_b2_source_summary.csv")) + [summary])
        summary_rows = read_rows(output_dir / ("04_exp149_current_source_summary.csv" if variant.startswith("exp149") else "07_exp150_b2_source_summary.csv"))
        write_text(output_dir / ("05_exp149_current_source_summary.md" if variant.startswith("exp149") else "08_exp150_b2_source_summary.md"), "# " + exp_id + "\n\nAll alpha values were pre-registered and are reported together. No TEST metric selected alpha. Results are retrospective counterfactual virtual wafer-run benchmark results, not prospective equipment validation.\n\n```json\n" + json.dumps(summary_rows, indent=2, sort_keys=True) + "\n```")


def run(output_dir=OUTPUT_DIR, dataset_limit=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    publish_dashboard = dataset_limit is None
    baseline = read_rows(BASELINE_PATH)
    names = sorted(row["dataset_name"] for row in baseline)
    if dataset_limit:
        names = names[: int(dataset_limit)]
    current_sources, b2_sources = load_source_coverage()
    routing = load_routing_maps()
    audit = grain_audit(names)
    write_rows(output_dir / "02_virtual_run_grain_audit.csv", audit)
    if not all(int(row["instance_level_binary_label_compatible"]) for row in audit):
        raise SystemExit("BLOCKER: non-binary instance labels found; see 02_virtual_run_grain_audit.csv")
    if any(int(row["train_instance_count"]) < 5 for row in audit):
        write_text(output_dir / "BLOCKER_REPORT_VIRTUAL_RUN.md", "# Virtual-run blocker\n\nSome datasets have `n_train < 5`. Those datasets remain in the audit and are review-only; they do not generate autonomous Hard alerts. The run continues for the remaining datasets under the pre-registered rule.")
    checkpoint = output_dir / "prediction_checkpoint.jsonl"
    restored = {}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if line.strip():
                payload = json.loads(line)
                restored[payload["dataset"]] = payload["rows"]
    rows = [row for name in sorted(restored) if name in names for row in restored[name]]
    pending = [name for name in names if name not in restored]
    errors = []
    if publish_dashboard:
        update_external_live_run("running", output_dir, len(restored), len(names), len(rows), started_at)
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(run_one, (name, current_sources, b2_sources, routing[name])): name for name in pending}
        for completed, future in enumerate(as_completed(futures), 1):
            name = futures[future]
            try:
                result = future.result()
                rows.extend(result)
                with checkpoint.open("a") as handle:
                    handle.write(json.dumps({"dataset": name, "rows": result}, sort_keys=True) + "\n")
                if publish_dashboard:
                    update_external_live_run("running", output_dir, len(restored) + completed, len(names), len(rows), started_at)
            except Exception as exc:
                errors.append({"dataset": name, "error": repr(exc)})
            done = len(restored) + completed
            if done % 10 == 0 or done == len(names):
                print(f"Progress: [{done:4d}/{len(names):4d}] rows={len(rows)} errors={len(errors)}", flush=True)
    if errors:
        write_rows(output_dir / "errors.csv", errors)
        if publish_dashboard:
            update_external_live_run("failed", output_dir, len(restored) + len(rows) // 8, len(names), len(rows), started_at, error=f"{len(errors)} dataset errors")
        raise SystemExit(f"{len(errors)} dataset errors; see {output_dir / 'errors.csv'}")
    rows.sort(key=lambda row: (row["variant"], float(row["alpha"]), row["dataset_name"]))
    write_rows(output_dir / "03_exp149_current_source_results.csv", [row for row in rows if row["variant"] == "exp149_current_source"])
    write_rows(output_dir / "06_exp150_b2_source_results.csv", [row for row in rows if row["variant"] == "exp150_b2_source"])
    write_summaries(output_dir, rows)
    comparison = []
    for variant in ("exp149_current_source", "exp150_b2_source"):
        for alpha in ALPHAS:
            comparison.append({"variant": variant, "alpha": alpha, **aggregate([row for row in rows if row["variant"] == variant and float(row["alpha"]) == alpha])})
    write_rows(output_dir / "09_alpha_sensitivity_comparison.csv", comparison)
    write_rows(output_dir / "10_dataset_level_candidate_diff.csv", [{
        "dataset": name,
        "alpha": alpha,
        "exp149_candidates": next(row["candidate_indices"] for row in rows if row["dataset_name"] == name and row["variant"] == "exp149_current_source" and float(row["alpha"]) == alpha),
        "exp150_candidates": next(row["candidate_indices"] for row in rows if row["dataset_name"] == name and row["variant"] == "exp150_b2_source" and float(row["alpha"]) == alpha),
    } for name in names for alpha in ALPHAS])
    write_text(output_dir / "00_preregistration.md", "# Exp149/150 virtual-run conformal policy\n\nCopied from the pre-registered virtual wafer-run policy. Alpha values are `0.005, 0.01, 0.02, 0.05`; no TEST metric selected a policy. Existing Exp145/148 and Exp137 outputs were not overwritten. Results are retrospective counterfactual and not prospective equipment validation.")
    manifest = []
    for path in sorted(output_dir.glob("*")):
        if path.is_file() and path.name != "MANIFEST.csv":
            manifest.append({"file": str(path.relative_to(output_dir)), "sha256": sha256_file(path), "retrospective_or_prospective": "retrospective_counterfactual", "autonomous_or_human_assisted": "autonomous_hard_and_human_assisted_review_separate"})
    write_rows(output_dir / "MANIFEST.csv", manifest)
    write_text(output_dir / "MANIFEST.md", "# Exp149/150 manifest\n\nAll files are retrospective counterfactual virtual wafer-run benchmark artifacts. Hard alert metrics are autonomous; combined F1 is human-assisted diagnostic only.\n\n" + "\n".join(f"- `{row['file']}`: `{row['sha256']}`" for row in manifest))
    if publish_dashboard:
        update_external_live_run(
            "completed",
            output_dir,
            len(names),
            len(names),
            len(rows),
            started_at,
            result_summary={"alpha_sensitivity": comparison, "errors": 0},
        )
    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    run(args.output_dir, args.dataset_limit)


if __name__ == "__main__":
    main()
