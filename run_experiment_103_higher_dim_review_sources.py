from __future__ import annotations

import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from run_experiment_40_original_score_normalization_sweep import (
    count_cap_threshold as exp40_count_cap_threshold,
    score_pair_for_config as exp40_score_pair_for_config,
)
from run_experiment_60_62_rocket_imaging_selector_variants import (
    evaluate_indices,
    prediction_bundle,
    results_path,
    summary_path,
)
from run_experiment_89_74d_with_exp84_candidate import as_float, format_indices, parse_indices
from run_model_hard_research_experiments import (
    prepare_series_pair_for_scale,
    score_pair_for_config as research_score_pair_for_config,
)
from run_original_improvement_experiment import DB_PATH, load_original_record, target_len_for_record
from run_rank_ensemble_calibration import align_series_lengths, load_dataset_data, z_normalize


DATA_DIR = Path("/Users/minho/Documents/Dataset")
EXPERIMENT_ID = "experiment_103_higher_dim_review_sources"
EXP93_PATH = DATA_DIR / "experiment_93_nonpos_candidate_reranker_results.csv"
EXP95_PATH = DATA_DIR / "experiment_95_topk_review_tier_results.csv"
EXP93_SELECTOR = "nonpos_weak_alert_replace"
EXP95_SELECTOR = "review_top1_strict"
STDOUT_LOG = DATA_DIR / f"{EXPERIMENT_ID}_stdout.log"
WORKERS = int(os.environ.get("EXP103_WORKERS", "4"))

SOURCE_RATE = 0.02
MAX_SOURCE_TOP = 12

SOURCE_CONFIGS = {
    "spectrogram_pca64": {
        "name": "train_global_minmax_clip_spectrogram_32_pca64_knn3",
        "kind": "imaging_knn",
        "image": "spectrogram",
        "series_scale": "train_global_minmax_clip",
        "size": 32,
        "pca": 64,
        "neighbors": 3,
    },
    "spectrogram_pca128": {
        "name": "train_global_minmax_clip_spectrogram_32_pca128_knn3",
        "kind": "imaging_knn",
        "image": "spectrogram",
        "series_scale": "train_global_minmax_clip",
        "size": 32,
        "pca": 128,
        "neighbors": 3,
    },
    "glcm_rp_pca64": {
        "name": "glcm_rp_32_pca64_knn3",
        "kind": "imaging_knn",
        "image": "rp",
        "feature_extractor": "glcm",
        "size": 32,
        "pca": 64,
        "neighbors": 3,
    },
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


def top_score_indices(bundle, count=MAX_SOURCE_TOP):
    scores = np.asarray(bundle["test_scores"], dtype=np.float64)
    order = np.argsort(scores)[::-1]
    return [int(idx) for idx in order[: min(count, len(order))]]


def cap_ranked(indices, limit):
    out = []
    for idx in indices:
        idx = int(idx)
        if idx not in out:
            out.append(idx)
        if len(out) >= limit:
            break
    return set(out)


def top_candidate_index(row):
    value = row.get("top_candidate")
    if value in (None, "", "nan"):
        return None
    try:
        if np.isnan(float(value)):
            return None
    except ValueError:
        return None
    return int(float(value))


def weak_exp93_signal(base_row):
    predicted = int(as_float(base_row.get("predicted_count")))
    train_exceed = as_float(base_row.get("train_exceed_rate"), 1.0)
    max_train_exceed = as_float(base_row.get("max_candidate_train_exceed_rate"), 1.0)
    support = as_float(base_row.get("top_candidate_support"))
    score_gain = as_float(base_row.get("top_candidate_score_gain"))
    return (
        predicted <= 1
        and train_exceed <= 0.015
        and max_train_exceed <= 0.015
        and support >= 4
        and score_gain <= 0.30
    )


def agreement_guard(base_row, hard_indices, existing_review, candidate_indices):
    top_idx = top_candidate_index(base_row)
    agreement_pool = set(hard_indices) | set(existing_review)
    if top_idx is not None:
        agreement_pool.add(top_idx)
    return bool(set(candidate_indices) & agreement_pool)


def index_metrics(y_test, selected_indices):
    selected = set(int(idx) for idx in selected_indices)
    true = {idx for idx, value in enumerate(y_test) if int(value) == 1}
    tp = len(selected & true)
    fp = len(selected - true)
    fn = len(true - selected)
    denom = 2 * tp + fp + fn
    return {
        "predicted_count": len(selected),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "f1": (2 * tp / denom) if denom else 0.0,
    }


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


def compute_imaging_source(dataset_name, source_name, config, y_test):
    record = load_original_record(dataset_name, DB_PATH)
    target_len = min(max(8, target_len_for_record(record, "actual_median")), 2048)
    X_train_raw = align_series_lengths(record["train_series"], target_len)
    X_test_raw = align_series_lengths(record["test_series"], target_len)
    X_train_z = z_normalize(X_train_raw).astype(np.float32)
    X_test_z = z_normalize(X_test_raw).astype(np.float32)
    X_train, X_test = prepare_series_pair_for_scale(
        config.get("series_scale", "per_series_z"),
        X_train_raw,
        X_test_raw,
        X_train_z,
        X_test_z,
    )
    train_scores, test_scores = research_score_pair_for_config(X_train, X_test, target_len, config, record)
    threshold, q_effective, cap_target = exp40_count_cap_threshold(train_scores, SOURCE_RATE)
    return prediction_bundle(source_name, y_test, train_scores, test_scores, threshold, q_effective, cap_target)


def compute_rocket_source(dataset_name, y_test):
    X_train, X_test, _ = load_dataset_data(dataset_name)
    seq_len = X_train.shape[1]
    X_train = z_normalize(X_train).astype(np.float32)
    X_test = z_normalize(X_test).astype(np.float32)
    config = {
        "name": "rocket_512_knn3_local_gap",
        "kind": "density_knn",
        "num_kernels": 512,
        "neighbors": 3,
        "mode": "local_gap",
    }
    train_scores, test_scores = exp40_score_pair_for_config(X_train, X_test, seq_len, config, {})
    threshold, q_effective, cap_target = exp40_count_cap_threshold(train_scores, SOURCE_RATE)
    return prediction_bundle("rocket_512_local_gap", y_test, train_scores, test_scores, threshold, q_effective, cap_target)


def source_bundle_map(dataset_name, y_test):
    out = {}
    for source_name, config in SOURCE_CONFIGS.items():
        out[source_name] = compute_imaging_source(dataset_name, source_name, config, y_test)
    out["rocket_512_local_gap"] = compute_rocket_source(dataset_name, y_test)
    return out


def make_review_row(name, y_test, base_row, selector, review_indices, reason, extras):
    hard = parse_indices(base_row.get("selected_indices"))
    rm = review_metrics(y_test, hard, review_indices)
    out = dict(base_row)
    out.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "dataset_name": name,
            "selector_name": selector,
            "config_name": selector,
            "selector_reason": reason,
            "score_family": "higher_dim_review_sources",
            "threshold_method": "selector",
            "selected_indices": format_indices(hard),
            "hard_selected_indices": format_indices(hard),
            "review_candidate_indices": format_indices(review_indices),
            "review_enabled": 1,
            "hard_replaced": 0,
            "hard_f1": as_float(base_row.get("f1")),
        }
    )
    out.update(rm)
    out.update(extras)
    return out


def make_hard_row(name, y_test, base_row, selector, selected_indices, reason, extras):
    metrics = index_metrics(y_test, selected_indices)
    out = dict(base_row)
    out.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "dataset_name": name,
            "selector_name": selector,
            "config_name": selector,
            "selector_reason": reason,
            "score_family": "higher_dim_review_sources",
            "threshold_method": "selector",
            "selected_indices": format_indices(selected_indices),
            "predicted_count": metrics["predicted_count"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "f1": metrics["f1"],
            "combined_f1": metrics["f1"],
            "combined_zero_f1": int(metrics["f1"] == 0.0),
            "review_candidate_count": 0,
            "review_tp": 0,
            "review_fp": 0,
            "review_enabled": 0,
        }
    )
    out.update(extras)
    return out


def choose_rows_for_dataset(args):
    name, base_row, existing_review_text = args
    _, _, y_test = load_dataset_data(name)
    hard = parse_indices(base_row.get("selected_indices"))
    existing_review = parse_indices(existing_review_text)
    weak = int(weak_exp93_signal(base_row))
    bundles = source_bundle_map(name, y_test)

    source_orders = {source: top_score_indices(bundle) for source, bundle in bundles.items()}
    source_candidates = {source: set(bundle["indices"]) for source, bundle in bundles.items()}
    source_agrees = {
        source: int(agreement_guard(base_row, hard, existing_review, source_candidates[source]))
        for source in bundles
    }
    active_sources = {
        source: bool(weak and source_agrees[source] and source_candidates[source])
        for source in bundles
    }

    def candidates_for(source_names):
        ordered = []
        active = []
        for source in source_names:
            if active_sources.get(source):
                active.append(source)
                for idx in source_orders[source]:
                    if idx in source_candidates[source] and idx not in hard:
                        ordered.append(idx)
        return cap_ranked(ordered, 3), active

    pca64_review, pca64_active = candidates_for(["spectrogram_pca64", "glcm_rp_pca64"])
    pca_review, pca_active = candidates_for(["spectrogram_pca64", "spectrogram_pca128", "glcm_rp_pca64"])
    all_review, all_active = candidates_for(
        ["spectrogram_pca64", "spectrogram_pca128", "glcm_rp_pca64", "rocket_512_local_gap"]
    )

    single_active = [source for source in all_active if active_sources.get(source)]
    hard_replace = len(single_active) == 1
    hard_indices = source_candidates[single_active[0]] if hard_replace else hard

    common = {
        "exp93_weak_signal": weak,
        "source_agreement_spectrogram_pca64": source_agrees["spectrogram_pca64"],
        "source_agreement_spectrogram_pca128": source_agrees["spectrogram_pca128"],
        "source_agreement_glcm_rp_pca64": source_agrees["glcm_rp_pca64"],
        "source_agreement_rocket_512_local_gap": source_agrees["rocket_512_local_gap"],
        "source_indices_spectrogram_pca64": format_indices(source_candidates["spectrogram_pca64"]),
        "source_indices_spectrogram_pca128": format_indices(source_candidates["spectrogram_pca128"]),
        "source_indices_glcm_rp_pca64": format_indices(source_candidates["glcm_rp_pca64"]),
        "source_indices_rocket_512_local_gap": format_indices(source_candidates["rocket_512_local_gap"]),
    }

    return [
        make_hard_row(
            name,
            y_test,
            base_row,
            "baseline_exp93_hard_only",
            hard,
            "control: Exp93 hard alert default",
            {**common, "selected_feature_sources": "none", "hard_replaced": 0},
        ),
        make_review_row(
            name,
            y_test,
            base_row,
            "review_pca64_sources_when_exp93_weak",
            pca64_review,
            "review only: PCA64 spectrogram/GLCM candidates when Exp93 weak and source agrees",
            {**common, "selected_feature_sources": ";".join(pca64_active) or "none"},
        ),
        make_review_row(
            name,
            y_test,
            base_row,
            "review_pca64_pca128_sources_when_exp93_weak",
            pca_review,
            "review only: PCA64/PCA128 candidates when Exp93 weak and source agrees",
            {**common, "selected_feature_sources": ";".join(pca_active) or "none"},
        ),
        make_review_row(
            name,
            y_test,
            base_row,
            "review_all_higher_dim_sources_when_exp93_weak",
            all_review,
            "review only: all higher-dim candidates when Exp93 weak and source agrees",
            {**common, "selected_feature_sources": ";".join(all_active) or "none"},
        ),
        make_hard_row(
            name,
            y_test,
            base_row,
            "hard_guard_single_higher_dim_source_when_exp93_weak",
            hard_indices,
            "diagnostic: hard replace only when exactly one higher-dim source is active",
            {
                **common,
                "selected_feature_sources": single_active[0] if hard_replace else "none",
                "hard_replaced": int(hard_replace),
            },
        ),
    ]


def summarize(rows):
    out = []
    for selector in sorted({row["selector_name"] for row in rows}):
        subset = [row for row in rows if row["selector_name"] == selector]
        vals = lambda key: [as_float(row.get(key)) for row in subset]
        f1s = vals("f1")
        combined_f1s = vals("combined_f1")
        hard_tp = sum(vals("tp"))
        hard_fp = sum(vals("fp"))
        review_tp = sum(vals("review_tp"))
        review_fp = sum(vals("review_fp"))
        review_count = sum(vals("review_candidate_count"))
        out.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "selector_name": selector,
                "config_name": selector,
                "threshold_method": "selector",
                "num_datasets": len(subset),
                "num_families": len({row["family"] for row in subset}),
                "mean_auc_roc": float(np.mean(vals("auc_roc"))),
                "mean_auc_pr": float(np.mean(vals("auc_pr"))),
                "mean_f1": float(np.mean(f1s)),
                "median_f1": float(np.median(f1s)),
                "zero_f1_count": sum(1 for value in f1s if value == 0.0),
                "mean_combined_f1": float(np.mean(combined_f1s)),
                "combined_zero_f1_count": sum(1 for value in combined_f1s if value == 0.0),
                "mean_predicted_count": float(np.mean(vals("predicted_count"))),
                "mean_tp": float(np.mean(vals("tp"))),
                "mean_fp": float(np.mean(vals("fp"))),
                "alert_precision": hard_tp / max(1.0, hard_tp + hard_fp),
                "mean_review_candidate_count": float(np.mean(vals("review_candidate_count"))),
                "review_candidates_total": int(review_count),
                "review_candidates_per_100_datasets": float(100.0 * review_count / max(1, len(subset))),
                "review_hit_datasets": sum(1 for row in subset if as_float(row.get("review_hit")) > 0),
                "review_precision": review_tp / max(1.0, review_tp + review_fp),
                "review_tp_total": int(review_tp),
                "review_fp_total": int(review_fp),
                "hard_replaced_count": sum(1 for row in subset if as_float(row.get("hard_replaced")) > 0),
                "mean_oracle_f1": float(np.mean(vals("oracle_f1"))),
            }
        )
    return sorted(out, key=lambda row: (row["mean_combined_f1"], -row["review_fp_total"]), reverse=True)


def run_experiment(dataset_limit=None):
    base_rows = {
        row["dataset_name"]: row
        for row in read_rows(EXP93_PATH)
        if row.get("selector_name") == EXP93_SELECTOR
    }
    review_rows = {
        row["dataset_name"]: row
        for row in read_rows(EXP95_PATH)
        if row.get("selector_name") == EXP95_SELECTOR
    }
    names = sorted(base_rows)
    if dataset_limit:
        names = names[: int(dataset_limit)]
    tasks = [(name, base_rows[name], review_rows.get(name, {}).get("review_candidate_indices", "")) for name in names]
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
    STDOUT_LOG.write_text(f"{EXPERIMENT_ID} finished rows={len(rows)} datasets={len(names)}\n{summary[0] if summary else ''}\n")
    print(f"{EXPERIMENT_ID} finished rows={len(rows)} datasets={len(names)}")
    if summary:
        print(summary[0])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-limit", type=int, default=None)
    args = parser.parse_args()
    run_experiment(args.dataset_limit)
