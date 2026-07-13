import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score

from run_experiment_40_original_score_normalization_sweep import (
    CONFIGS as EXP40_CONFIGS,
    count_cap_threshold as exp40_count_cap_threshold,
    score_pair_for_config as exp40_score_pair_for_config,
)
from run_experiment_29_train_normal_threshold_calibration import train_false_positive_stats
from run_model_hard_research_experiments import (
    prepare_series_pair_for_scale,
    score_pair_for_config as imaging_score_pair_for_config,
)
from run_original_improvement_experiment import DATA_DIR, DB_PATH, load_original_record, target_len_for_record
from run_rank_ensemble_calibration import align_series_lengths, load_dataset_data, z_normalize
from run_rank_threshold_calibration import top_k_oracle_f1


EXPERIMENTS = {
    "experiment_60_selector_fp_guard_variants": {
        "label": "Exp 60 - ROCKET imaging selector FP guard variants",
        "kind": "csv_selector",
    },
    "experiment_61_selector_index_agreement": {
        "label": "Exp 61 - ROCKET imaging selector sample-index agreement",
        "kind": "index_selector",
    },
    "experiment_62_selector_guarded_index_agreement": {
        "label": "Exp 62 - ROCKET imaging selector guarded index agreement",
        "kind": "index_selector",
    },
    "experiment_63_guarded_cap_sweep": {
        "label": "Exp 63 - Guarded index agreement cap sweep",
        "kind": "index_selector",
    },
    "experiment_64_guarded_with_fallback": {
        "label": "Exp 64 - Guarded selector with fallback",
        "kind": "index_selector",
    },
    "experiment_65_confidence_tier_selector": {
        "label": "Exp 65 - Confidence tier selector",
        "kind": "index_selector",
    },
    "experiment_66_train_normal_alert_budget": {
        "label": "Exp 66 - Train-normal alert budget selector",
        "kind": "index_selector",
    },
    "experiment_67_hard_family_fallback_selector": {
        "label": "Exp 67 - Hardness signal fallback selector",
        "kind": "index_selector",
    },
    "experiment_68_final_operational_selector": {
        "label": "Exp 68 - Final operational selector candidates",
        "kind": "index_selector",
    },
    "experiment_68b_final_operational_fallback_sweep": {
        "label": "Exp 68b - Final operational fallback sweep",
        "kind": "index_selector",
    },
    "experiment_69_operational_train_exceed_calibration": {
        "label": "Exp 69 - Operational train exceed calibration",
        "kind": "calibrated_selector",
    },
    "experiment_69b_no_prediction_fallback_calibration": {
        "label": "Exp 69b - No-prediction fallback calibration",
        "kind": "calibrated_fallback_selector",
    },
    "experiment_70_zero_mode_family_repair_selector": {
        "label": "Exp 70 - Zero-mode family repair selector",
        "kind": "zero_mode_selector",
    },
    "experiment_71a_large_data_rocket_fallback": {
        "label": "Exp 71a - Large-data ROCKET fallback repair",
        "kind": "large_data_repair_selector",
    },
    "experiment_71b_large_data_rocket_review_tier": {
        "label": "Exp 71b - Large-data ROCKET review-tier repair",
        "kind": "large_data_repair_selector",
    },
    "experiment_72a_large_data_rank_ensemble": {
        "label": "Exp 72a - Large-data rank ensemble repair",
        "kind": "large_data_repair_selector",
    },
    "experiment_72b_large_data_source_disagreement": {
        "label": "Exp 72b - Large-data source disagreement repair",
        "kind": "large_data_repair_selector",
    },
    "experiment_73a_large_rank_rocket_guard": {
        "label": "Exp 73a - Large-data rank ROCKET-overlap guard",
        "kind": "large_data_repair_selector",
    },
    "experiment_73b_large_rank_two_model_guard": {
        "label": "Exp 73b - Large-data rank two-model guard",
        "kind": "large_data_repair_selector",
    },
    "experiment_73c_large_rank_budget_guard": {
        "label": "Exp 73c - Large-data rank budget guard",
        "kind": "large_data_repair_selector",
    },
    "experiment_73d_large_rank_combined_guard": {
        "label": "Exp 73d - Large-data rank combined guard",
        "kind": "large_data_repair_selector",
    },
    "experiment_74a_large_rank_margin_guard": {
        "label": "Exp 74a - Large-data rank confidence margin guard",
        "kind": "large_data_repair_selector",
    },
    "experiment_74b_large_rank_family_budget": {
        "label": "Exp 74b - Large-data family-aware rank budget",
        "kind": "large_data_repair_selector",
    },
    "experiment_74c_large_rank_margin_family_guard": {
        "label": "Exp 74c - Large-data margin and family guard",
        "kind": "large_data_repair_selector",
    },
    "experiment_74d_large_rank_review_tier_split": {
        "label": "Exp 74d - Large-data alert/review tier split",
        "kind": "large_data_repair_selector",
    },
}

CSV_CANDIDATES = {
    "rocket_exp40": {
        "path": DATA_DIR / "experiment_40_original_score_normalization_sweep_results.csv",
        "config_name": "rocket_256_knn3_local_gap",
        "threshold_method": "count_cap_3pct",
    },
    "exp55_best": {
        "path": DATA_DIR / "experiment_55_imaging_scaling_sweep_results.csv",
        "config_name": "train_global_minmax_clip_spectrogram_32_pca32_knn3",
        "threshold_method": "count_cap_3pct",
    },
    "exp56_best": {
        "path": DATA_DIR / "experiment_56_imaging_glcm_texture_probe_results.csv",
        "config_name": "glcm_rp_32_pca32_knn3",
        "threshold_method": "count_cap_2pct",
    },
}

IMAGING_CONFIGS = {
    "exp55_best": {
        "name": "train_global_minmax_clip_spectrogram_32_pca32_knn3",
        "kind": "imaging_knn",
        "image": "spectrogram",
        "series_scale": "train_global_minmax_clip",
        "size": 32,
        "pca": 32,
        "neighbors": 3,
    },
    "exp56_best": {
        "name": "glcm_rp_32_pca32_knn3",
        "kind": "imaging_knn",
        "image": "rp",
        "feature_extractor": "glcm",
        "size": 32,
        "pca": 32,
        "neighbors": 3,
    },
}


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default=0):
    return int(round(as_float(value, default)))


def results_path(exp_id):
    return DATA_DIR / f"{exp_id}_results.csv"


def summary_path(exp_id):
    return DATA_DIR / f"{exp_id}_summary.csv"


def log_path(exp_id):
    return DATA_DIR / f"{exp_id}_stdout.log"


def read_csv_candidates():
    selected = {}
    for name, spec in CSV_CANDIDATES.items():
        rows = {}
        with spec["path"].open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("config_name") == spec["config_name"] and row.get("threshold_method") == spec["threshold_method"]:
                    rows[row["dataset_name"]] = row
        if not rows:
            raise RuntimeError(f"No rows found for {name}")
        selected[name] = rows
    common = set.intersection(*(set(rows) for rows in selected.values()))
    if len(common) != 1117:
        raise RuntimeError(f"Expected 1117 common datasets, got {len(common)}")
    return selected, sorted(common)


def operational_budget(row, rate):
    return max(1, int(math.ceil(max(1, as_int(row.get("test_size"))) * rate)))


def predicted_count(row):
    return as_int(row.get("predicted_count"))


def train_exceed_rate(row):
    return as_float(row.get("train_exceed_rate"))


def passes_guard(row, rate=0.02, train_rate=0.015, min_count=1):
    count = predicted_count(row)
    return min_count <= count <= operational_budget(row, rate) and train_exceed_rate(row) <= train_rate


def select_exp60(strategy, candidates, dataset_name):
    rocket = candidates["rocket_exp40"][dataset_name]
    exp55 = candidates["exp55_best"][dataset_name]
    exp56 = candidates["exp56_best"][dataset_name]
    if strategy == "agreement_count_v1_reference":
        nonzero = {name for name, row in [("rocket_exp40", rocket), ("exp55_best", exp55), ("exp56_best", exp56)] if predicted_count(row) > 0}
        if "rocket_exp40" in nonzero and ({"exp55_best", "exp56_best"} & nonzero):
            return "rocket_exp40", "rocket_agrees_with_imaging"
        if {"exp55_best", "exp56_best"} <= nonzero and "rocket_exp40" not in nonzero:
            return min(["exp55_best", "exp56_best"], key=lambda name: predicted_count(candidates[name][dataset_name])), "imaging_pair_agrees_rocket_zero"
        if nonzero == {"rocket_exp40"}:
            return "rocket_exp40", "rocket_only_nonzero"
        return "none", "no_count_agreement"
    if strategy == "agreement_fp_guard_2pct":
        selected, reason = select_exp60("agreement_count_v1_reference", candidates, dataset_name)
        if selected == "none":
            return selected, reason
        if passes_guard(candidates[selected][dataset_name], rate=0.02, train_rate=0.015):
            return selected, reason + "_guard_pass"
        if selected != "rocket_exp40" and passes_guard(rocket, rate=0.03, train_rate=0.035):
            return "rocket_exp40", "fallback_rocket_after_imaging_guard_fail"
        return "none", reason + "_guard_fail"
    if strategy == "agreement_fp_guard_3pct":
        selected, reason = select_exp60("agreement_count_v1_reference", candidates, dataset_name)
        if selected == "none":
            return selected, reason
        if passes_guard(candidates[selected][dataset_name], rate=0.03, train_rate=0.02):
            return selected, reason + "_guard_pass"
        return "none", reason + "_guard_fail"
    if strategy == "rocket_default":
        return "rocket_exp40", "baseline_default_rocket"
    raise ValueError(strategy)


def empty_row(reference, exp_id, strategy, reason):
    row = dict(reference)
    row.update(
        {
            "experiment_id": exp_id,
            "config_name": strategy,
            "selector_name": strategy,
            "selected_candidate": "none",
            "selector_reason": reason,
            "threshold_method": "selector",
            "score_family": "selector",
            "predicted_count": "0",
            "tp": "0",
            "fp": "0",
            "fn": str(as_int(reference.get("anomaly_count"))),
            "f1": "0.0",
            "auc_roc": "0.5",
            "auc_pr": str(as_int(reference.get("anomaly_count")) / max(1, as_int(reference.get("test_size")))),
            "oracle_f1": "0.0",
        }
    )
    return row


def materialize_csv_selection(exp_id, strategy, selected, reason, candidates, dataset_name):
    reference = candidates["rocket_exp40"][dataset_name]
    if selected == "none":
        return empty_row(reference, exp_id, strategy, reason)
    row = dict(candidates[selected][dataset_name])
    row.update(
        {
            "experiment_id": exp_id,
            "config_name": strategy,
            "selector_name": strategy,
            "selected_candidate": selected,
            "selector_reason": reason,
            "candidate_config_name": row.get("config_name", ""),
            "candidate_threshold_method": row.get("threshold_method", ""),
            "threshold_method": "selector",
            "score_family": "selector",
        }
    )
    return row


def run_exp60(exp_id, dataset_limit=None):
    candidates, datasets = read_csv_candidates()
    if dataset_limit:
        datasets = datasets[:dataset_limit]
    strategies = ["rocket_default", "agreement_count_v1_reference", "agreement_fp_guard_2pct", "agreement_fp_guard_3pct"]
    rows = []
    for dataset_name in datasets:
        for strategy in strategies:
            selected, reason = select_exp60(strategy, candidates, dataset_name)
            rows.append(materialize_csv_selection(exp_id, strategy, selected, reason, candidates, dataset_name))
    return rows


def metric_summary(y_true, scores):
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    try:
        auc_roc = roc_auc_score(y_true, scores)
    except ValueError:
        auc_roc = 0.5
    precision, recall, _ = precision_recall_curve(y_true, scores)
    return {
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc(recall, precision)),
        "oracle_f1": float(top_k_oracle_f1(y_true, scores)),
    }


def evaluate_indices(y_true, scores, selected_indices):
    y_true = np.asarray(y_true, dtype=np.int64)
    selected = np.zeros(len(y_true), dtype=np.int64)
    selected[list(selected_indices)] = 1
    tp = int(((selected == 1) & (y_true == 1)).sum())
    fp = int(((selected == 1) & (y_true == 0)).sum())
    fn = int(((selected == 0) & (y_true == 1)).sum())
    metrics = metric_summary(y_true, scores)
    return {
        "predicted_count": int(selected.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "f1": float(f1_score(y_true, selected, zero_division=0)),
        **metrics,
    }


def load_candidate_predictions(dataset_name, threshold_rates=None):
    threshold_rates = threshold_rates or {}
    X_train, X_test, y_test = load_dataset_data(dataset_name)
    seq_len = X_train.shape[1]
    X_train_z = z_normalize(X_train).astype(np.float32)
    X_test_z = z_normalize(X_test).astype(np.float32)
    rocket_config = [c for c in EXP40_CONFIGS if c["name"] == "rocket_256_knn3_local_gap"][0]
    rocket_train, rocket_test = exp40_score_pair_for_config(X_train_z, X_test_z, seq_len, rocket_config, {})
    rocket_rate = threshold_rates.get("rocket_exp40", 0.03)
    rocket_threshold, rocket_q, rocket_cap = exp40_count_cap_threshold(rocket_train, rocket_rate)

    record = load_original_record(dataset_name, DB_PATH)
    target_len = min(max(8, target_len_for_record(record, "actual_median")), 2048)
    X_train_raw = align_series_lengths(record["train_series"], target_len)
    X_test_raw = align_series_lengths(record["test_series"], target_len)
    X_train_z2 = z_normalize(X_train_raw).astype(np.float32)
    X_test_z2 = z_normalize(X_test_raw).astype(np.float32)

    out = {}
    out["rocket_exp40"] = prediction_bundle("rocket_exp40", y_test, rocket_train, rocket_test, rocket_threshold, rocket_q, rocket_cap)
    for name, config in IMAGING_CONFIGS.items():
        Xtr, Xte = prepare_series_pair_for_scale(config.get("series_scale", "per_series_z"), X_train_raw, X_test_raw, X_train_z2, X_test_z2)
        train_scores, test_scores = imaging_score_pair_for_config(Xtr, Xte, target_len, config, record)
        rate = threshold_rates.get(name, 0.03 if name == "exp55_best" else 0.02)
        threshold, q_effective, cap_target = exp40_count_cap_threshold(train_scores, rate)
        out[name] = prediction_bundle(name, y_test, train_scores, test_scores, threshold, q_effective, cap_target)
    return record, y_test, out


def prediction_bundle(name, y_test, train_scores, test_scores, threshold, q_effective, cap_target):
    indices = set(np.flatnonzero(np.asarray(test_scores) > threshold).astype(int).tolist())
    train_exceed_count, train_exceed = train_false_positive_stats(train_scores, threshold)
    return {
        "name": name,
        "indices": indices,
        "train_scores": np.asarray(train_scores),
        "test_scores": np.asarray(test_scores),
        "threshold": threshold,
        "q_effective": q_effective,
        "cap_target": cap_target,
        "train_exceed_count": train_exceed_count,
        "train_exceed_rate": train_exceed,
        **evaluate_indices(y_test, test_scores, indices),
    }


def selection(indices, score_source="exp55_best", reason=""):
    return {"indices": set(indices), "score_source": score_source, "reason": reason}


def bundle_counts(bundles):
    return {name: len(bundle["indices"]) for name, bundle in bundles.items()}


def compact_counts_reason(bundles):
    counts = bundle_counts(bundles)
    return "counts_" + "_".join(f"{name}:{counts[name]}" for name in ["rocket_exp40", "exp55_best", "exp56_best"])


def cap_count_for_rate(bundle, rate):
    return max(1, int(math.ceil(len(bundle["test_scores"]) * rate)))


def cap_indices_count(indices, bundle, count):
    count = max(1, int(count))
    if len(indices) <= count:
        return set(indices)
    scores = bundle["test_scores"]
    return set(sorted(indices, key=lambda idx: scores[idx], reverse=True)[:count])


def cap_indices_min_rate_max(indices, bundle, rate, max_count):
    return cap_indices_count(indices, bundle, min(cap_count_for_rate(bundle, rate), max_count))


def train_normal_budget(bundles, multiplier=1.5, min_rate=0.005, max_rate=0.03, max_count=None):
    rates = [as_float(bundle.get("train_exceed_rate")) for bundle in bundles.values()]
    rate = min(max(float(np.mean(rates)) * multiplier, min_rate), max_rate)
    count = cap_count_for_rate(bundles["rocket_exp40"], rate)
    if max_count is not None:
        count = min(count, max_count)
    return max(1, count), rate


def choose_indices(exp_id, bundles):
    rocket = bundles["rocket_exp40"]["indices"]
    exp55 = bundles["exp55_best"]["indices"]
    exp56 = bundles["exp56_best"]["indices"]
    two_of_three_raw = indices_at_least(2, [rocket, exp55, exp56])
    three_of_three_raw = indices_at_least(3, [rocket, exp55, exp56])
    imaging_pair = exp55 & exp56
    counts_reason = compact_counts_reason(bundles)
    if exp_id == "experiment_61_selector_index_agreement":
        return {
            "index_2of3": selection(two_of_three_raw, "exp55_best", "at_least_two_models_agree"),
            "rocket_when_any_index_overlap": selection(
                rocket if (rocket & exp55) or (rocket & exp56) else set(),
                "rocket_exp40",
                "rocket_used_when_any_index_overlap",
            ),
            "imaging_pair_when_rocket_zero": selection(
                imaging_pair if not rocket else rocket,
                "exp56_best" if not rocket else "rocket_exp40",
                "imaging_pair_fallback_when_rocket_empty" if not rocket else "rocket_nonempty",
            ),
            "rocket_default_index_reference": selection(rocket, "rocket_exp40", "baseline_rocket"),
        }
    if exp_id == "experiment_62_selector_guarded_index_agreement":
        two_of_three = cap_indices(two_of_three_raw, bundles["rocket_exp40"], 0.02)
        if rocket and len(rocket) <= max(1, math.ceil(len(bundles["rocket_exp40"]["test_scores"]) * 0.03)):
            guarded_rocket = rocket
        else:
            guarded_rocket = set()
        if not rocket:
            pair = cap_indices(exp55 & exp56, bundles["exp56_best"], 0.02)
        else:
            pair = guarded_rocket
        return {
            "guarded_index_2of3": selection(two_of_three, "exp55_best", f"2of3_cap_2pct_{counts_reason}"),
            "guarded_rocket_or_imaging_pair": selection(pair, "rocket_exp40" if rocket else "exp56_best", f"rocket_or_pair_guard_{counts_reason}"),
            "guarded_rocket_index_reference": selection(guarded_rocket, "rocket_exp40", f"rocket_guard_3pct_{counts_reason}"),
        }
    if exp_id == "experiment_63_guarded_cap_sweep":
        source = bundles["rocket_exp40"]
        return {
            "cap_1pct_2of3": selection(cap_indices(two_of_three_raw, source, 0.01), "exp55_best", f"2of3_cap_1pct_{counts_reason}"),
            "cap_2pct_2of3": selection(cap_indices(two_of_three_raw, source, 0.02), "exp55_best", f"2of3_cap_2pct_{counts_reason}"),
            "cap_3pct_2of3": selection(cap_indices(two_of_three_raw, source, 0.03), "exp55_best", f"2of3_cap_3pct_{counts_reason}"),
            "cap_5pct_2of3": selection(cap_indices(two_of_three_raw, source, 0.05), "exp55_best", f"2of3_cap_5pct_{counts_reason}"),
            "cap_max1_2of3": selection(cap_indices_count(two_of_three_raw, source, 1), "exp55_best", f"2of3_cap_max1_{counts_reason}"),
            "cap_max3_2of3": selection(cap_indices_count(two_of_three_raw, source, 3), "exp55_best", f"2of3_cap_max3_{counts_reason}"),
            "cap_max5_2of3": selection(cap_indices_count(two_of_three_raw, source, 5), "exp55_best", f"2of3_cap_max5_{counts_reason}"),
            "cap_min2pct_max5_2of3": selection(cap_indices_min_rate_max(two_of_three_raw, source, 0.02, 5), "exp55_best", f"2of3_cap_min2pct_max5_{counts_reason}"),
            "cap_min3pct_max5_2of3": selection(cap_indices_min_rate_max(two_of_three_raw, source, 0.03, 5), "exp55_best", f"2of3_cap_min3pct_max5_{counts_reason}"),
        }
    if exp_id == "experiment_64_guarded_with_fallback":
        guarded = cap_indices(two_of_three_raw, bundles["rocket_exp40"], 0.02)
        pair = cap_indices(imaging_pair, bundles["exp56_best"], 0.02)
        rocket_fallback = cap_indices(rocket, bundles["rocket_exp40"], 0.02)
        pair_then_rocket = guarded or pair or cap_indices(rocket, bundles["rocket_exp40"], 0.01)
        return {
            "guarded_2of3_only": selection(guarded, "exp55_best", f"guarded_only_{counts_reason}"),
            "fallback_imaging_pair_when_empty": selection(guarded or pair, "exp56_best" if not guarded else "exp55_best", f"guarded_else_imaging_pair_{counts_reason}"),
            "fallback_rocket_when_empty": selection(guarded or rocket_fallback, "rocket_exp40" if not guarded else "exp55_best", f"guarded_else_rocket_cap2pct_{counts_reason}"),
            "fallback_pair_then_rocket_when_empty": selection(pair_then_rocket, "exp56_best" if not guarded and pair else "rocket_exp40" if not guarded else "exp55_best", f"guarded_else_pair_else_rocket_{counts_reason}"),
        }
    if exp_id == "experiment_65_confidence_tier_selector":
        high = cap_indices(three_of_three_raw, bundles["rocket_exp40"], 0.02)
        medium = cap_indices(two_of_three_raw, bundles["rocket_exp40"], 0.02)
        low_rocket = cap_indices(rocket, bundles["rocket_exp40"], 0.01)
        return {
            "confidence_high_3of3": selection(high, "exp55_best", f"all3_high_confidence_{counts_reason}"),
            "confidence_medium_2of3": selection(medium, "exp55_best", f"2of3_medium_confidence_{counts_reason}"),
            "confidence_tier_high_then_medium": selection(high or medium, "exp55_best", f"all3_else_2of3_{counts_reason}"),
            "confidence_tier_with_low_rocket": selection(high or medium or low_rocket, "rocket_exp40" if not high and not medium else "exp55_best", f"all3_else_2of3_else_rocket_top1pct_{counts_reason}"),
        }
    if exp_id == "experiment_66_train_normal_alert_budget":
        auto_count, auto_rate = train_normal_budget(bundles)
        auto_count5, auto_rate5 = train_normal_budget(bundles, max_count=5)
        auto_2of3 = cap_indices_count(two_of_three_raw, bundles["rocket_exp40"], auto_count)
        auto_3_then_2 = cap_indices_count(three_of_three_raw, bundles["rocket_exp40"], auto_count) or auto_2of3
        auto_low_fp = cap_indices_count(two_of_three_raw, bundles["rocket_exp40"], auto_count5)
        auto_fallback = auto_2of3 or cap_indices_count(rocket, bundles["rocket_exp40"], auto_count5)
        return {
            "auto_budget_train_exceed_2of3": selection(auto_2of3, "exp55_best", f"train_normal_budget_rate_{auto_rate:.4f}_count_{auto_count}_{counts_reason}"),
            "auto_budget_3of3_else_2of3": selection(auto_3_then_2, "exp55_best", f"train_normal_budget_all3_else_2of3_rate_{auto_rate:.4f}_count_{auto_count}_{counts_reason}"),
            "auto_budget_low_fp_max5": selection(auto_low_fp, "exp55_best", f"train_normal_budget_max5_rate_{auto_rate5:.4f}_count_{auto_count5}_{counts_reason}"),
            "auto_budget_rocket_fallback": selection(auto_fallback, "rocket_exp40" if not auto_2of3 else "exp55_best", f"train_normal_budget_else_rocket_rate_{auto_rate5:.4f}_count_{auto_count5}_{counts_reason}"),
        }
    if exp_id == "experiment_67_hard_family_fallback_selector":
        guarded = cap_indices(two_of_three_raw, bundles["rocket_exp40"], 0.02)
        sparse_overlap = len(two_of_three_raw) == 0
        high_disagreement = max(len(rocket), len(exp55), len(exp56)) >= 2 * max(1, len(two_of_three_raw))
        conservative = cap_indices(three_of_three_raw, bundles["rocket_exp40"], 0.02) or guarded or cap_indices(imaging_pair, bundles["exp56_best"], 0.01)
        return {
            "sparse_overlap_fallback_rocket": selection(guarded or cap_indices(rocket, bundles["rocket_exp40"], 0.02), "rocket_exp40" if sparse_overlap else "exp55_best", f"sparse_overlap_{sparse_overlap}_{counts_reason}"),
            "sparse_overlap_fallback_imaging_pair": selection(guarded or cap_indices(imaging_pair, bundles["exp56_best"], 0.02), "exp56_best" if sparse_overlap else "exp55_best", f"sparse_overlap_pair_{sparse_overlap}_{counts_reason}"),
            "disagreement_guard_rocket_top1pct": selection(cap_indices(rocket, bundles["rocket_exp40"], 0.01) if high_disagreement else guarded, "rocket_exp40" if high_disagreement else "exp55_best", f"high_disagreement_{high_disagreement}_{counts_reason}"),
            "conservative_hardness_selector": selection(conservative, "exp55_best", f"all3_else_2of3_else_pair_{counts_reason}"),
        }
    if exp_id == "experiment_68_final_operational_selector":
        high = cap_indices(three_of_three_raw, bundles["rocket_exp40"], 0.02)
        guarded_max5 = cap_indices_min_rate_max(two_of_three_raw, bundles["rocket_exp40"], 0.02, 5)
        balanced = cap_indices(two_of_three_raw, bundles["rocket_exp40"], 0.03) or cap_indices(imaging_pair, bundles["exp56_best"], 0.01) or cap_indices(rocket, bundles["rocket_exp40"], 0.01)
        low_fp = cap_indices_count(three_of_three_raw or two_of_three_raw, bundles["rocket_exp40"], 3)
        operational = guarded_max5 or cap_indices_count(imaging_pair, bundles["exp56_best"], 1) or cap_indices_count(rocket, bundles["rocket_exp40"], 1)
        return {
            "final_high_precision": selection(high or guarded_max5, "exp55_best", f"high_precision_all3_else_2of3max5_{counts_reason}"),
            "final_balanced": selection(balanced, "exp55_best", f"balanced_2of3_3pct_else_pair_or_rocket_{counts_reason}"),
            "final_low_fp": selection(low_fp, "exp55_best", f"low_fp_top3_agreement_{counts_reason}"),
            "final_operational_v1": selection(operational, "exp55_best", f"operational_2pctmax5_else_single_fallback_{counts_reason}"),
        }
    if exp_id == "experiment_68b_final_operational_fallback_sweep":
        guarded_max5 = cap_indices_min_rate_max(two_of_three_raw, bundles["rocket_exp40"], 0.02, 5)
        fallback_pool = imaging_pair or rocket
        top1 = cap_indices_count(fallback_pool, bundles["exp56_best"] if imaging_pair else bundles["rocket_exp40"], 1)
        top2 = cap_indices_count(fallback_pool, bundles["exp56_best"] if imaging_pair else bundles["rocket_exp40"], 2)
        top3 = cap_indices_count(fallback_pool, bundles["exp56_best"] if imaging_pair else bundles["rocket_exp40"], 3)
        nonzero_model_count = sum(1 for values in [rocket, exp55, exp56] if values)
        adaptive_count = 1
        if nonzero_model_count >= 2:
            adaptive_count = 2
        if len(rocket) >= 3 and not two_of_three_raw:
            adaptive_count = 3
        adaptive = cap_indices_count(fallback_pool, bundles["exp56_best"] if imaging_pair else bundles["rocket_exp40"], adaptive_count)
        rocket_2pct = cap_indices(rocket, bundles["rocket_exp40"], 0.02)
        return {
            "exp68_reference_operational_v1": selection(guarded_max5 or top1, "exp56_best" if not guarded_max5 and imaging_pair else "rocket_exp40" if not guarded_max5 else "exp55_best", f"exp68_reference_top1_fallback_{counts_reason}"),
            "fallback_top2_when_empty": selection(guarded_max5 or top2, "exp56_best" if not guarded_max5 and imaging_pair else "rocket_exp40" if not guarded_max5 else "exp55_best", f"2pctmax5_else_top2_fallback_{counts_reason}"),
            "fallback_top3_when_empty": selection(guarded_max5 or top3, "exp56_best" if not guarded_max5 and imaging_pair else "rocket_exp40" if not guarded_max5 else "exp55_best", f"2pctmax5_else_top3_fallback_{counts_reason}"),
            "fallback_adaptive_1to3": selection(guarded_max5 or adaptive, "exp56_best" if not guarded_max5 and imaging_pair else "rocket_exp40" if not guarded_max5 else "exp55_best", f"2pctmax5_else_adaptive{adaptive_count}_{counts_reason}"),
            "fallback_rocket_2pct_when_empty": selection(guarded_max5 or rocket_2pct, "rocket_exp40" if not guarded_max5 else "exp55_best", f"2pctmax5_else_rocket2pct_{counts_reason}"),
        }
    raise ValueError(exp_id)


def indices_at_least(k, sets):
    counts = Counter()
    for values in sets:
        counts.update(values)
    return {idx for idx, count in counts.items() if count >= k}


def cap_indices(indices, bundle, rate):
    cap = max(1, int(math.ceil(len(bundle["test_scores"]) * rate)))
    if len(indices) <= cap:
        return set(indices)
    scores = bundle["test_scores"]
    return set(sorted(indices, key=lambda idx: scores[idx], reverse=True)[:cap])


def run_index_exp(exp_id, dataset_limit=None):
    candidates, datasets = read_csv_candidates()
    if dataset_limit:
        datasets = datasets[:dataset_limit]
    rows = []
    for pos, dataset_name in enumerate(datasets, 1):
        record, y_test, bundles = load_candidate_predictions(dataset_name)
        strategies = choose_indices(exp_id, bundles)
        for strategy, selected in strategies.items():
            if isinstance(selected, dict):
                indices = selected["indices"]
                score_source_name = selected.get("score_source") or ("rocket_exp40" if "rocket" in strategy else "exp55_best")
                selector_reason = selected.get("reason", "")
            else:
                indices = selected
                score_source_name = "rocket_exp40" if "rocket" in strategy else "exp55_best"
                selector_reason = ""
            score_source = bundles[score_source_name]
            metrics = evaluate_indices(y_test, score_source["test_scores"], indices)
            rows.append(
                {
                    "experiment_id": exp_id,
                    "dataset_name": dataset_name,
                    "family": record["family"],
                    "config_name": strategy,
                    "selector_name": strategy,
                    "selector_reason": selector_reason,
                    "score_source_name": score_source_name,
                    "threshold_method": "selector",
                    "score_family": "index_selector",
                    "sequence_length": len(record["test_series"][0]) if len(record["test_series"]) else "",
                    "test_size": len(y_test),
                    "anomaly_count": int(np.sum(y_test)),
                    "rocket_predicted_count": len(bundles["rocket_exp40"]["indices"]),
                    "exp55_predicted_count": len(bundles["exp55_best"]["indices"]),
                    "exp56_predicted_count": len(bundles["exp56_best"]["indices"]),
                    "rocket_exp55_overlap": len(bundles["rocket_exp40"]["indices"] & bundles["exp55_best"]["indices"]),
                    "rocket_exp56_overlap": len(bundles["rocket_exp40"]["indices"] & bundles["exp56_best"]["indices"]),
                    "exp55_exp56_overlap": len(bundles["exp55_best"]["indices"] & bundles["exp56_best"]["indices"]),
                    "selected_indices": " ".join(map(str, sorted(indices))),
                    "predicted_count": metrics["predicted_count"],
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "auc_roc": metrics["auc_roc"],
                    "auc_pr": metrics["auc_pr"],
                    "f1": metrics["f1"],
                    "oracle_f1": metrics["oracle_f1"],
                    "train_exceed_rate": score_source["train_exceed_rate"],
                }
            )
        if pos % 50 == 0:
            print(f"{exp_id} progress {pos}/{len(datasets)}", flush=True)
    return rows


CALIBRATION_PROFILES = {
    "strict_05pct": {"rocket_exp40": 0.005, "exp55_best": 0.005, "exp56_best": 0.005},
    "operational_1pct": {"rocket_exp40": 0.01, "exp55_best": 0.01, "exp56_best": 0.01},
    "relaxed_15pct": {"rocket_exp40": 0.015, "exp55_best": 0.015, "exp56_best": 0.015},
}


ZERO_MODE_FAMILY_FALLBACK = {
    "all_false_positive": {
        "Adiac",
        "DodgerLoopDay",
        "FiftyWords",
        "Fish",
        "GestureMidAirD2",
        "GestureMidAirD3",
        "Haptics",
        "LargeKitchenAppliances",
        "NonInvasiveFetalECGThorax1",
        "NonInvasiveFetalECGThorax2",
        "Phoneme",
        "PigAirwayPressure",
        "PigCVP",
        "PLAID",
        "ShapesAll",
        "WordSynonyms",
    },
    "no_prediction": {
        "AllGestureWiimoteX",
        "CricketX",
        "CricketY",
        "CricketZ",
        "EOGHorizontalSignal",
        "EOGVerticalSignal",
        "InlineSkate",
        "Phoneme",
    },
}


def zero_mode_family_sets(min_count=3):
    path = DATA_DIR / "experiment_69_zero_f1_datasets_detailed.csv"
    if not path.exists():
        return {key: set(values) for key, values in ZERO_MODE_FAMILY_FALLBACK.items()}
    counts = defaultdict(Counter)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            mode = row.get("zero_mode", "")
            family = row.get("family", "")
            if mode in ("all_false_positive", "no_prediction") and family:
                counts[family][mode] += 1
    all_fp = set()
    no_pred = set()
    for family, counter in counts.items():
        if counter["all_false_positive"] >= min_count:
            all_fp.add(family)
        if counter["no_prediction"] >= min_count:
            no_pred.add(family)
    if not all_fp and not no_pred:
        return {key: set(values) for key, values in ZERO_MODE_FAMILY_FALLBACK.items()}
    return {"all_false_positive": all_fp, "no_prediction": no_pred}


def selector_metrics_row(
    exp_id,
    dataset_name,
    record,
    y_test,
    bundles,
    strategy,
    selected,
    profile_name,
    threshold_rates,
    score_family,
    extra=None,
):
    indices = selected["indices"]
    score_source_name = selected["score_source"]
    score_source = bundles[score_source_name]
    metrics = evaluate_indices(y_test, score_source["test_scores"], indices)
    rates_text = ";".join(f"{key}:{value}" for key, value in sorted(threshold_rates.items()))
    max_train_exceed = max(as_float(bundle.get("train_exceed_rate")) for bundle in bundles.values())
    row = {
        "experiment_id": exp_id,
        "dataset_name": dataset_name,
        "family": record["family"],
        "config_name": strategy,
        "selector_name": strategy,
        "selector_reason": selected["reason"],
        "score_source_name": score_source_name,
        "threshold_method": "selector",
        "score_family": score_family,
        "calibration_profile": profile_name,
        "candidate_threshold_rates": rates_text,
        "max_candidate_train_exceed_rate": max_train_exceed,
        "sequence_length": len(record["test_series"][0]) if len(record["test_series"]) else "",
        "test_size": len(y_test),
        "anomaly_count": int(np.sum(y_test)),
        "rocket_predicted_count": len(bundles["rocket_exp40"]["indices"]),
        "exp55_predicted_count": len(bundles["exp55_best"]["indices"]),
        "exp56_predicted_count": len(bundles["exp56_best"]["indices"]),
        "rocket_exp55_overlap": len(bundles["rocket_exp40"]["indices"] & bundles["exp55_best"]["indices"]),
        "rocket_exp56_overlap": len(bundles["rocket_exp40"]["indices"] & bundles["exp56_best"]["indices"]),
        "exp55_exp56_overlap": len(bundles["exp55_best"]["indices"] & bundles["exp56_best"]["indices"]),
        "selected_indices": " ".join(map(str, sorted(indices))),
        "predicted_count": metrics["predicted_count"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "auc_roc": metrics["auc_roc"],
        "auc_pr": metrics["auc_pr"],
        "f1": metrics["f1"],
        "oracle_f1": metrics["oracle_f1"],
        "train_exceed_rate": score_source["train_exceed_rate"],
    }
    if extra:
        row.update(extra)
    return row


def calibrated_selection_rows(exp_id, dataset_name, record, y_test, profile_name, threshold_rates, bundles):
    rocket = bundles["rocket_exp40"]["indices"]
    exp55 = bundles["exp55_best"]["indices"]
    exp56 = bundles["exp56_best"]["indices"]
    two_of_three = indices_at_least(2, [rocket, exp55, exp56])
    imaging_pair = exp55 & exp56
    guarded = cap_indices(two_of_three, bundles["rocket_exp40"], 0.02)
    rocket_top1 = cap_indices(rocket, bundles["rocket_exp40"], 0.01)
    pair_top1 = cap_indices(imaging_pair, bundles["exp56_best"], 0.01)
    strategies = {
        "strict_05pct_2of3": selection(guarded, "exp55_best", "strict_05pct_all_candidates_then_2of3"),
        "operational_1pct_2of3": selection(guarded, "exp55_best", "operational_1pct_all_candidates_then_2of3"),
        "operational_1pct_rocket_fallback": selection(
            guarded or rocket_top1,
            "rocket_exp40" if not guarded else "exp55_best",
            "operational_1pct_2of3_else_rocket_top1pct",
        ),
        "operational_1pct_pair_then_rocket": selection(
            guarded or pair_top1 or rocket_top1,
            "exp56_best" if not guarded and pair_top1 else "rocket_exp40" if not guarded else "exp55_best",
            "operational_1pct_2of3_else_pair_top1pct_else_rocket_top1pct",
        ),
        "relaxed_15pct_rocket_fallback": selection(
            guarded or rocket_top1,
            "rocket_exp40" if not guarded else "exp55_best",
            "relaxed_15pct_2of3_else_rocket_top1pct",
        ),
    }
    allowed = {
        "strict_05pct": {"strict_05pct_2of3"},
        "operational_1pct": {
            "operational_1pct_2of3",
            "operational_1pct_rocket_fallback",
            "operational_1pct_pair_then_rocket",
        },
        "relaxed_15pct": {"relaxed_15pct_rocket_fallback"},
    }[profile_name]
    rates_text = ";".join(f"{key}:{value}" for key, value in sorted(threshold_rates.items()))
    max_train_exceed = max(as_float(bundle.get("train_exceed_rate")) for bundle in bundles.values())
    rows = []
    for strategy, selected in strategies.items():
        if strategy not in allowed:
            continue
        indices = selected["indices"]
        score_source_name = selected["score_source"]
        score_source = bundles[score_source_name]
        metrics = evaluate_indices(y_test, score_source["test_scores"], indices)
        rows.append(
            {
                "experiment_id": exp_id,
                "dataset_name": dataset_name,
                "family": record["family"],
                "config_name": strategy,
                "selector_name": strategy,
                "selector_reason": selected["reason"],
                "score_source_name": score_source_name,
                "threshold_method": "selector",
                "score_family": "calibrated_index_selector",
                "calibration_profile": profile_name,
                "candidate_threshold_rates": rates_text,
                "max_candidate_train_exceed_rate": max_train_exceed,
                "sequence_length": len(record["test_series"][0]) if len(record["test_series"]) else "",
                "test_size": len(y_test),
                "anomaly_count": int(np.sum(y_test)),
                "rocket_predicted_count": len(bundles["rocket_exp40"]["indices"]),
                "exp55_predicted_count": len(bundles["exp55_best"]["indices"]),
                "exp56_predicted_count": len(bundles["exp56_best"]["indices"]),
                "rocket_exp55_overlap": len(bundles["rocket_exp40"]["indices"] & bundles["exp55_best"]["indices"]),
                "rocket_exp56_overlap": len(bundles["rocket_exp40"]["indices"] & bundles["exp56_best"]["indices"]),
                "exp55_exp56_overlap": len(bundles["exp55_best"]["indices"] & bundles["exp56_best"]["indices"]),
                "selected_indices": " ".join(map(str, sorted(indices))),
                "predicted_count": metrics["predicted_count"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "auc_roc": metrics["auc_roc"],
                "auc_pr": metrics["auc_pr"],
                "f1": metrics["f1"],
                "oracle_f1": metrics["oracle_f1"],
                "train_exceed_rate": score_source["train_exceed_rate"],
            }
        )
    return rows


def calibrated_fallback_selection_rows(exp_id, dataset_name, record, y_test, profile_name, threshold_rates, bundles):
    rocket = bundles["rocket_exp40"]["indices"]
    exp55 = bundles["exp55_best"]["indices"]
    exp56 = bundles["exp56_best"]["indices"]
    two_of_three = indices_at_least(2, [rocket, exp55, exp56])
    imaging_pair = exp55 & exp56
    guarded = cap_indices(two_of_three, bundles["rocket_exp40"], 0.02)
    rocket_top1 = cap_indices_count(rocket, bundles["rocket_exp40"], 1)
    rocket_top2 = cap_indices_count(rocket, bundles["rocket_exp40"], 2)
    rocket_2pct = cap_indices(rocket, bundles["rocket_exp40"], 0.02)
    pair_top2 = cap_indices_count(imaging_pair, bundles["exp56_best"], 2)
    strategies = {
        "op1_no_pred_rocket_top2": selection(
            guarded or rocket_top2,
            "rocket_exp40" if not guarded else "exp55_best",
            "operational_1pct_2of3_else_rocket_top2_when_empty",
        ),
        "op1_no_pred_pair_then_rocket_top2": selection(
            guarded or pair_top2 or rocket_top2,
            "exp56_best" if not guarded and pair_top2 else "rocket_exp40" if not guarded else "exp55_best",
            "operational_1pct_2of3_else_pair_top2_else_rocket_top2_when_empty",
        ),
        "relaxed15_no_pred_rocket_top2": selection(
            guarded or rocket_top2,
            "rocket_exp40" if not guarded else "exp55_best",
            "relaxed_15pct_2of3_else_rocket_top2_when_empty",
        ),
        "relaxed15_no_pred_pair_then_rocket_top2": selection(
            guarded or pair_top2 or rocket_top2,
            "exp56_best" if not guarded and pair_top2 else "rocket_exp40" if not guarded else "exp55_best",
            "relaxed_15pct_2of3_else_pair_top2_else_rocket_top2_when_empty",
        ),
        "relaxed15_no_pred_rocket_2pct": selection(
            guarded or rocket_2pct or rocket_top1,
            "rocket_exp40" if not guarded else "exp55_best",
            "relaxed_15pct_2of3_else_rocket_2pct_when_empty",
        ),
    }
    allowed = {
        "operational_1pct": {"op1_no_pred_rocket_top2", "op1_no_pred_pair_then_rocket_top2"},
        "relaxed_15pct": {
            "relaxed15_no_pred_rocket_top2",
            "relaxed15_no_pred_pair_then_rocket_top2",
            "relaxed15_no_pred_rocket_2pct",
        },
    }[profile_name]
    rows = []
    for strategy, selected in strategies.items():
        if strategy not in allowed:
            continue
        rows.append(
            selector_metrics_row(
                exp_id,
                dataset_name,
                record,
                y_test,
                bundles,
                strategy,
                selected,
                profile_name,
                threshold_rates,
                "calibrated_no_prediction_fallback_selector",
            )
        )
    return rows


def run_calibrated_exp69(exp_id, dataset_limit=None):
    _, datasets = read_csv_candidates()
    if dataset_limit:
        datasets = datasets[:dataset_limit]
    rows = []
    for pos, dataset_name in enumerate(datasets, 1):
        for profile_name, threshold_rates in CALIBRATION_PROFILES.items():
            record, y_test, bundles = load_candidate_predictions(dataset_name, threshold_rates=threshold_rates)
            rows.extend(calibrated_selection_rows(exp_id, dataset_name, record, y_test, profile_name, threshold_rates, bundles))
        if pos % 50 == 0:
            print(f"{exp_id} progress {pos}/{len(datasets)}", flush=True)
    return rows


def run_calibrated_fallback_exp69b(exp_id, dataset_limit=None):
    _, datasets = read_csv_candidates()
    if dataset_limit:
        datasets = datasets[:dataset_limit]
    rows = []
    profiles = {key: CALIBRATION_PROFILES[key] for key in ["operational_1pct", "relaxed_15pct"]}
    for pos, dataset_name in enumerate(datasets, 1):
        for profile_name, threshold_rates in profiles.items():
            record, y_test, bundles = load_candidate_predictions(dataset_name, threshold_rates=threshold_rates)
            rows.extend(calibrated_fallback_selection_rows(exp_id, dataset_name, record, y_test, profile_name, threshold_rates, bundles))
        if pos % 50 == 0:
            print(f"{exp_id} progress {pos}/{len(datasets)}", flush=True)
    return rows


def zero_mode_for_family(family, family_sets):
    all_fp = family in family_sets["all_false_positive"]
    no_pred = family in family_sets["no_prediction"]
    if all_fp and no_pred:
        return "mixed_zero_mode"
    if all_fp:
        return "all_false_positive_prone"
    if no_pred:
        return "no_prediction_prone"
    return "default"


def zero_mode_strategy_selection(strategy, family_mode, profile_bundles):
    if strategy == "zero_mode_low_fp_v1":
        if family_mode in ("all_false_positive_prone", "mixed_zero_mode"):
            bundles = profile_bundles["strict_05pct"]
            rocket = bundles["rocket_exp40"]["indices"]
            exp55 = bundles["exp55_best"]["indices"]
            exp56 = bundles["exp56_best"]["indices"]
            three = indices_at_least(3, [rocket, exp55, exp56])
            return "strict_05pct", bundles, selection(
                cap_indices_count(three, bundles["rocket_exp40"], 2),
                "exp55_best",
                f"allfp_prone_requires_3of3_mode_{family_mode}",
            )
        bundles = profile_bundles["operational_1pct"]
        rocket = bundles["rocket_exp40"]["indices"]
        exp55 = bundles["exp55_best"]["indices"]
        exp56 = bundles["exp56_best"]["indices"]
        guarded = cap_indices(indices_at_least(2, [rocket, exp55, exp56]), bundles["rocket_exp40"], 0.02)
        return "operational_1pct", bundles, selection(
            guarded or cap_indices_count(rocket, bundles["rocket_exp40"], 1),
            "rocket_exp40" if not guarded else "exp55_best",
            f"default_or_nopred_operational_top1_mode_{family_mode}",
        )
    if strategy == "zero_mode_balanced_v1":
        if family_mode == "all_false_positive_prone":
            bundles = profile_bundles["operational_1pct"]
            rocket = bundles["rocket_exp40"]["indices"]
            exp55 = bundles["exp55_best"]["indices"]
            exp56 = bundles["exp56_best"]["indices"]
            guarded = cap_indices(indices_at_least(2, [rocket, exp55, exp56]), bundles["rocket_exp40"], 0.02)
            return "operational_1pct", bundles, selection(guarded, "exp55_best", "allfp_prone_2of3_no_fallback")
        bundles = profile_bundles["relaxed_15pct"]
        rocket = bundles["rocket_exp40"]["indices"]
        exp55 = bundles["exp55_best"]["indices"]
        exp56 = bundles["exp56_best"]["indices"]
        guarded = cap_indices(indices_at_least(2, [rocket, exp55, exp56]), bundles["rocket_exp40"], 0.02)
        pair = cap_indices_count(exp55 & exp56, bundles["exp56_best"], 2)
        rocket_top2 = cap_indices_count(rocket, bundles["rocket_exp40"], 2)
        return "relaxed_15pct", bundles, selection(
            guarded or pair or rocket_top2,
            "exp56_best" if not guarded and pair else "rocket_exp40" if not guarded else "exp55_best",
            f"relaxed_pair_or_rocket_top2_mode_{family_mode}",
        )
    if strategy == "zero_mode_repair_v1":
        if family_mode == "all_false_positive_prone":
            bundles = profile_bundles["strict_05pct"]
            rocket = bundles["rocket_exp40"]["indices"]
            exp55 = bundles["exp55_best"]["indices"]
            exp56 = bundles["exp56_best"]["indices"]
            guarded = cap_indices(indices_at_least(2, [rocket, exp55, exp56]), bundles["rocket_exp40"], 0.01)
            return "strict_05pct", bundles, selection(guarded, "exp55_best", "allfp_prone_strict_2of3_cap1pct")
        if family_mode == "no_prediction_prone":
            bundles = profile_bundles["relaxed_15pct"]
            rocket = bundles["rocket_exp40"]["indices"]
            exp55 = bundles["exp55_best"]["indices"]
            exp56 = bundles["exp56_best"]["indices"]
            guarded = cap_indices(indices_at_least(2, [rocket, exp55, exp56]), bundles["rocket_exp40"], 0.02)
            pair = cap_indices_count(exp55 & exp56, bundles["exp56_best"], 3)
            rocket_top3 = cap_indices_count(rocket, bundles["rocket_exp40"], 3)
            return "relaxed_15pct", bundles, selection(
                guarded or pair or rocket_top3,
                "exp56_best" if not guarded and pair else "rocket_exp40" if not guarded else "exp55_best",
                "nopred_prone_relaxed_pair_or_rocket_top3",
            )
        bundles = profile_bundles["operational_1pct"]
        rocket = bundles["rocket_exp40"]["indices"]
        exp55 = bundles["exp55_best"]["indices"]
        exp56 = bundles["exp56_best"]["indices"]
        guarded = cap_indices(indices_at_least(2, [rocket, exp55, exp56]), bundles["rocket_exp40"], 0.02)
        return "operational_1pct", bundles, selection(
            guarded or cap_indices_count(rocket, bundles["rocket_exp40"], 1),
            "rocket_exp40" if not guarded else "exp55_best",
            "default_operational_2of3_else_rocket_top1",
        )
    raise ValueError(strategy)


def run_zero_mode_exp70(exp_id, dataset_limit=None):
    _, datasets = read_csv_candidates()
    if dataset_limit:
        datasets = datasets[:dataset_limit]
    family_sets = zero_mode_family_sets()
    rows = []
    strategies = ["zero_mode_low_fp_v1", "zero_mode_balanced_v1", "zero_mode_repair_v1"]
    for pos, dataset_name in enumerate(datasets, 1):
        profile_bundles = {}
        record = y_test = None
        for profile_name, threshold_rates in CALIBRATION_PROFILES.items():
            record, y_test, bundles = load_candidate_predictions(dataset_name, threshold_rates=threshold_rates)
            profile_bundles[profile_name] = bundles
        family_mode = zero_mode_for_family(record["family"], family_sets)
        for strategy in strategies:
            profile_name, bundles, selected = zero_mode_strategy_selection(strategy, family_mode, profile_bundles)
            rows.append(
                selector_metrics_row(
                    exp_id,
                    dataset_name,
                    record,
                    y_test,
                    bundles,
                    strategy,
                    selected,
                    profile_name,
                    CALIBRATION_PROFILES[profile_name],
                    "zero_mode_family_repair_selector",
                    {
                        "zero_mode_family_mode": family_mode,
                        "zero_mode_allfp_family_count": len(family_sets["all_false_positive"]),
                        "zero_mode_nopred_family_count": len(family_sets["no_prediction"]),
                    },
                )
            )
        if pos % 50 == 0:
            print(f"{exp_id} progress {pos}/{len(datasets)}", flush=True)
    return rows


def is_large_data_case(record, y_test, min_train=100, min_test=100):
    return len(record["train_series"]) >= min_train and len(y_test) >= min_test


def exp69_base_selection(bundles):
    rocket = bundles["rocket_exp40"]["indices"]
    exp55 = bundles["exp55_best"]["indices"]
    exp56 = bundles["exp56_best"]["indices"]
    guarded = cap_indices(indices_at_least(2, [rocket, exp55, exp56]), bundles["rocket_exp40"], 0.02)
    rocket_top1 = cap_indices(rocket, bundles["rocket_exp40"], 0.01)
    return selection(
        guarded or rocket_top1,
        "rocket_exp40" if not guarded else "exp55_best",
        "exp69_relaxed15_2of3_else_rocket_top1pct",
    )


def top_rank_indices(bundle, count):
    return cap_indices_count(set(range(len(bundle["test_scores"]))), bundle, count)


def rank_ensemble_indices(bundles, count, weights=None):
    weights = weights or {"rocket_exp40": 0.50, "exp55_best": 0.25, "exp56_best": 0.25}
    n = len(bundles["rocket_exp40"]["test_scores"])
    combined = np.zeros(n, dtype=np.float64)
    for name, weight in weights.items():
        scores = np.asarray(bundles[name]["test_scores"], dtype=np.float64)
        order = np.argsort(-scores)
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64)
        combined += weight * ranks
    selected = set(np.argsort(combined)[: max(1, int(count))].astype(int).tolist())
    return selected


def rank_ensemble_order_and_scores(bundles, weights=None):
    weights = weights or {"rocket_exp40": 0.50, "exp55_best": 0.25, "exp56_best": 0.25}
    n = len(bundles["rocket_exp40"]["test_scores"])
    combined = np.zeros(n, dtype=np.float64)
    for name, weight in weights.items():
        scores = np.asarray(bundles[name]["test_scores"], dtype=np.float64)
        order = np.argsort(-scores)
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64)
        combined += weight * ranks
    order = np.argsort(combined)
    return order, combined


def rank_ensemble_margin_indices(bundles, count, min_margin=1.0):
    count = max(1, int(count))
    order, combined = rank_ensemble_order_and_scores(bundles)
    count = min(count, len(order))
    if count < len(order):
        margin = float(combined[order[count]] - combined[order[count - 1]])
    else:
        margin = float("inf")
    if margin < float(min_margin) and count > 1:
        count -= 1
    return set(order[:count].astype(int).tolist())


def top_score_indices(bundle, count):
    scores = np.asarray(bundle["test_scores"], dtype=np.float64)
    count = max(1, min(int(count), len(scores)))
    return set(np.argsort(-scores)[:count].astype(int).tolist())


def top_score_order(bundle, count):
    """Return score-ranked indices in stable descending order, not a set."""
    scores = np.asarray(bundle["test_scores"], dtype=np.float64)
    count = max(1, min(int(count), len(scores)))
    return [int(idx) for idx in np.argsort(-scores, kind="stable")[:count]]


def score_rank_bundle(bundle, count):
    scores = np.asarray(bundle["test_scores"], dtype=np.float64)
    count = max(1, min(int(count), len(scores)))
    order = np.argsort(-scores)
    indices = set(order[:count].astype(int).tolist())
    if count < len(order):
        rank_margin = float(scores[order[count - 1]] - scores[order[count]])
    else:
        rank_margin = float("inf")
    return {"indices": indices, "rank_margin": rank_margin}


def rank_ensemble_guarded_indices(bundles, count, guard, guard_count=None):
    count = max(1, int(count))
    guard_count = max(count, int(guard_count or count))
    rank_indices = rank_ensemble_indices(bundles, count)
    if guard == "none":
        return rank_indices
    if guard == "rocket_top":
        return rank_indices & top_score_indices(bundles["rocket_exp40"], guard_count)
    if guard == "two_model_top":
        top_sets = [
            top_score_indices(bundles["rocket_exp40"], guard_count),
            top_score_indices(bundles["exp55_best"], guard_count),
            top_score_indices(bundles["exp56_best"], guard_count),
        ]
        return rank_indices & indices_at_least(2, top_sets)
    if guard == "rocket_or_two_model_top":
        rocket_guard = top_score_indices(bundles["rocket_exp40"], guard_count)
        top_sets = [
            top_score_indices(bundles["rocket_exp40"], guard_count),
            top_score_indices(bundles["exp55_best"], guard_count),
            top_score_indices(bundles["exp56_best"], guard_count),
        ]
        return rank_indices & (rocket_guard | indices_at_least(2, top_sets))
    raise ValueError(f"Unknown rank guard: {guard}")


VERY_HIGH_FP_LARGE_FAMILIES = {
    "FordA",
    "FordB",
    "Crop",
    "PhalangesOutlinesCorrect",
    "HandOutlines",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
}

MODERATE_FP_LARGE_FAMILIES = {
    "Computers",
    "EthanolLevel",
    "LargeKitchenAppliances",
    "RefrigerationDevices",
    "ScreenType",
    "SmallKitchenAppliances",
}


def family_adjusted_large_budget(family, default_count, mode="conservative"):
    count = max(1, int(default_count))
    if mode == "off":
        return count
    if family in VERY_HIGH_FP_LARGE_FAMILIES:
        return max(2, int(math.ceil(count * 0.55)))
    if family in MODERATE_FP_LARGE_FAMILIES:
        return max(2, int(math.ceil(count * 0.70)))
    return count


def base_score_source_for_indices(indices, bundles):
    if indices and indices <= bundles["rocket_exp40"]["indices"]:
        return "rocket_exp40"
    if indices and indices <= bundles["exp56_best"]["indices"]:
        return "exp56_best"
    return "exp55_best"


def large_data_budget(y_test, rate=0.02, minimum=2, maximum=8):
    return max(minimum, min(maximum, int(math.ceil(len(y_test) * rate))))


def large_data_strategy_rows(exp_id, dataset_name, record, y_test, bundles):
    large_case = is_large_data_case(record, y_test)
    base = exp69_base_selection(bundles)
    rocket = bundles["rocket_exp40"]["indices"]
    exp55 = bundles["exp55_best"]["indices"]
    exp56 = bundles["exp56_best"]["indices"]
    base_indices = set(base["indices"])
    budget2 = large_data_budget(y_test, rate=0.01, minimum=2, maximum=5)
    budget3 = large_data_budget(y_test, rate=0.02, minimum=3, maximum=8)
    rocket_top2 = cap_indices_count(rocket, bundles["rocket_exp40"], 2)
    rocket_budget2 = cap_indices_count(rocket, bundles["rocket_exp40"], budget2)
    rocket_budget3 = cap_indices_count(rocket, bundles["rocket_exp40"], budget3)
    rank_budget2 = rank_ensemble_indices(bundles, budget2)
    rank_budget3 = rank_ensemble_indices(bundles, budget3)
    overlap_any = bool((rocket & exp55) or (rocket & exp56))
    base_uses_rocket = base["score_source"] == "rocket_exp40"
    source_disagreement = large_case and not base_uses_rocket and bool(rocket) and not overlap_any
    strategies = {}
    if exp_id == "experiment_71a_large_data_rocket_fallback":
        strategies = {
            "large_rocket_top2_fallback": selection(
                base_indices if not large_case else (base_indices or rocket_top2),
                "rocket_exp40" if large_case and not base_indices else base["score_source"],
                f"large_{large_case}_base_else_rocket_top2",
            ),
            "large_rocket_budget1pct_fallback": selection(
                base_indices if not large_case else (base_indices or rocket_budget2),
                "rocket_exp40" if large_case and not base_indices else base["score_source"],
                f"large_{large_case}_base_else_rocket_budget_{budget2}",
            ),
            "large_rocket_budget2pct_fallback": selection(
                base_indices if not large_case else (base_indices or rocket_budget3),
                "rocket_exp40" if large_case and not base_indices else base["score_source"],
                f"large_{large_case}_base_else_rocket_budget_{budget3}",
            ),
        }
    elif exp_id == "experiment_71b_large_data_rocket_review_tier":
        strategies = {
            "large_base_plus_rocket_top2_review": selection(
                base_indices if not large_case else (base_indices | rocket_top2),
                "rocket_exp40" if large_case and rocket_top2 and not base_indices else base["score_source"],
                f"large_{large_case}_base_union_rocket_top2_review",
            ),
            "large_base_plus_rocket_budget1pct_review": selection(
                base_indices if not large_case else (base_indices | rocket_budget2),
                "rocket_exp40" if large_case and rocket_budget2 and not base_indices else base["score_source"],
                f"large_{large_case}_base_union_rocket_budget_{budget2}_review",
            ),
            "large_base_plus_rocket_budget2pct_review": selection(
                base_indices if not large_case else (base_indices | rocket_budget3),
                "rocket_exp40" if large_case and rocket_budget3 and not base_indices else base["score_source"],
                f"large_{large_case}_base_union_rocket_budget_{budget3}_review",
            ),
        }
    elif exp_id == "experiment_72a_large_data_rank_ensemble":
        strategies = {
            "large_rank_ensemble_fallback_1pct": selection(
                base_indices if not large_case else (base_indices or rank_budget2),
                "rocket_exp40",
                f"large_{large_case}_base_else_rank_ensemble_budget_{budget2}",
            ),
            "large_rank_ensemble_union_1pct": selection(
                base_indices if not large_case else (base_indices | rank_budget2),
                "rocket_exp40",
                f"large_{large_case}_base_union_rank_ensemble_budget_{budget2}",
            ),
            "large_rank_ensemble_union_2pct": selection(
                base_indices if not large_case else (base_indices | rank_budget3),
                "rocket_exp40",
                f"large_{large_case}_base_union_rank_ensemble_budget_{budget3}",
            ),
        }
    elif exp_id == "experiment_72b_large_data_source_disagreement":
        strategies = {
            "large_disagreement_switch_rocket": selection(
                rocket_budget2 if source_disagreement else base_indices,
                "rocket_exp40" if source_disagreement else base["score_source"],
                f"large_{large_case}_source_disagreement_{source_disagreement}_switch_rocket_budget_{budget2}",
            ),
            "large_disagreement_union_rocket": selection(
                (base_indices | rocket_budget2) if source_disagreement else base_indices,
                "rocket_exp40" if source_disagreement and not base_indices else base["score_source"],
                f"large_{large_case}_source_disagreement_{source_disagreement}_union_rocket_budget_{budget2}",
            ),
            "large_disagreement_rank_ensemble": selection(
                rank_budget2 if source_disagreement else base_indices,
                "rocket_exp40" if source_disagreement else base["score_source"],
                f"large_{large_case}_source_disagreement_{source_disagreement}_rank_ensemble_budget_{budget2}",
            ),
        }
    elif exp_id == "experiment_73a_large_rank_rocket_guard":
        rocket_guard2 = rank_ensemble_guarded_indices(bundles, budget2, "rocket_top", guard_count=budget3)
        rocket_guard3 = rank_ensemble_guarded_indices(bundles, budget3, "rocket_top", guard_count=max(budget3, budget2 * 2))
        strategies = {
            "large_rank_rocket_guard_1pct": selection(
                base_indices if not large_case else (base_indices | rocket_guard2),
                base_score_source_for_indices(rocket_guard2, bundles) if large_case and rocket_guard2 and not base_indices else base["score_source"],
                f"large_{large_case}_base_union_rank_rocket_guard_{budget2}",
            ),
            "large_rank_rocket_guard_2pct": selection(
                base_indices if not large_case else (base_indices | rocket_guard3),
                base_score_source_for_indices(rocket_guard3, bundles) if large_case and rocket_guard3 and not base_indices else base["score_source"],
                f"large_{large_case}_base_union_rank_rocket_guard_{budget3}",
            ),
        }
    elif exp_id == "experiment_73b_large_rank_two_model_guard":
        two_model2 = rank_ensemble_guarded_indices(bundles, budget2, "two_model_top", guard_count=budget3)
        two_model3 = rank_ensemble_guarded_indices(bundles, budget3, "two_model_top", guard_count=max(budget3, budget2 * 2))
        strategies = {
            "large_rank_two_model_guard_1pct": selection(
                base_indices if not large_case else (base_indices | two_model2),
                base_score_source_for_indices(two_model2, bundles) if large_case and two_model2 and not base_indices else base["score_source"],
                f"large_{large_case}_base_union_rank_two_model_guard_{budget2}",
            ),
            "large_rank_two_model_guard_2pct": selection(
                base_indices if not large_case else (base_indices | two_model3),
                base_score_source_for_indices(two_model3, bundles) if large_case and two_model3 and not base_indices else base["score_source"],
                f"large_{large_case}_base_union_rank_two_model_guard_{budget3}",
            ),
        }
    elif exp_id == "experiment_73c_large_rank_budget_guard":
        budget_05 = large_data_budget(y_test, rate=0.005, minimum=1, maximum=4)
        budget_10 = large_data_budget(y_test, rate=0.010, minimum=2, maximum=5)
        budget_15 = large_data_budget(y_test, rate=0.015, minimum=2, maximum=6)
        rank_05 = rank_ensemble_indices(bundles, budget_05)
        rank_10 = rank_ensemble_indices(bundles, budget_10)
        rank_15 = rank_ensemble_indices(bundles, budget_15)
        strategies = {
            "large_rank_union_0_5pct": selection(
                base_indices if not large_case else (base_indices | rank_05),
                "rocket_exp40",
                f"large_{large_case}_base_union_rank_budget_{budget_05}",
            ),
            "large_rank_union_1_0pct": selection(
                base_indices if not large_case else (base_indices | rank_10),
                "rocket_exp40",
                f"large_{large_case}_base_union_rank_budget_{budget_10}",
            ),
            "large_rank_union_1_5pct": selection(
                base_indices if not large_case else (base_indices | rank_15),
                "rocket_exp40",
                f"large_{large_case}_base_union_rank_budget_{budget_15}",
            ),
        }
    elif exp_id == "experiment_73d_large_rank_combined_guard":
        combined2 = rank_ensemble_guarded_indices(bundles, budget2, "rocket_or_two_model_top", guard_count=budget3)
        combined3 = rank_ensemble_guarded_indices(bundles, budget3, "rocket_or_two_model_top", guard_count=max(budget3, budget2 * 2))
        strict2 = rank_ensemble_guarded_indices(bundles, budget2, "rocket_top", guard_count=budget2)
        strategies = {
            "large_rank_combined_guard_1pct": selection(
                base_indices if not large_case else (base_indices | combined2),
                base_score_source_for_indices(combined2, bundles) if large_case and combined2 and not base_indices else base["score_source"],
                f"large_{large_case}_base_union_rank_combined_guard_{budget2}",
            ),
            "large_rank_combined_guard_2pct": selection(
                base_indices if not large_case else (base_indices | combined3),
                base_score_source_for_indices(combined3, bundles) if large_case and combined3 and not base_indices else base["score_source"],
                f"large_{large_case}_base_union_rank_combined_guard_{budget3}",
            ),
            "large_rank_strict_rocket_guard_1pct": selection(
                base_indices if not large_case else (base_indices | strict2),
                base_score_source_for_indices(strict2, bundles) if large_case and strict2 and not base_indices else base["score_source"],
                f"large_{large_case}_base_union_rank_strict_rocket_guard_{budget2}",
            ),
        }
    elif exp_id == "experiment_74a_large_rank_margin_guard":
        margin2 = rank_ensemble_margin_indices(bundles, budget2, min_margin=1.0)
        margin3 = rank_ensemble_margin_indices(bundles, budget3, min_margin=1.0)
        margin3_strict = rank_ensemble_margin_indices(bundles, budget3, min_margin=2.0)
        strategies = {
            "large_rank_margin_1pct": selection(
                base_indices if not large_case else (base_indices | margin2),
                "rocket_exp40",
                f"large_{large_case}_base_union_rank_margin1_budget_{budget2}",
            ),
            "large_rank_margin_2pct": selection(
                base_indices if not large_case else (base_indices | margin3),
                "rocket_exp40",
                f"large_{large_case}_base_union_rank_margin1_budget_{budget3}",
            ),
            "large_rank_margin_strict_2pct": selection(
                base_indices if not large_case else (base_indices | margin3_strict),
                "rocket_exp40",
                f"large_{large_case}_base_union_rank_margin2_budget_{budget3}",
            ),
        }
    elif exp_id == "experiment_74b_large_rank_family_budget":
        fam_budget2 = family_adjusted_large_budget(record["family"], budget2, mode="conservative")
        fam_budget3 = family_adjusted_large_budget(record["family"], budget3, mode="conservative")
        fam_rank2 = rank_ensemble_indices(bundles, fam_budget2)
        fam_rank3 = rank_ensemble_indices(bundles, fam_budget3)
        fam_rocket3 = rank_ensemble_guarded_indices(bundles, fam_budget3, "rocket_top", guard_count=max(fam_budget3, budget2))
        strategies = {
            "large_family_rank_budget_1pct": selection(
                base_indices if not large_case else (base_indices | fam_rank2),
                "rocket_exp40",
                f"large_{large_case}_family_{record['family']}_rank_budget_{fam_budget2}_from_{budget2}",
            ),
            "large_family_rank_budget_2pct": selection(
                base_indices if not large_case else (base_indices | fam_rank3),
                "rocket_exp40",
                f"large_{large_case}_family_{record['family']}_rank_budget_{fam_budget3}_from_{budget3}",
            ),
            "large_family_rocket_guard_2pct": selection(
                base_indices if not large_case else (base_indices | fam_rocket3),
                base_score_source_for_indices(fam_rocket3, bundles) if large_case and fam_rocket3 and not base_indices else base["score_source"],
                f"large_{large_case}_family_{record['family']}_rocket_guard_budget_{fam_budget3}_from_{budget3}",
            ),
        }
    elif exp_id == "experiment_74c_large_rank_margin_family_guard":
        fam_budget3 = family_adjusted_large_budget(record["family"], budget3, mode="conservative")
        margin_fam = rank_ensemble_margin_indices(bundles, fam_budget3, min_margin=1.0)
        margin_fam_strict = rank_ensemble_margin_indices(bundles, fam_budget3, min_margin=2.0)
        rocket_margin = rank_ensemble_guarded_indices(bundles, fam_budget3, "rocket_top", guard_count=max(fam_budget3, budget2))
        strategies = {
            "large_margin_family_budget": selection(
                base_indices if not large_case else (base_indices | margin_fam),
                "rocket_exp40",
                f"large_{large_case}_margin1_family_budget_{fam_budget3}_from_{budget3}",
            ),
            "large_strict_margin_family_budget": selection(
                base_indices if not large_case else (base_indices | margin_fam_strict),
                "rocket_exp40",
                f"large_{large_case}_margin2_family_budget_{fam_budget3}_from_{budget3}",
            ),
            "large_rocket_margin_family_guard": selection(
                base_indices if not large_case else (base_indices | (rocket_margin & margin_fam)),
                base_score_source_for_indices(rocket_margin, bundles) if large_case and rocket_margin and not base_indices else base["score_source"],
                f"large_{large_case}_rocket_guard_intersect_margin_family_budget_{fam_budget3}_from_{budget3}",
            ),
        }
    elif exp_id == "experiment_74d_large_rank_review_tier_split":
        primary = rank_ensemble_guarded_indices(bundles, budget3, "rocket_top", guard_count=max(budget3, budget2 * 2))
        review_pool = rank_ensemble_indices(bundles, budget3) - primary
        review_top = cap_indices_count(review_pool, bundles["rocket_exp40"], max(1, budget2))
        primary_plus_review = primary | review_top
        strategies = {
            "large_primary_rocket_guard_only": selection(
                base_indices if not large_case else (base_indices | primary),
                base_score_source_for_indices(primary, bundles) if large_case and primary and not base_indices else base["score_source"],
                f"large_{large_case}_primary_rocket_guard_budget_{budget3}",
            ),
            "large_primary_plus_review_limited": selection(
                base_indices if not large_case else (base_indices | primary_plus_review),
                "rocket_exp40",
                f"large_{large_case}_primary_guard_plus_review_budget_{budget2}",
            ),
            "large_review_only_shadow": selection(
                set() if not large_case else review_top,
                "rocket_exp40",
                f"large_{large_case}_review_only_shadow_budget_{budget2}",
            ),
        }
    else:
        raise ValueError(exp_id)
    rows = []
    for strategy, selected in strategies.items():
        rows.append(
            selector_metrics_row(
                exp_id,
                dataset_name,
                record,
                y_test,
                bundles,
                strategy,
                selected,
                "relaxed_15pct",
                CALIBRATION_PROFILES["relaxed_15pct"],
                "large_data_allfp_repair_selector",
                {
                    "large_data_case": int(large_case),
                    "train_normal_count": len(record["train_series"]),
                    "large_data_budget_1pct": budget2,
                    "large_data_budget_2pct": budget3,
                    "base_score_source_name": base["score_source"],
                    "source_disagreement": int(source_disagreement),
                    "rocket_exp55_or_exp56_overlap_any": int(overlap_any),
                },
            )
        )
    return rows


def run_large_data_repair_exp(exp_id, dataset_limit=None):
    _, datasets = read_csv_candidates()
    if dataset_limit:
        datasets = datasets[:dataset_limit]
    rows = []
    threshold_rates = CALIBRATION_PROFILES["relaxed_15pct"]
    for pos, dataset_name in enumerate(datasets, 1):
        record, y_test, bundles = load_candidate_predictions(dataset_name, threshold_rates=threshold_rates)
        rows.extend(large_data_strategy_rows(exp_id, dataset_name, record, y_test, bundles))
        if pos % 50 == 0:
            print(f"{exp_id} progress {pos}/{len(datasets)}", flush=True)
    return rows


def summarize(rows):
    out = []
    for strategy in sorted({row["selector_name"] for row in rows}):
        subset = [row for row in rows if row["selector_name"] == strategy]
        vals = lambda key: [as_float(row[key]) for row in subset]
        out.append(
            {
                "experiment_id": subset[0]["experiment_id"],
                "selector_name": strategy,
                "config_name": strategy,
                "threshold_method": "selector",
                "num_datasets": len(subset),
                "num_families": len({row["family"] for row in subset}),
                "mean_auc_roc": float(np.mean(vals("auc_roc"))),
                "mean_auc_pr": float(np.mean(vals("auc_pr"))),
                "mean_f1": float(np.mean(vals("f1"))),
                "median_f1": float(np.median(vals("f1"))),
                "p25_f1": float(np.percentile(vals("f1"), 25)),
                "zero_f1_count": sum(1 for row in subset if as_float(row["f1"]) == 0.0),
                "ge_0_5_count": sum(1 for row in subset if as_float(row["f1"]) >= 0.5),
                "mean_predicted_count": float(np.mean(vals("predicted_count"))),
                "mean_anomaly_count": float(np.mean(vals("anomaly_count"))),
                "mean_tp": float(np.mean(vals("tp"))),
                "mean_fp": float(np.mean(vals("fp"))),
                "mean_fn": float(np.mean(vals("fn"))),
                "mean_train_exceed_rate": float(np.mean(vals("train_exceed_rate"))),
                "mean_oracle_f1": float(np.mean(vals("oracle_f1"))),
            }
        )
    return sorted(out, key=lambda row: (row["mean_f1"], -row["mean_fp"]), reverse=True)


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(exp_id, dataset_limit=None):
    if EXPERIMENTS[exp_id]["kind"] == "csv_selector":
        rows = run_exp60(exp_id, dataset_limit)
    elif EXPERIMENTS[exp_id]["kind"] == "calibrated_selector":
        rows = run_calibrated_exp69(exp_id, dataset_limit)
    elif EXPERIMENTS[exp_id]["kind"] == "calibrated_fallback_selector":
        rows = run_calibrated_fallback_exp69b(exp_id, dataset_limit)
    elif EXPERIMENTS[exp_id]["kind"] == "zero_mode_selector":
        rows = run_zero_mode_exp70(exp_id, dataset_limit)
    elif EXPERIMENTS[exp_id]["kind"] == "large_data_repair_selector":
        rows = run_large_data_repair_exp(exp_id, dataset_limit)
    else:
        rows = run_index_exp(exp_id, dataset_limit)
    write_csv(results_path(exp_id), rows)
    summary = summarize(rows)
    write_csv(summary_path(exp_id), summary)
    with log_path(exp_id).open("w") as f:
        f.write(f"{exp_id} finished. rows={len(rows)} datasets={len({row['dataset_name'] for row in rows})}\n")
        if summary:
            f.write(str(summary[0]) + "\n")
    print(f"{exp_id} finished. rows={len(rows)} datasets={len({row['dataset_name'] for row in rows})}")
    if summary:
        print(summary[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id", choices=sorted(EXPERIMENTS))
    parser.add_argument("--dataset-limit", type=int, default=None)
    args = parser.parse_args()
    run_experiment(args.experiment_id, args.dataset_limit)


if __name__ == "__main__":
    main()
