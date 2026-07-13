from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from run_experiment_60_62_rocket_imaging_selector_variants import results_path, summary_path
from run_experiment_89_74d_with_exp84_candidate import as_float, format_indices, parse_indices
from run_rank_ensemble_calibration import load_dataset_data


DATA_DIR = Path("/Users/minho/Documents/Dataset")
EXPERIMENT_ID = "experiment_106_gated_score_combo_selector"
EXP93_PATH = DATA_DIR / "experiment_93_nonpos_candidate_reranker_results.csv"
EXP95_PATH = DATA_DIR / "experiment_95_topk_review_tier_results.csv"
EXP103_PATH = DATA_DIR / "experiment_103_higher_dim_review_sources_results.csv"
EXP105_PATH = DATA_DIR / "experiment_105_score_combination_methods_results.csv"

EXP93_SELECTOR = "nonpos_weak_alert_replace"
EXP95_SELECTOR = "review_top1_strict"
EXP103_SELECTOR = "review_all_higher_dim_sources_when_exp93_weak"

COMBO_SPECS = {
    "spectrogram_agreement_3of3": ("spectrogram_agreement_3of3", "fixed_agreement"),
    "spectrogram_agreement_2of3": ("spectrogram_agreement_2of3", "fixed_agreement"),
    "all_dims_rank_min_2pct": ("spectrogram_glcm_rp_all_dims_rank_min", "count_cap_2pct"),
    "glcm_rp_agreement_2of3": ("glcm_rp_agreement_2of3", "fixed_agreement"),
}

STDOUT_LOG = DATA_DIR / f"{EXPERIMENT_ID}_stdout.log"


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


def row_map(path, selector=None, config=None, threshold=None):
    out = {}
    for row in read_rows(path):
        if selector is not None and row.get("selector_name") != selector:
            continue
        if config is not None and row.get("config_name") != config:
            continue
        if threshold is not None and row.get("threshold_method") != threshold:
            continue
        out[row["dataset_name"]] = row
    return out


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


def context_indices(base_row, hard_indices, existing_review, exp103_review):
    top_idx = top_candidate_index(base_row)
    out = set(hard_indices) | set(existing_review) | set(exp103_review)
    if top_idx is not None:
        out.add(top_idx)
    return out


def cap_ranked(indices, limit):
    out = []
    for idx in indices:
        idx = int(idx)
        if idx not in out:
            out.append(idx)
        if len(out) >= limit:
            break
    return set(out)


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
            "score_family": "gated_score_combo_selector",
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
            "score_family": "gated_score_combo_selector",
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


def load_combo_maps():
    maps = {}
    for alias, (config, threshold) in COMBO_SPECS.items():
        maps[alias] = row_map(EXP105_PATH, config=config, threshold=threshold)
    return maps


def ordered_candidates(row, hard):
    selected = parse_indices(row.get("selected_indices")) - set(hard)
    ranked = parse_indices(row.get("top_score_indices"))
    return [idx for idx in ranked if idx in selected]


def build_rows():
    base_rows = row_map(EXP93_PATH, selector=EXP93_SELECTOR)
    existing_review_rows = row_map(EXP95_PATH, selector=EXP95_SELECTOR)
    exp103_rows = row_map(EXP103_PATH, selector=EXP103_SELECTOR)
    combo_maps = load_combo_maps()

    missing = set(base_rows)
    for alias, rows in combo_maps.items():
        missing = missing - set(rows)
    if missing:
        raise SystemExit(f"Missing Exp105 combo rows: {len(missing)}")

    rows = []
    for name in sorted(base_rows):
        base = base_rows[name]
        _, _, y_test = load_dataset_data(name)
        hard = parse_indices(base.get("selected_indices"))
        existing_review = parse_indices(existing_review_rows.get(name, {}).get("review_candidate_indices"))
        exp103_review = parse_indices(exp103_rows.get(name, {}).get("review_candidate_indices"))
        context = context_indices(base, hard, existing_review, exp103_review)
        weak = int(weak_exp93_signal(base))

        combo_candidates = {}
        combo_agrees = {}
        combo_safe = {}
        for alias, combo_rows in combo_maps.items():
            row = combo_rows[name]
            candidates = ordered_candidates(row, hard)
            combo_candidates[alias] = candidates
            combo_agrees[alias] = int(bool(set(candidates) & context))
            combo_safe[alias] = int(as_float(row.get("train_exceed_rate"), 1.0) <= 0.015)

        def active(alias):
            return bool(weak and combo_agrees[alias] and combo_safe[alias] and combo_candidates[alias])

        conservative_sources = ["spectrogram_agreement_3of3", "all_dims_rank_min_2pct", "glcm_rp_agreement_2of3"]
        sensitive_sources = ["spectrogram_agreement_2of3", "spectrogram_agreement_3of3", "all_dims_rank_min_2pct"]

        conservative_ordered = []
        sensitive_ordered = []
        active_conservative = []
        active_sensitive = []
        for alias in conservative_sources:
            if active(alias):
                active_conservative.append(alias)
                conservative_ordered.extend(combo_candidates[alias])
        for alias in sensitive_sources:
            if active(alias):
                active_sensitive.append(alias)
                sensitive_ordered.extend(combo_candidates[alias])

        review_conservative = cap_ranked(conservative_ordered, 3)
        review_sensitive = cap_ranked(sensitive_ordered, 4)
        review_existing_plus = cap_ranked(list(existing_review) + [idx for idx in conservative_ordered if idx not in existing_review], 3)
        review_exp103_plus = cap_ranked(list(exp103_review) + [idx for idx in conservative_ordered if idx not in exp103_review], 3)

        hard_replace = (
            weak
            and len(active_conservative) == 1
            and len(review_conservative) == 1
            and int(as_float(base.get("predicted_count"))) <= 1
        )
        hard_indices = review_conservative if hard_replace else hard

        common = {
            "exp93_weak_signal": weak,
            "context_indices": format_indices(context),
            "active_conservative_sources": ";".join(active_conservative) or "none",
            "active_sensitive_sources": ";".join(active_sensitive) or "none",
            "spectrogram_agreement_3of3_agrees": combo_agrees["spectrogram_agreement_3of3"],
            "spectrogram_agreement_2of3_agrees": combo_agrees["spectrogram_agreement_2of3"],
            "all_dims_rank_min_2pct_agrees": combo_agrees["all_dims_rank_min_2pct"],
            "glcm_rp_agreement_2of3_agrees": combo_agrees["glcm_rp_agreement_2of3"],
            "spectrogram_agreement_3of3_safe": combo_safe["spectrogram_agreement_3of3"],
            "spectrogram_agreement_2of3_safe": combo_safe["spectrogram_agreement_2of3"],
            "all_dims_rank_min_2pct_safe": combo_safe["all_dims_rank_min_2pct"],
            "glcm_rp_agreement_2of3_safe": combo_safe["glcm_rp_agreement_2of3"],
            "combo_candidates_spectrogram_agreement_3of3": format_indices(combo_candidates["spectrogram_agreement_3of3"]),
            "combo_candidates_spectrogram_agreement_2of3": format_indices(combo_candidates["spectrogram_agreement_2of3"]),
            "combo_candidates_all_dims_rank_min_2pct": format_indices(combo_candidates["all_dims_rank_min_2pct"]),
            "combo_candidates_glcm_rp_agreement_2of3": format_indices(combo_candidates["glcm_rp_agreement_2of3"]),
        }

        rows.append(
            make_hard_row(
                name,
                y_test,
                base,
                "baseline_exp93_hard_only",
                hard,
                "control: Exp93 hard alert default",
                {**common, "hard_replaced": 0},
            )
        )
        rows.append(
            make_review_row(
                name,
                y_test,
                base,
                "review_combo_conservative_when_exp93_weak",
                review_conservative,
                "review only: conservative combo sources when Exp93 weak and source agrees with context",
                common,
            )
        )
        rows.append(
            make_review_row(
                name,
                y_test,
                base,
                "review_combo_sensitive_when_exp93_weak",
                review_sensitive,
                "review only: sensitive spectrogram combo sources when Exp93 weak and source agrees with context",
                common,
            )
        )
        rows.append(
            make_review_row(
                name,
                y_test,
                base,
                "review_existing_plus_combo_cap3",
                review_existing_plus,
                "review only: existing review lane plus conservative combo candidates capped at three",
                common,
            )
        )
        rows.append(
            make_review_row(
                name,
                y_test,
                base,
                "review_exp103_plus_combo_cap3",
                review_exp103_plus,
                "review only: Exp103 review lane plus conservative combo candidates capped at three",
                common,
            )
        )
        rows.append(
            make_hard_row(
                name,
                y_test,
                base,
                "hard_single_combo_replace_when_exp93_weak",
                hard_indices,
                "diagnostic: hard replace only when one conservative combo proposes one agreed candidate",
                {**common, "hard_replaced": int(hard_replace)},
            )
        )
    return rows


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


def run_experiment():
    rows = build_rows()
    write_csv(results_path(EXPERIMENT_ID), rows)
    summary = summarize(rows)
    write_csv(summary_path(EXPERIMENT_ID), summary)
    STDOUT_LOG.write_text(f"{EXPERIMENT_ID} finished rows={len(rows)}\n{summary[0] if summary else ''}\n")
    print(f"{EXPERIMENT_ID} finished rows={len(rows)}")
    if summary:
        print(summary[0])


if __name__ == "__main__":
    run_experiment()
