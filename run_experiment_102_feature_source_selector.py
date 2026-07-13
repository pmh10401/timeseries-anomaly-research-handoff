from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from run_experiment_60_62_rocket_imaging_selector_variants import results_path, summary_path
from run_experiment_89_74d_with_exp84_candidate import as_float, format_indices, parse_indices
from run_rank_ensemble_calibration import load_dataset_data


DATA_DIR = Path("/Users/minho/Documents/Dataset")
EXPERIMENT_ID = "experiment_102_feature_source_selector"
EXP93_PATH = DATA_DIR / "experiment_93_nonpos_candidate_reranker_results.csv"
EXP95_PATH = DATA_DIR / "experiment_95_topk_review_tier_results.csv"
EXP99_PATH = DATA_DIR / "experiment_99_spectral_derivative_feature_score_results.csv"
EXP101_PATH = DATA_DIR / "experiment_101_shapelet_normal_prototype_results.csv"
EXP93_SELECTOR = "nonpos_weak_alert_replace"
EXP95_SELECTOR = "review_top1_strict"
EXP99_FAMILY_SELECTOR = "train_family_spectral_q98_cap2"
EXP99_RESEARCH_SELECTOR = "research_spectral_q98_cap2"
EXP101_FAMILY_SELECTOR = "train_family_shapelet_q98_cap2"
EXP101_RESEARCH_SELECTOR = "research_shapelet_q98_cap2"
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


def index_metrics(y_test, selected_indices):
    selected = set(int(i) for i in selected_indices)
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


def agreement_guard(base_row, hard_indices, existing_review, candidate_indices):
    top_idx = top_candidate_index(base_row)
    agreement_pool = set(hard_indices) | set(existing_review)
    if top_idx is not None:
        agreement_pool.add(top_idx)
    return bool(set(candidate_indices) & agreement_pool)


def cap_ranked(indices, limit):
    out = []
    for idx in indices:
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
            "score_family": "feature_source_selector",
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
            "score_family": "feature_source_selector",
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


def row_map(path, selector):
    return {row["dataset_name"]: row for row in read_rows(path) if row.get("selector_name") == selector}


def build_rows():
    base_rows = row_map(EXP93_PATH, EXP93_SELECTOR)
    existing_review_rows = row_map(EXP95_PATH, EXP95_SELECTOR)
    spectral_rows = row_map(EXP99_PATH, EXP99_FAMILY_SELECTOR)
    spectral_research_rows = row_map(EXP99_PATH, EXP99_RESEARCH_SELECTOR)
    shapelet_rows = row_map(EXP101_PATH, EXP101_FAMILY_SELECTOR)
    shapelet_research_rows = row_map(EXP101_PATH, EXP101_RESEARCH_SELECTOR)
    missing = set(base_rows) - set(spectral_rows) - set(shapelet_rows)
    if missing:
        raise SystemExit(f"Missing feature source rows: {len(missing)}")

    rows = []
    for name in sorted(base_rows):
        base = base_rows[name]
        _, _, y_test = load_dataset_data(name)
        hard = parse_indices(base.get("selected_indices"))
        existing_review = parse_indices(existing_review_rows.get(name, {}).get("review_candidate_indices"))
        spectral = parse_indices(spectral_rows[name].get("selected_indices"))
        shapelet = parse_indices(shapelet_rows[name].get("selected_indices"))
        spectral_research = parse_indices(spectral_research_rows.get(name, {}).get("selected_indices"))
        shapelet_research = parse_indices(shapelet_research_rows.get(name, {}).get("selected_indices"))

        weak = int(weak_exp93_signal(base))
        spectral_target = int(as_float(spectral_rows[name].get("spectral_target")))
        shapelet_target = int(as_float(shapelet_rows[name].get("shapelet_target")))
        spectral_agrees = int(agreement_guard(base, hard, existing_review, spectral))
        shapelet_agrees = int(agreement_guard(base, hard, existing_review, shapelet))
        spectral_active = bool(weak and spectral_target and spectral_agrees)
        shapelet_active = bool(weak and shapelet_target and shapelet_agrees)
        spectral_review = spectral - hard if spectral_active else set()
        shapelet_review = shapelet - hard if shapelet_active else set()
        feature_review = cap_ranked(list(spectral_review) + [idx for idx in shapelet_review if idx not in spectral_review], 3)
        research_review = cap_ranked(
            list(spectral_research - hard) + [idx for idx in (shapelet_research - hard) if idx not in spectral_research],
            4,
        )

        source_count = int(spectral_active) + int(shapelet_active)
        hard_feature_replace = source_count == 1 and len(feature_review) <= 2
        if hard_feature_replace and spectral_active:
            hard_feature_indices = spectral
        elif hard_feature_replace and shapelet_active:
            hard_feature_indices = shapelet
        else:
            hard_feature_indices = hard

        common = {
            "exp93_weak_signal": weak,
            "spectral_target": spectral_target,
            "shapelet_target": shapelet_target,
            "spectral_agrees": spectral_agrees,
            "shapelet_agrees": shapelet_agrees,
            "spectral_candidate_indices": format_indices(spectral),
            "shapelet_candidate_indices": format_indices(shapelet),
            "selected_feature_source_count": source_count,
            "research_only_selector": 0,
        }

        rows.append(
            make_hard_row(
                name,
                y_test,
                base,
                "baseline_exp93_hard_only",
                hard,
                "control: Exp93 hard alert default",
                {**common, "hard_replaced": 0, "selected_feature_sources": "none"},
            )
        )
        rows.append(
            make_review_row(
                name,
                y_test,
                base,
                "review_existing_top1_only",
                existing_review,
                "control: existing Exp95/96 top1 review lane",
                {**common, "selected_feature_sources": "existing"},
            )
        )
        rows.append(
            make_review_row(
                name,
                y_test,
                base,
                "review_spectral_shapelet_weak_agreement",
                feature_review,
                "operational: spectral/shapelet review candidates only when Exp93 weak and source agrees",
                {
                    **common,
                    "selected_feature_sources": ";".join(
                        source
                        for source, enabled in [("spectral", bool(spectral_review)), ("shapelet", bool(shapelet_review))]
                        if enabled
                    )
                    or "none",
                },
            )
        )
        rows.append(
            make_review_row(
                name,
                y_test,
                base,
                "review_existing_plus_feature_sources_cap3",
                cap_ranked(list(existing_review) + [idx for idx in feature_review if idx not in existing_review], 3),
                "diagnostic: existing review plus guarded spectral/shapelet sources capped at three",
                {**common, "selected_feature_sources": "existing_plus_feature_sources"},
            )
        )
        rows.append(
            make_hard_row(
                name,
                y_test,
                base,
                "hard_guard_single_feature_source_when_exp93_weak",
                hard_feature_indices,
                "diagnostic: hard replace only when exactly one guarded feature source is active",
                {
                    **common,
                    "hard_replaced": int(hard_feature_replace),
                    "selected_feature_sources": "single_guarded_source" if hard_feature_replace else "none",
                },
            )
        )
        rows.append(
            make_review_row(
                name,
                y_test,
                base,
                "review_research_spectral_shapelet_upper_bound",
                research_review,
                "research-only: Exp97 spectral/shapelet diagnostic targets as review candidates",
                {**common, "selected_feature_sources": "research_spectral_shapelet", "research_only_selector": 1},
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
                "research_only_selector": int(max(vals("research_only_selector") or [0])),
                "mean_oracle_f1": float(np.mean(vals("oracle_f1"))),
            }
        )
    return sorted(out, key=lambda row: (row["research_only_selector"], row["mean_combined_f1"], -row["mean_fp"]), reverse=True)


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
