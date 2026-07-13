from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from run_experiment_60_62_rocket_imaging_selector_variants import results_path, summary_path
from run_experiment_89_74d_with_exp84_candidate import as_float, format_indices, parse_indices
from run_rank_ensemble_calibration import load_dataset_data


DATA_DIR = Path("/Users/minho/Documents/Dataset")
EXPERIMENT_ID = "experiment_100_spectral_review_and_guard"
EXP93_PATH = DATA_DIR / "experiment_93_nonpos_candidate_reranker_results.csv"
EXP95_PATH = DATA_DIR / "experiment_95_topk_review_tier_results.csv"
EXP99_PATH = DATA_DIR / "experiment_99_spectral_derivative_feature_score_results.csv"
EXP93_SELECTOR = "nonpos_weak_alert_replace"
EXP95_SELECTOR = "review_top1_strict"
EXP99_RESEARCH_SELECTOR = "research_spectral_q98_cap2"
EXP99_FAMILY_SELECTOR = "train_family_spectral_q98_cap2"
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


def cap_ranked(indices, limit):
    return set(list(indices)[:limit])


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


def agreement_guard(base_row, hard_indices, review_indices, spectral_indices):
    top_idx = top_candidate_index(base_row)
    agreement_pool = set(hard_indices) | set(review_indices)
    if top_idx is not None:
        agreement_pool.add(top_idx)
    return bool(set(spectral_indices) & agreement_pool)


def make_review_row(name, y_test, base_row, review_name, review_indices, reason, extras):
    hard_indices = parse_indices(base_row.get("selected_indices"))
    rm = review_metrics(y_test, hard_indices, review_indices)
    out = dict(base_row)
    out.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "dataset_name": name,
            "config_name": review_name,
            "selector_name": review_name,
            "selector_reason": reason,
            "score_family": "spectral_review_and_guard",
            "threshold_method": "selector",
            "selected_indices": format_indices(hard_indices),
            "review_candidate_indices": format_indices(review_indices),
            "hard_selected_indices": format_indices(hard_indices),
            "hard_f1": as_float(base_row.get("f1")),
            "hard_tp": int(as_float(base_row.get("tp"))),
            "hard_fp": int(as_float(base_row.get("fp"))),
            "hard_fn": int(as_float(base_row.get("fn"))),
            "review_enabled": 1,
            "hard_replaced": 0,
        }
    )
    out.update(rm)
    out.update(extras)
    return out


def make_hard_row(name, y_test, base_row, selector_name, selected_indices, reason, extras):
    metrics = index_metrics(y_test, selected_indices)
    out = dict(base_row)
    out.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "dataset_name": name,
            "config_name": selector_name,
            "selector_name": selector_name,
            "selector_reason": reason,
            "score_family": "spectral_review_and_guard",
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


def build_rows():
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
    exp99_research = {
        row["dataset_name"]: row
        for row in read_rows(EXP99_PATH)
        if row.get("selector_name") == EXP99_RESEARCH_SELECTOR
    }
    exp99_family = {
        row["dataset_name"]: row
        for row in read_rows(EXP99_PATH)
        if row.get("selector_name") == EXP99_FAMILY_SELECTOR
    }
    missing = set(base_rows) - set(exp99_family)
    if missing:
        raise SystemExit(f"Missing Exp99 family rows: {len(missing)}")

    rows = []
    for name in sorted(base_rows):
        base = base_rows[name]
        _, _, y_test = load_dataset_data(name)
        hard = parse_indices(base.get("selected_indices"))
        existing_review = parse_indices(review_rows.get(name, {}).get("review_candidate_indices"))
        spectral_research_row = exp99_research.get(name, {})
        spectral_family_row = exp99_family[name]
        spectral_research = parse_indices(spectral_research_row.get("selected_indices"))
        spectral_family = parse_indices(spectral_family_row.get("selected_indices"))
        research_target = int(as_float(spectral_research_row.get("research_target")))
        spectral_target = int(as_float(spectral_family_row.get("spectral_target")))
        weak = int(weak_exp93_signal(base))
        agrees = int(agreement_guard(base, hard, existing_review, spectral_family))

        common = {
            "spectral_target": spectral_target,
            "research_target": research_target,
            "exp93_weak_signal": weak,
            "spectral_agrees_with_exp93_context": agrees,
            "spectral_candidate_indices": format_indices(spectral_family),
            "existing_review_candidate_indices": format_indices(existing_review),
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
                {**common, "hard_replaced": 0},
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
                {**common, "review_source": "existing_top1"},
            )
        )

        rows.append(
            make_review_row(
                name,
                y_test,
                base,
                "review_spectral_research_only_q98_cap2",
                spectral_research - hard if research_target else set(),
                "research-only: add Exp99 spectral candidate to review lane",
                {**common, "review_source": "spectral_research", "research_only_selector": 1},
            )
        )

        rows.append(
            make_review_row(
                name,
                y_test,
                base,
                "review_spectral_family_q98_cap2",
                spectral_family - hard if spectral_target else set(),
                "train/family gated: add spectral candidate to review lane only",
                {**common, "review_source": "spectral_family"},
            )
        )

        merged_review = cap_ranked(list(existing_review) + [idx for idx in spectral_family if idx not in existing_review], 3)
        rows.append(
            make_review_row(
                name,
                y_test,
                base,
                "review_existing_top1_plus_spectral_family_cap3",
                merged_review - hard if spectral_target else existing_review,
                "existing review lane plus spectral candidates, capped at three",
                {**common, "review_source": "existing_plus_spectral_cap3"},
            )
        )

        guarded_review = bool(spectral_target and weak and agrees)
        rows.append(
            make_review_row(
                name,
                y_test,
                base,
                "review_spectral_family_when_exp93_weak_agrees",
                spectral_family - hard if guarded_review else set(),
                "train/family gated: add spectral review candidate only when Exp93 weak and spectral agrees",
                {**common, "review_source": "spectral_family_weak_agreement"},
            )
        )

        research_replace = bool(research_target and weak)
        rows.append(
            make_hard_row(
                name,
                y_test,
                base,
                "hard_guard_research_spectral_when_exp93_weak",
                spectral_research if research_replace else hard,
                "research-only: replace hard alert with spectral only when Exp93 weak signal",
                {**common, "hard_replaced": int(research_replace), "research_only_selector": 1},
            )
        )

        operational_replace = bool(spectral_target and weak and agrees)
        rows.append(
            make_hard_row(
                name,
                y_test,
                base,
                "hard_guard_spectral_agreement_when_exp93_weak",
                spectral_family if operational_replace else hard,
                "operational guard: replace only when Exp93 weak and spectral agrees with existing context",
                {**common, "hard_replaced": int(operational_replace)},
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
                "mean_combined_tp": float(np.mean(vals("combined_tp"))),
                "mean_combined_fp": float(np.mean(vals("combined_fp"))),
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
