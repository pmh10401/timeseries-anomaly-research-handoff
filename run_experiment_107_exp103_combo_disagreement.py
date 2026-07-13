from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from run_experiment_60_62_rocket_imaging_selector_variants import results_path, summary_path
from run_experiment_89_74d_with_exp84_candidate import as_float, format_indices, parse_indices
from run_rank_ensemble_calibration import load_dataset_data


DATA_DIR = Path("/Users/minho/Documents/Dataset")
EXPERIMENT_ID = "experiment_107_exp103_combo_disagreement"

EXP93_PATH = DATA_DIR / "experiment_93_nonpos_candidate_reranker_results.csv"
EXP103_PATH = DATA_DIR / "experiment_103_higher_dim_review_sources_results.csv"
EXP106_PATH = DATA_DIR / "experiment_106_gated_score_combo_selector_results.csv"

EXP93_SELECTOR = "nonpos_weak_alert_replace"
EXP103_SELECTOR = "review_all_higher_dim_sources_when_exp93_weak"
EXP106_CONSERVATIVE = "review_combo_conservative_when_exp93_weak"
EXP106_SENSITIVE = "review_combo_sensitive_when_exp93_weak"

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


def row_map(path, selector):
    return {row["dataset_name"]: row for row in read_rows(path) if row.get("selector_name") == selector}


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
        "combined_zero_f1": int(combined_f1 == 0.0),
        "review_hit": int(review_tp > 0),
    }


def cap_ordered(indices, limit):
    out = []
    for idx in indices:
        idx = int(idx)
        if idx not in out:
            out.append(idx)
        if len(out) >= limit:
            break
    return set(out)


def ordered_union(first, second, cap):
    return cap_ordered(list(first) + [idx for idx in second if idx not in first], cap)


def make_row(name, y_test, base_row, selector, review_indices, reason, extras):
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
            "score_family": "exp103_combo_disagreement",
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


def build_rows():
    base_rows = row_map(EXP93_PATH, EXP93_SELECTOR)
    exp103_rows = row_map(EXP103_PATH, EXP103_SELECTOR)
    conservative_rows = row_map(EXP106_PATH, EXP106_CONSERVATIVE)
    sensitive_rows = row_map(EXP106_PATH, EXP106_SENSITIVE)

    missing = set(base_rows) - set(exp103_rows) - set(conservative_rows) - set(sensitive_rows)
    if missing:
        raise SystemExit(f"Missing comparison rows: {len(missing)}")

    rows = []
    for name in sorted(base_rows):
        base = base_rows[name]
        _, _, y_test = load_dataset_data(name)
        exp103 = parse_indices(exp103_rows[name].get("review_candidate_indices"))
        conservative = parse_indices(conservative_rows[name].get("review_candidate_indices"))
        sensitive = parse_indices(sensitive_rows[name].get("review_candidate_indices"))

        unique_cons = conservative - exp103
        unique_sens = sensitive - exp103
        common_cons = conservative & exp103
        common_sens = sensitive & exp103
        exp103_not_cons = exp103 - conservative
        exp103_not_sens = exp103 - sensitive

        common = {
            "exp103_review_indices": format_indices(exp103),
            "combo_conservative_indices": format_indices(conservative),
            "combo_sensitive_indices": format_indices(sensitive),
            "common_conservative_exp103": format_indices(common_cons),
            "common_sensitive_exp103": format_indices(common_sens),
            "unique_conservative_not_exp103": format_indices(unique_cons),
            "unique_sensitive_not_exp103": format_indices(unique_sens),
            "exp103_not_conservative": format_indices(exp103_not_cons),
            "exp103_not_sensitive": format_indices(exp103_not_sens),
            "unique_conservative_count": len(unique_cons),
            "unique_sensitive_count": len(unique_sens),
        }

        rows.append(make_row(name, y_test, base, "baseline_exp93_hard_only", set(), "control: Exp93 hard alerts only", common))
        rows.append(make_row(name, y_test, base, "review_exp103_only", exp103, "Exp103 review lane only", common))
        rows.append(
            make_row(
                name,
                y_test,
                base,
                "review_combo_conservative_only",
                conservative,
                "Exp106 conservative combo review lane only",
                common,
            )
        )
        rows.append(
            make_row(
                name,
                y_test,
                base,
                "review_combo_sensitive_only",
                sensitive,
                "Exp106 sensitive combo review lane only",
                common,
            )
        )
        rows.append(
            make_row(
                name,
                y_test,
                base,
                "review_unique_conservative_not_exp103",
                unique_cons,
                "Only conservative combo candidates not already proposed by Exp103",
                common,
            )
        )
        rows.append(
            make_row(
                name,
                y_test,
                base,
                "review_unique_sensitive_not_exp103",
                unique_sens,
                "Only sensitive combo candidates not already proposed by Exp103",
                common,
            )
        )
        rows.append(
            make_row(
                name,
                y_test,
                base,
                "review_exp103_plus_unique_conservative_cap3",
                ordered_union(exp103, unique_cons, 3),
                "Exp103 plus conservative combo-only candidates capped at three",
                common,
            )
        )
        rows.append(
            make_row(
                name,
                y_test,
                base,
                "review_exp103_plus_unique_sensitive_cap4",
                ordered_union(exp103, unique_sens, 4),
                "Exp103 plus sensitive combo-only candidates capped at four",
                common,
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
                "mean_review_candidate_count": float(np.mean(vals("review_candidate_count"))),
                "review_candidates_total": int(review_count),
                "review_candidates_per_100_datasets": float(100.0 * review_count / max(1, len(subset))),
                "review_hit_datasets": sum(1 for row in subset if as_float(row.get("review_hit")) > 0),
                "review_precision": review_tp / max(1.0, review_tp + review_fp),
                "review_tp_total": int(review_tp),
                "review_fp_total": int(review_fp),
                "mean_oracle_f1": float(np.mean(vals("oracle_f1"))),
            }
        )
    return sorted(out, key=lambda row: (row["mean_combined_f1"], row["review_precision"]), reverse=True)


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
