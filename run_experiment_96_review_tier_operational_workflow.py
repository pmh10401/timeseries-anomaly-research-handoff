from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from run_experiment_89_74d_with_exp84_candidate import as_float
from run_experiment_60_62_rocket_imaging_selector_variants import results_path, summary_path


DATA_DIR = Path("/Users/minho/Documents/Dataset")
EXPERIMENT_ID = "experiment_96_review_tier_operational_workflow"
EXP95_PATH = DATA_DIR / "experiment_95_topk_review_tier_results.csv"
STDOUT_LOG = DATA_DIR / f"{EXPERIMENT_ID}_stdout.log"

POLICIES = {
    "hard_only_exp93": {
        "source_selector": "review_top1_strict",
        "review_enabled": False,
        "recommendation_rank": 2,
        "description": "Exp93 hard alerts only; no review candidates shown.",
    },
    "review_lane_top1_strict": {
        "source_selector": "review_top1_strict",
        "review_enabled": True,
        "recommendation_rank": 0,
        "description": "Exp93 hard alerts plus one strict review candidate lane.",
    },
    "review_lane_top2_balanced": {
        "source_selector": "review_top2_balanced",
        "review_enabled": True,
        "recommendation_rank": 1,
        "description": "Exp93 hard alerts plus up to two balanced review candidates.",
    },
    "review_lane_top3_broad": {
        "source_selector": "review_top3_broad",
        "review_enabled": True,
        "recommendation_rank": 3,
        "description": "Diagnostic broad review lane; not recommended as default.",
    },
}


def read_dict_rows(path: Path):
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


def policy_row(source_row, policy_name, policy):
    review_enabled = bool(policy["review_enabled"])
    review_count = int(as_float(source_row.get("review_candidate_count"))) if review_enabled else 0
    review_tp = int(as_float(source_row.get("review_tp"))) if review_enabled else 0
    review_fp = int(as_float(source_row.get("review_fp"))) if review_enabled else 0
    combined_f1 = as_float(source_row.get("combined_f1")) if review_enabled else as_float(source_row.get("f1"))
    combined_zero = int(combined_f1 == 0.0)
    out = dict(source_row)
    out.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "selector_name": policy_name,
            "config_name": policy_name,
            "review_policy": policy_name,
            "operational_recommendation_rank": policy["recommendation_rank"],
            "selector_reason": policy["description"],
            "score_family": "review_tier_operational_workflow",
            "review_enabled": int(review_enabled),
            "review_candidate_count": review_count,
            "review_tp": review_tp,
            "review_fp": review_fp,
            "combined_f1": combined_f1,
            "combined_zero_f1": combined_zero,
            "review_hit": int(review_tp > 0),
            "hard_zero_f1": int(as_float(source_row.get("f1")) == 0.0),
            "zero_f1_rescued_by_review": int(
                review_enabled and as_float(source_row.get("f1")) == 0.0 and combined_f1 > 0.0
            ),
        }
    )
    return out


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
        review_precision = review_tp / max(1.0, review_tp + review_fp)
        hard_precision = hard_tp / max(1.0, hard_tp + hard_fp)
        by_family = {}
        for row in subset:
            by_family.setdefault(row["family"], []).append(as_float(row.get("combined_f1")))
        out.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "selector_name": selector,
                "config_name": selector,
                "threshold_method": "selector",
                "num_datasets": len(subset),
                "num_families": len(by_family),
                "mean_auc_roc": float(np.mean(vals("auc_roc"))),
                "mean_auc_pr": float(np.mean(vals("auc_pr"))),
                "mean_f1": float(np.mean(f1s)),
                "median_f1": float(np.median(f1s)),
                "zero_f1_count": sum(1 for value in f1s if value == 0.0),
                "mean_combined_f1": float(np.mean(combined_f1s)),
                "combined_zero_f1_count": sum(1 for value in combined_f1s if value == 0.0),
                "family_macro_combined_f1": float(np.mean([np.mean(v) for v in by_family.values()])),
                "mean_predicted_count": float(np.mean(vals("predicted_count"))),
                "mean_fp": float(np.mean(vals("fp"))),
                "mean_tp": float(np.mean(vals("tp"))),
                "mean_fn": float(np.mean(vals("fn"))),
                "alert_precision": hard_precision,
                "mean_review_candidate_count": float(np.mean(vals("review_candidate_count"))),
                "review_candidates_total": int(review_count),
                "review_candidates_per_100_datasets": float(100.0 * review_count / max(1, len(subset))),
                "review_hit_datasets": sum(1 for row in subset if as_float(row.get("review_hit")) > 0),
                "review_precision": review_precision,
                "zero_f1_rescued_by_review": sum(
                    1 for row in subset if as_float(row.get("zero_f1_rescued_by_review")) > 0
                ),
                "review_fp_total": int(review_fp),
                "review_tp_total": int(review_tp),
                "mean_oracle_f1": float(np.mean(vals("oracle_f1"))),
                "operational_recommendation_rank": int(min(vals("operational_recommendation_rank") or [99])),
            }
        )
    return sorted(out, key=lambda row: row["operational_recommendation_rank"])


def run_experiment():
    source_rows = read_dict_rows(EXP95_PATH)
    by_selector = {}
    for row in source_rows:
        by_selector.setdefault(row.get("selector_name"), []).append(row)
    out_rows = []
    for policy_name, policy in POLICIES.items():
        source_selector = policy["source_selector"]
        rows = by_selector.get(source_selector, [])
        if not rows:
            raise SystemExit(f"Missing Exp95 selector rows: {source_selector}")
        out_rows.extend(policy_row(row, policy_name, policy) for row in rows)
    write_csv(results_path(EXPERIMENT_ID), out_rows)
    summary = summarize(out_rows)
    write_csv(summary_path(EXPERIMENT_ID), summary)
    STDOUT_LOG.write_text(f"{EXPERIMENT_ID} finished rows={len(out_rows)}\n{summary[0] if summary else ''}\n")
    print(f"{EXPERIMENT_ID} finished rows={len(out_rows)}")
    if summary:
        print(summary[0])


if __name__ == "__main__":
    run_experiment()
