from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from run_experiment_89_74d_with_exp84_candidate import as_float
from run_experiment_60_62_rocket_imaging_selector_variants import results_path, summary_path


DATA_DIR = Path("/Users/minho/Documents/Dataset")
EXPERIMENT_ID = "experiment_92_operational_hybrid_selector"
EXP91_PATH = DATA_DIR / "experiment_91_guarded_candidate_union_repair_results.csv"
STDOUT_LOG = DATA_DIR / f"{EXPERIMENT_ID}_stdout.log"


def read_dict_rows(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_selector(rows, selector_name):
    return {row["dataset_name"]: row for row in rows if row.get("selector_name") == selector_name}


def hybrid_row(dataset_name, exp89_row, exp90_safe_row, exp91_replace_row, exp90_union_row):
    replaced = as_float(exp91_replace_row.get("replaced_count")) > 0
    source = exp91_replace_row if replaced else exp90_safe_row
    out = dict(source)
    out.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "selector_name": "operational_noalert_top1_sparse_tail_replace",
            "config_name": "operational_noalert_top1_sparse_tail_replace",
            "threshold_method": "selector",
            "selector_reason": (
                "Exp90 no-alert top1 repair; if Exp89 had one non-tail alert and Exp91 found a guarded tail candidate, replace it"
            ),
            "selected_source_experiment_id": source.get("experiment_id", ""),
            "selected_source_selector_name": source.get("selector_name", ""),
            "hybrid_used_tail_replace": int(replaced),
            "reference_exp89_f1": exp89_row.get("f1", ""),
            "reference_exp90_safe_f1": exp90_safe_row.get("f1", ""),
            "reference_exp90_union_f1": exp90_union_row.get("f1", ""),
        }
    )
    return out


def passthrough(row, selector_name, reason):
    out = dict(row)
    out.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "selector_name": selector_name,
            "config_name": selector_name,
            "threshold_method": "selector",
            "selector_reason": reason,
            "selected_source_experiment_id": row.get("experiment_id", ""),
            "selected_source_selector_name": row.get("selector_name", ""),
            "hybrid_used_tail_replace": 0,
        }
    )
    return out


def summarize(rows):
    out = []
    for selector in sorted({row["selector_name"] for row in rows}):
        subset = [row for row in rows if row["selector_name"] == selector]
        vals = lambda key: [as_float(row.get(key)) for row in subset]
        f1s = vals("f1")
        by_family = {}
        for row in subset:
            by_family.setdefault(row["family"], []).append(as_float(row.get("f1")))
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
                "p25_f1": float(np.percentile(f1s, 25)),
                "zero_f1_count": sum(1 for value in f1s if value == 0.0),
                "ge_0_5_count": sum(1 for value in f1s if value >= 0.5),
                "family_macro_f1": float(np.mean([np.mean(v) for v in by_family.values()])),
                "mean_predicted_count": float(np.mean(vals("predicted_count"))),
                "mean_anomaly_count": float(np.mean(vals("anomaly_count"))),
                "mean_tp": float(np.mean(vals("tp"))),
                "mean_fp": float(np.mean(vals("fp"))),
                "mean_fn": float(np.mean(vals("fn"))),
                "mean_oracle_f1": float(np.mean(vals("oracle_f1"))),
                "tail_replace_used_datasets": sum(1 for row in subset if as_float(row.get("hybrid_used_tail_replace")) > 0),
            }
        )
    return sorted(out, key=lambda row: (row["mean_f1"], -row["mean_fp"]), reverse=True)


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


def run_experiment():
    rows = read_dict_rows(EXP91_PATH)
    exp89 = load_selector(rows, "reference_exp89_best")
    exp90_safe = load_selector(rows, "reference_exp90_noalert_top1")
    exp90_union = load_selector(rows, "reference_exp90_candidate_union")
    exp91_replace = load_selector(rows, "noalert_plus_sparse_tail_replace")
    datasets = sorted(exp89)
    if not (set(exp89) == set(exp90_safe) == set(exp90_union) == set(exp91_replace)):
        raise SystemExit("Exp91 reference selector coverage mismatch")
    out = []
    for dataset_name in datasets:
        out.append(passthrough(exp89[dataset_name], "reference_exp89_best", "reference: Exp89 best"))
        out.append(passthrough(exp90_safe[dataset_name], "reference_exp90_noalert_top1", "reference: Exp90 operating-safe repair"))
        out.append(passthrough(exp90_union[dataset_name], "reference_exp90_candidate_union", "reference: Exp90 aggressive union repair"))
        out.append(passthrough(exp91_replace[dataset_name], "reference_exp91_sparse_tail_replace", "reference: Exp91 guarded tail replacement"))
        out.append(hybrid_row(dataset_name, exp89[dataset_name], exp90_safe[dataset_name], exp91_replace[dataset_name], exp90_union[dataset_name]))
    write_csv(results_path(EXPERIMENT_ID), out)
    summary = summarize(out)
    write_csv(summary_path(EXPERIMENT_ID), summary)
    STDOUT_LOG.write_text(f"{EXPERIMENT_ID} finished rows={len(out)} datasets={len(datasets)}\n{summary[0] if summary else ''}\n")
    print(f"{EXPERIMENT_ID} finished rows={len(out)} datasets={len(datasets)}")
    if summary:
        print(summary[0])


if __name__ == "__main__":
    run_experiment()
