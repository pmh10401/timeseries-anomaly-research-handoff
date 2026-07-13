from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from run_original_improvement_experiment import DB_PATH, load_original_record


DATA_DIR = Path("/Users/minho/Documents/Dataset")
EXPERIMENT_ID = "experiment_88_true_agreement_exp84_selector"

BASE_PATH = DATA_DIR / "experiment_74d_large_rank_review_tier_split_results.csv"
EXP87_PATH = DATA_DIR / "experiment_87_exp84_index_diagnostics_results.csv"
DETAIL_PATH = DATA_DIR / f"{EXPERIMENT_ID}_results.csv"
SUMMARY_PATH = DATA_DIR / f"{EXPERIMENT_ID}_summary.csv"

BASE_PRIMARY_SELECTOR = "large_primary_rocket_guard_only"
BASE_REVIEW_SELECTOR = "large_primary_plus_review_limited"
EXP87_CONFIG = "aeon_mrh_mr1024_hk4_g32_prune1024_stable_tail_local_gap_knn3"


def as_float(value, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_indices(value) -> Set[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return set()
    return {int(float(item)) for item in text.split()}


def format_indices(indices) -> str:
    return " ".join(str(int(idx)) for idx in sorted(indices))


def first_top_index(row: Optional[Mapping]) -> Optional[int]:
    if row is None:
        return None
    indices = parse_indices(row.get("top_score_indices"))
    if not indices:
        text = str(row.get("top_score_indices", "")).strip()
        if text and text.lower() != "nan":
            return int(float(text.split()[0]))
        return None
    text = str(row.get("top_score_indices", "")).strip()
    return int(float(text.split()[0])) if text else min(indices)


def predicted_rate(row: Mapping) -> float:
    return as_float(row.get("predicted_count")) / max(as_int(row.get("test_size")), 1)


def row_by_dataset(df: pd.DataFrame, selector_name: str) -> dict:
    subset = df[df["selector_name"] == selector_name].copy()
    if subset["dataset_name"].duplicated().any():
        dup = subset.loc[subset["dataset_name"].duplicated(), "dataset_name"].head(10).tolist()
        raise SystemExit(f"duplicate rows for {selector_name}: {dup}")
    return {str(row["dataset_name"]): row for _, row in subset.iterrows()}


def exp87_by_threshold(df: pd.DataFrame) -> dict:
    subset = df[df["config_name"] == EXP87_CONFIG].copy()
    out = {}
    for _, row in subset.iterrows():
        key = (str(row["dataset_name"]), str(row["threshold_method"]))
        if key in out:
            raise SystemExit(f"duplicate Exp87 row: {key}")
        out[key] = row
    return out


def evaluate_selected_indices(dataset_name: str, selected_indices: Set[int]) -> dict:
    y_true = load_original_record(dataset_name, DB_PATH)["y_test"]
    selected = np.zeros(len(y_true), dtype=np.int64)
    valid = [idx for idx in selected_indices if 0 <= idx < len(selected)]
    if valid:
        selected[valid] = 1
    tp = int(((selected == 1) & (y_true == 1)).sum())
    fp = int(((selected == 1) & (y_true == 0)).sum())
    fn = int(((selected == 0) & (y_true == 1)).sum())
    return {
        "predicted_count": int(selected.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "f1": float(f1_score(y_true, selected, zero_division=0)),
    }


def train_safe(row: Optional[Mapping], max_train_exceed=0.015) -> bool:
    return row is not None and as_float(row.get("train_exceed_rate"), default=1.0) <= max_train_exceed


def confident_top1(row: Optional[Mapping], min_threshold_margin=0.0, min_top_gap=0.0) -> bool:
    if row is None:
        return False
    return (
        as_float(row.get("top1_threshold_margin"), default=-1.0) >= min_threshold_margin
        and as_float(row.get("top1_top2_margin"), default=0.0) >= min_top_gap
    )


def make_passthrough_row(source_row: pd.Series, selector_name: str, source_name: str, reason: str) -> dict:
    out = source_row.to_dict()
    out["experiment_id"] = EXPERIMENT_ID
    out["selector_name"] = selector_name
    out["selector_reason"] = reason
    out["config_name"] = selector_name
    out["threshold_method"] = "selector"
    out["score_source_name"] = source_name
    out["selected_source_experiment_id"] = source_row.get("experiment_id", "")
    out["selected_source_config_name"] = source_row.get("config_name", "")
    out["selected_source_threshold_method"] = source_row.get("threshold_method", "")
    out["selected_source_selector_name"] = source_row.get("selector_name", "")
    out["selected_indices"] = source_row.get("selected_indices", "")
    out["agreement_overlap_count"] = 0
    out["used_exp87_specialist"] = int(str(source_name).startswith("exp87"))
    out["used_computed_indices"] = 0
    out["base_predicted_count"] = source_row.get("predicted_count", "")
    out["exp87_predicted_count"] = ""
    return out


def make_computed_row(
    base_row: pd.Series,
    exp87_row: pd.Series,
    selector_name: str,
    selected_indices: Set[int],
    reason: str,
) -> dict:
    metrics = evaluate_selected_indices(str(base_row["dataset_name"]), selected_indices)
    base_indices = parse_indices(base_row.get("selected_indices"))
    exp87_indices = parse_indices(exp87_row.get("selected_indices"))
    out = base_row.to_dict()
    out.update(metrics)
    out["experiment_id"] = EXPERIMENT_ID
    out["selector_name"] = selector_name
    out["selector_reason"] = reason
    out["config_name"] = selector_name
    out["threshold_method"] = "selector"
    out["score_source_name"] = "exp87_74d_true_agreement"
    out["selected_source_experiment_id"] = exp87_row.get("experiment_id", "")
    out["selected_source_config_name"] = exp87_row.get("config_name", "")
    out["selected_source_threshold_method"] = exp87_row.get("threshold_method", "")
    out["selected_source_selector_name"] = ""
    out["selected_indices"] = format_indices(selected_indices)
    out["auc_roc"] = exp87_row.get("auc_roc", base_row.get("auc_roc", ""))
    out["auc_pr"] = exp87_row.get("auc_pr", base_row.get("auc_pr", ""))
    out["oracle_f1"] = exp87_row.get("oracle_f1", base_row.get("oracle_f1", ""))
    out["train_exceed_rate"] = exp87_row.get("train_exceed_rate", base_row.get("train_exceed_rate", ""))
    out["agreement_overlap_count"] = len(base_indices & exp87_indices)
    out["used_exp87_specialist"] = 1
    out["used_computed_indices"] = 1
    out["base_predicted_count"] = len(base_indices)
    out["exp87_predicted_count"] = len(exp87_indices)
    out["exp87_top1_threshold_margin"] = exp87_row.get("top1_threshold_margin", "")
    out["exp87_top1_top2_margin"] = exp87_row.get("top1_top2_margin", "")
    return out


def choose_row(
    selector_name: str,
    base_primary: pd.Series,
    base_review: pd.Series,
    exp87_rows: Mapping[Tuple[str, str], pd.Series],
) -> dict:
    dataset_name = str(base_primary["dataset_name"])
    exp87_cap3 = exp87_rows.get((dataset_name, "count_cap_3pct"))
    exp87_fg = exp87_rows.get((dataset_name, "family_guard_v1"))
    exp87_cap2 = exp87_rows.get((dataset_name, "count_cap_2pct"))

    if selector_name == "baseline_74d_primary":
        return make_passthrough_row(base_primary, selector_name, "exp74d_primary", "control")
    if selector_name == "baseline_74d_review_limited":
        return make_passthrough_row(base_review, selector_name, "exp74d_review_limited", "control")

    if exp87_cap3 is None:
        return make_passthrough_row(base_primary, selector_name, "exp74d_primary", "outside Exp87 hard subset")

    base_indices = parse_indices(base_primary.get("selected_indices"))

    if selector_name in {
        "agreement_cap3_intersection_else_primary",
        "agreement_fg_intersection_else_primary",
        "agreement_cap2_intersection_else_primary",
        "agreement_cap3_only_else_primary",
        "agreement_fg_only_else_primary",
    }:
        exp87_row = {
            "agreement_cap3_intersection_else_primary": exp87_cap3,
            "agreement_fg_intersection_else_primary": exp87_fg,
            "agreement_cap2_intersection_else_primary": exp87_cap2,
            "agreement_cap3_only_else_primary": exp87_cap3,
            "agreement_fg_only_else_primary": exp87_fg,
        }[selector_name]
        exp87_indices = parse_indices(exp87_row.get("selected_indices")) if exp87_row is not None else set()
        overlap = base_indices & exp87_indices
        if overlap:
            return make_computed_row(base_primary, exp87_row, selector_name, overlap, "Exp74d and Exp87 selected the same index")
        if selector_name.endswith("_only_else_primary"):
            return make_computed_row(base_primary, exp87_row, selector_name, set(), "no true agreement")
        return make_passthrough_row(base_primary, selector_name, "exp74d_primary", "no agreement fallback")

    if selector_name == "agreement_or_top1_noalert_fg_else_primary":
        exp87_indices = parse_indices(exp87_fg.get("selected_indices")) if exp87_fg is not None else set()
        overlap = base_indices & exp87_indices
        if overlap:
            return make_computed_row(base_primary, exp87_fg, selector_name, overlap, "true agreement")
        top1 = first_top_index(exp87_fg)
        if not base_indices and top1 is not None and train_safe(exp87_fg) and confident_top1(exp87_fg):
            return make_computed_row(base_primary, exp87_fg, selector_name, {top1}, "Exp74d no-alert repaired by train-safe Exp87 top1")
        return make_passthrough_row(base_primary, selector_name, "exp74d_primary", "fallback")

    if selector_name == "top1_noalert_margin_fg_else_primary":
        top1 = first_top_index(exp87_fg)
        if not base_indices and top1 is not None and train_safe(exp87_fg) and confident_top1(exp87_fg, min_threshold_margin=0.0, min_top_gap=0.0):
            return make_computed_row(base_primary, exp87_fg, selector_name, {top1}, "Exp74d no-alert repaired by Exp87 top1")
        return make_passthrough_row(base_primary, selector_name, "exp74d_primary", "fallback")

    if selector_name == "top1_noalert_strong_margin_fg_else_primary":
        top1 = first_top_index(exp87_fg)
        if not base_indices and top1 is not None and train_safe(exp87_fg) and confident_top1(exp87_fg, min_threshold_margin=0.05, min_top_gap=0.01):
            return make_computed_row(base_primary, exp87_fg, selector_name, {top1}, "Exp74d no-alert repaired by strong Exp87 top1")
        return make_passthrough_row(base_primary, selector_name, "exp74d_primary", "fallback")

    raise SystemExit(f"unknown selector: {selector_name}")


def build_detail() -> pd.DataFrame:
    if not BASE_PATH.exists() or not EXP87_PATH.exists():
        raise SystemExit("missing Exp74d or Exp87 input CSV")
    base = pd.read_csv(BASE_PATH)
    exp87 = pd.read_csv(EXP87_PATH)
    primary = row_by_dataset(base, BASE_PRIMARY_SELECTOR)
    review = row_by_dataset(base, BASE_REVIEW_SELECTOR)
    exp87_rows = exp87_by_threshold(exp87)

    if set(primary) != set(review):
        raise SystemExit("baseline coverage mismatch")

    selectors = [
        "baseline_74d_primary",
        "baseline_74d_review_limited",
        "agreement_cap3_intersection_else_primary",
        "agreement_fg_intersection_else_primary",
        "agreement_cap2_intersection_else_primary",
        "agreement_cap3_only_else_primary",
        "agreement_fg_only_else_primary",
        "agreement_or_top1_noalert_fg_else_primary",
        "top1_noalert_margin_fg_else_primary",
        "top1_noalert_strong_margin_fg_else_primary",
    ]
    rows = []
    for dataset_name in sorted(primary):
        for selector in selectors:
            rows.append(choose_row(selector, primary[dataset_name], review[dataset_name], exp87_rows))
    detail = pd.DataFrame(rows)
    counts = detail.groupby("selector_name")["dataset_name"].nunique()
    bad = counts[counts != len(primary)]
    if not bad.empty:
        raise SystemExit(f"coverage mismatch: {bad.to_dict()}")
    return detail


def family_macro(group: pd.DataFrame) -> float:
    family_means = group.groupby("family", dropna=False)["f1"].mean()
    return float(family_means.mean()) if not family_means.empty else 0.0


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (selector_name, config_name, threshold_method), group in detail.groupby(
        ["selector_name", "config_name", "threshold_method"], dropna=False
    ):
        f1 = pd.to_numeric(group["f1"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "selector_name": selector_name,
                "config_name": config_name,
                "threshold_method": threshold_method,
                "num_datasets": int(group["dataset_name"].nunique()),
                "num_families": int(group["family"].nunique()),
                "mean_auc_roc": pd.to_numeric(group["auc_roc"], errors="coerce").mean(),
                "mean_auc_pr": pd.to_numeric(group["auc_pr"], errors="coerce").mean(),
                "mean_f1": float(f1.mean()),
                "median_f1": float(f1.median()),
                "p25_f1": float(f1.quantile(0.25)),
                "zero_f1_count": int((f1 <= 0).sum()),
                "ge_0_5_count": int((f1 >= 0.5).sum()),
                "family_macro_f1": family_macro(group),
                "mean_predicted_count": pd.to_numeric(group["predicted_count"], errors="coerce").mean(),
                "mean_anomaly_count": pd.to_numeric(group["anomaly_count"], errors="coerce").mean(),
                "mean_tp": pd.to_numeric(group["tp"], errors="coerce").mean(),
                "mean_fp": pd.to_numeric(group["fp"], errors="coerce").mean(),
                "mean_fn": pd.to_numeric(group["fn"], errors="coerce").mean(),
                "mean_train_exceed_rate": pd.to_numeric(group["train_exceed_rate"], errors="coerce").mean(),
                "mean_oracle_f1": pd.to_numeric(group["oracle_f1"], errors="coerce").mean(),
                "exp87_used_datasets": int(pd.to_numeric(group["used_exp87_specialist"], errors="coerce").sum()),
                "computed_index_rows": int(pd.to_numeric(group["used_computed_indices"], errors="coerce").sum()),
                "mean_agreement_overlap_count": pd.to_numeric(group["agreement_overlap_count"], errors="coerce").mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_f1", "zero_f1_count"], ascending=[False, True])


def run_experiment() -> None:
    detail = build_detail()
    summary = summarize(detail)
    DETAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"wrote {DETAIL_PATH} rows={len(detail)}")
    print(f"wrote {SUMMARY_PATH} rows={len(summary)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run_experiment()
