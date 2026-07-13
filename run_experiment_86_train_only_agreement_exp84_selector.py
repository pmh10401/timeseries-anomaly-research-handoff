from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import pandas as pd


DATA_DIR = Path("/Users/minho/Documents/Dataset")
EXPERIMENT_ID = "experiment_86_train_only_agreement_exp84_selector"

BASE_PATH = DATA_DIR / "experiment_74d_large_rank_review_tier_split_results.csv"
SPECIALIST_PATH = DATA_DIR / "experiment_84_feature_pruning_operational_stability_results.csv"
DETAIL_PATH = DATA_DIR / f"{EXPERIMENT_ID}_results.csv"
SUMMARY_PATH = DATA_DIR / f"{EXPERIMENT_ID}_summary.csv"

BASE_PRIMARY_SELECTOR = "large_primary_rocket_guard_only"
BASE_REVIEW_SELECTOR = "large_primary_plus_review_limited"
SPECIALIST_CONFIG = "aeon_mrh_mr1024_hk4_g32_prune1024_stable_tail_local_gap_knn3"
SPECIALIST_CAP3 = "count_cap_3pct"
SPECIALIST_FAMILY_GUARD = "family_guard_v1"
SPECIALIST_CAP2 = "count_cap_2pct"


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


def require_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"missing required input: {path}")


def load_inputs() -> Tuple[pd.DataFrame, Dict[Tuple[str, str], pd.Series]]:
    require_file(BASE_PATH)
    require_file(SPECIALIST_PATH)
    base = pd.read_csv(BASE_PATH)
    specialist = pd.read_csv(SPECIALIST_PATH)

    missing_base = {BASE_PRIMARY_SELECTOR, BASE_REVIEW_SELECTOR} - set(base["selector_name"].astype(str))
    if missing_base:
        raise SystemExit(f"missing baseline selectors in {BASE_PATH.name}: {sorted(missing_base)}")

    specialist = specialist[
        (specialist["config_name"] == SPECIALIST_CONFIG)
        & specialist["threshold_method"].isin(
            [SPECIALIST_CAP3, SPECIALIST_FAMILY_GUARD, SPECIALIST_CAP2]
        )
    ].copy()
    if specialist.empty:
        raise SystemExit(f"missing Exp84 specialist rows for {SPECIALIST_CONFIG}")

    by_threshold: Dict[Tuple[str, str], pd.Series] = {}
    for _, row in specialist.iterrows():
        key = (str(row["dataset_name"]), str(row["threshold_method"]))
        if key in by_threshold:
            raise SystemExit(f"duplicate specialist row: {key}")
        by_threshold[key] = row
    return base, by_threshold


def specialist_row(
    dataset_name: str,
    threshold_method: str,
    by_threshold: Mapping[Tuple[str, str], pd.Series],
) -> Optional[pd.Series]:
    return by_threshold.get((dataset_name, threshold_method))


def train_safe(row: Optional[Mapping], max_train_exceed: float) -> bool:
    if row is None:
        return False
    return as_float(row.get("train_exceed_rate"), default=1.0) <= max_train_exceed


def base_overlap_any(row: Mapping) -> bool:
    if as_int(row.get("rocket_exp55_or_exp56_overlap_any")) > 0:
        return True
    return (
        as_int(row.get("rocket_exp55_overlap"))
        + as_int(row.get("rocket_exp56_overlap"))
        + as_int(row.get("exp55_exp56_overlap"))
    ) > 0


def base_strong_overlap(row: Mapping) -> bool:
    overlaps = [
        as_int(row.get("rocket_exp55_overlap")),
        as_int(row.get("rocket_exp56_overlap")),
        as_int(row.get("exp55_exp56_overlap")),
    ]
    return max(overlaps) >= 2 or sum(1 for item in overlaps if item > 0) >= 2


def base_has_no_alert(row: Mapping) -> bool:
    return as_int(row.get("predicted_count")) <= 0


def predicted_rate(row: Mapping) -> float:
    test_size = max(as_int(row.get("test_size")), 1)
    return as_float(row.get("predicted_count")) / test_size


def choose_source(
    selector_name: str,
    base_primary: pd.Series,
    base_review: pd.Series,
    by_threshold: Mapping[Tuple[str, str], pd.Series],
) -> Tuple[pd.Series, str, str]:
    dataset_name = str(base_primary["dataset_name"])
    cap3 = specialist_row(dataset_name, SPECIALIST_CAP3, by_threshold)
    family_guard = specialist_row(dataset_name, SPECIALIST_FAMILY_GUARD, by_threshold)
    cap2 = specialist_row(dataset_name, SPECIALIST_CAP2, by_threshold)

    if selector_name == "baseline_74d_primary":
        return base_primary, "exp74d_primary", "control: existing primary baseline"
    if selector_name == "baseline_74d_review_limited":
        return base_review, "exp74d_review_limited", "control: existing review-limited baseline"

    if selector_name == "exp84_train_te005_fg_else_primary":
        if train_safe(family_guard, 0.005):
            return family_guard, "exp84_family_guard_train_te005", "Exp84 train_exceed<=0.5pct"
    elif selector_name == "exp84_train_te010_fg_else_primary":
        if train_safe(family_guard, 0.010):
            return family_guard, "exp84_family_guard_train_te010", "Exp84 train_exceed<=1.0pct"
    elif selector_name == "exp84_train_te015_fg_else_primary":
        if train_safe(family_guard, 0.015):
            return family_guard, "exp84_family_guard_train_te015", "Exp84 train_exceed<=1.5pct"
    elif selector_name == "exp84_train_te015_cap2_else_primary":
        if train_safe(cap2, 0.015):
            return cap2, "exp84_cap2_train_te015", "Exp84 cap2 train_exceed<=1.5pct"
    elif selector_name == "exp84_train_te015_cap3_else_primary":
        if train_safe(cap3, 0.015):
            return cap3, "exp84_cap3_train_te015", "Exp84 cap3 train_exceed<=1.5pct"
    elif selector_name == "exp84_agree_proxy_te015_fg_else_primary":
        if train_safe(family_guard, 0.015) and base_overlap_any(base_primary):
            return family_guard, "exp84_family_guard_agree_proxy", "train-safe Exp84 plus Exp74d internal overlap"
    elif selector_name == "exp84_strong_agree_proxy_te015_fg_else_primary":
        if train_safe(family_guard, 0.015) and base_strong_overlap(base_primary):
            return family_guard, "exp84_family_guard_strong_agree_proxy", "train-safe Exp84 plus strong Exp74d overlap"
    elif selector_name == "exp84_noalert_repair_te015_fg_else_primary":
        if train_safe(family_guard, 0.015) and base_has_no_alert(base_primary):
            return family_guard, "exp84_family_guard_noalert_repair", "train-safe Exp84 repairs Exp74d no-alert cases"
    elif selector_name == "exp84_agree_or_noalert_te015_fg_else_primary":
        if train_safe(family_guard, 0.015) and (
            base_overlap_any(base_primary) or base_has_no_alert(base_primary)
        ):
            return family_guard, "exp84_family_guard_agree_or_noalert", "train-safe Exp84 plus overlap or no-alert repair"

    return base_primary, "exp74d_primary", "fallback: primary baseline"


def with_selector_metadata(
    source_row: pd.Series,
    selector_name: str,
    source_name: str,
    reason: str,
    base_primary: pd.Series,
) -> dict:
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
    out["selected_source_predicted_rate"] = predicted_rate(source_row)
    out["used_exp84_specialist"] = int(str(source_name).startswith("exp84"))
    out["base_overlap_any"] = int(base_overlap_any(base_primary))
    out["base_strong_overlap"] = int(base_strong_overlap(base_primary))
    out["base_no_alert"] = int(base_has_no_alert(base_primary))
    out["base_predicted_count"] = as_int(base_primary.get("predicted_count"))
    out["base_train_exceed_rate"] = as_float(base_primary.get("train_exceed_rate"))
    out["selector_uses_family_performance_prior"] = 0
    return out


def build_detail_rows(base: pd.DataFrame, by_threshold: Mapping[Tuple[str, str], pd.Series]) -> List[dict]:
    primary = {
        str(row["dataset_name"]): row
        for _, row in base[base["selector_name"] == BASE_PRIMARY_SELECTOR].iterrows()
    }
    review = {
        str(row["dataset_name"]): row
        for _, row in base[base["selector_name"] == BASE_REVIEW_SELECTOR].iterrows()
    }
    if set(primary) != set(review):
        raise SystemExit("baseline primary/review dataset coverage mismatch")

    selector_names = [
        "baseline_74d_primary",
        "baseline_74d_review_limited",
        "exp84_train_te005_fg_else_primary",
        "exp84_train_te010_fg_else_primary",
        "exp84_train_te015_fg_else_primary",
        "exp84_train_te015_cap2_else_primary",
        "exp84_train_te015_cap3_else_primary",
        "exp84_agree_proxy_te015_fg_else_primary",
        "exp84_strong_agree_proxy_te015_fg_else_primary",
        "exp84_noalert_repair_te015_fg_else_primary",
        "exp84_agree_or_noalert_te015_fg_else_primary",
    ]

    rows: List[dict] = []
    for dataset_name in sorted(primary):
        base_primary = primary[dataset_name]
        base_review = review[dataset_name]
        for selector_name in selector_names:
            source_row, source_name, reason = choose_source(
                selector_name, base_primary, base_review, by_threshold
            )
            rows.append(
                with_selector_metadata(
                    source_row, selector_name, source_name, reason, base_primary
                )
            )
    return rows


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
                "mean_auc_roc": pd.to_numeric(group.get("auc_roc"), errors="coerce").mean(),
                "mean_auc_pr": pd.to_numeric(group.get("auc_pr"), errors="coerce").mean(),
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
                "mean_train_exceed_rate": pd.to_numeric(
                    group["train_exceed_rate"], errors="coerce"
                ).mean(),
                "mean_oracle_f1": pd.to_numeric(group["oracle_f1"], errors="coerce").mean(),
                "exp84_used_datasets": int(
                    pd.to_numeric(group["used_exp84_specialist"], errors="coerce").sum()
                ),
                "family_prior_used": int(
                    pd.to_numeric(
                        group["selector_uses_family_performance_prior"], errors="coerce"
                    ).max()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_f1", "zero_f1_count"], ascending=[False, True])


def validate(detail: pd.DataFrame) -> None:
    expected = 1117
    counts = detail.groupby("selector_name")["dataset_name"].nunique()
    bad = counts[counts != expected]
    if not bad.empty:
        raise SystemExit(f"dataset coverage mismatch: {bad.to_dict()}")
    if int(detail["selector_uses_family_performance_prior"].sum()) != 0:
        raise SystemExit("Exp86 must not use family performance prior")
    required = ["f1", "tp", "fp", "fn", "predicted_count", "train_exceed_rate"]
    missing = [col for col in required if col not in detail.columns]
    if missing:
        raise SystemExit(f"missing metric columns: {missing}")


def run_experiment() -> None:
    base, by_threshold = load_inputs()
    detail = pd.DataFrame(build_detail_rows(base, by_threshold))
    validate(detail)
    summary = summarize(detail)
    DETAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"wrote {DETAIL_PATH} rows={len(detail)}")
    print(f"wrote {SUMMARY_PATH} rows={len(summary)}")
    print(summary.head(12).to_string(index=False))


if __name__ == "__main__":
    run_experiment()
