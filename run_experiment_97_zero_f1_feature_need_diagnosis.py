from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from run_experiment_89_74d_with_exp84_candidate import as_float, parse_indices
from run_experiment_60_62_rocket_imaging_selector_variants import results_path, summary_path


DATA_DIR = Path("/Users/minho/Documents/Dataset")
EXPERIMENT_ID = "experiment_97_zero_f1_feature_need_diagnosis"
EXP93_PATH = DATA_DIR / "experiment_93_nonpos_candidate_reranker_results.csv"
EXP90_DIAG_PATH = DATA_DIR / "experiment_90_zero_f1_root_cause.csv"
EXP95_PATH = DATA_DIR / "experiment_95_topk_review_tier_results.csv"
STDOUT_LOG = DATA_DIR / f"{EXPERIMENT_ID}_stdout.log"
EXP93_SELECTOR = "nonpos_weak_alert_replace"

SPECTRAL_FAMILIES = {
    "Phoneme",
    "CricketX",
    "CricketY",
    "CricketZ",
    "GestureMidAirD1",
    "GestureMidAirD2",
    "GestureMidAirD3",
    "AllGestureWiimoteX",
    "AllGestureWiimoteY",
    "AllGestureWiimoteZ",
    "EOGHorizontalSignal",
    "EOGVerticalSignal",
    "InlineSkate",
    "EthanolLevel",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "UWaveGestureLibraryAll",
    "Haptics",
}

SHAPE_FAMILIES = {
    "Adiac",
    "ArrowHead",
    "BeetleFly",
    "BirdChicken",
    "Fish",
    "HandOutlines",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxTW",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxTW",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxTW",
    "ShapesAll",
    "ShapeletSim",
    "SwedishLeaf",
    "OSULeaf",
    "Plane",
    "Trace",
    "Worms",
    "WormsTwoClass",
    "WordSynonyms",
    "FiftyWords",
}

TINY_POOL_FAMILIES = {
    "PigAirwayPressure",
    "PigArtPressure",
    "PigCVP",
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


def load_exp93_zero_rows():
    rows = read_rows(EXP93_PATH)
    return {
        row["dataset_name"]: row
        for row in rows
        if row.get("selector_name") == EXP93_SELECTOR and as_float(row.get("f1")) == 0.0
    }


def load_exp90_diag():
    if not EXP90_DIAG_PATH.exists():
        return {}
    return {row["dataset_name"]: row for row in read_rows(EXP90_DIAG_PATH)}


def load_review_hits():
    if not EXP95_PATH.exists():
        return {}
    out = defaultdict(dict)
    for row in read_rows(EXP95_PATH):
        name = row["dataset_name"]
        selector = row.get("selector_name")
        if selector not in {"review_top1_strict", "review_top2_balanced", "review_top3_broad", "review_top5_diagnostic"}:
            continue
        out[name][selector] = {
            "review_candidate_count": int(as_float(row.get("review_candidate_count"))),
            "review_tp": int(as_float(row.get("review_tp"))),
            "review_fp": int(as_float(row.get("review_fp"))),
            "combined_f1": as_float(row.get("combined_f1")),
            "review_indices": row.get("review_candidate_indices", ""),
        }
    return out


def min_rank(diag):
    values = []
    for key in [
        "rocket_exp40_best_anom_rank",
        "exp55_best_best_anom_rank",
        "exp56_best_best_anom_rank",
    ]:
        values.append(as_float(diag.get(key), 999.0))
    return int(min(values) if values else 999)


def topk_bucket(best_rank):
    if best_rank <= 1:
        return "top1"
    if best_rank <= 3:
        return "top3"
    if best_rank <= 10:
        return "top10"
    if best_rank <= 20:
        return "top20"
    return "beyond20"


def feature_need(row, diag, review):
    family = row.get("family", "")
    train_count = int(as_float(row.get("train_normal_count"), 0))
    anomaly_count = int(as_float(row.get("anomaly_count"), 0))
    best_rank = min_rank(diag)
    review_top1_hit = int(review.get("review_top1_strict", {}).get("review_tp", 0) > 0)
    review_top2_hit = int(review.get("review_top2_balanced", {}).get("review_tp", 0) > 0)
    review_top3_hit = int(review.get("review_top3_broad", {}).get("review_tp", 0) > 0)

    if train_count <= 10 or family in TINY_POOL_FAMILIES:
        primary = "A_tiny_train_normal_pooling"
    elif review_top1_hit or review_top2_hit or review_top3_hit:
        primary = "B_review_tier_candidate"
    elif family in SPECTRAL_FAMILIES:
        primary = "C_spectral_derivative_feature"
    elif family in SHAPE_FAMILIES:
        primary = "D_shapelet_prototype_feature"
    elif best_rank <= 10:
        primary = "E_score_calibration_candidate"
    else:
        primary = "F_new_representation_needed"

    reasons = []
    if train_count <= 10:
        reasons.append("tiny_train")
    if family in TINY_POOL_FAMILIES:
        reasons.append("pig_family_pooling")
    if review_top1_hit:
        reasons.append("review_top1_hit")
    elif review_top2_hit:
        reasons.append("review_top2_hit")
    elif review_top3_hit:
        reasons.append("review_top3_hit")
    if family in SPECTRAL_FAMILIES:
        reasons.append("spectral_family")
    if family in SHAPE_FAMILIES:
        reasons.append("shape_family")
    if anomaly_count <= 1:
        reasons.append("single_anomaly_sensitive")
    if best_rank <= 10:
        reasons.append(f"best_rank_{topk_bucket(best_rank)}")
    else:
        reasons.append("rank_not_top10")
    return primary, ";".join(reasons), best_rank


def summarize(rows):
    out = []
    groups = defaultdict(list)
    for row in rows:
        groups[row["feature_need_primary"]].append(row)
    for key, subset in groups.items():
        families = Counter(row["family"] for row in subset)
        out.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "selector_name": key,
                "config_name": key,
                "threshold_method": "diagnosis",
                "num_datasets": len(subset),
                "num_families": len(families),
                "mean_auc_roc": float(np.mean([as_float(r.get("auc_roc")) for r in subset])),
                "mean_auc_pr": float(np.mean([as_float(r.get("auc_pr")) for r in subset])),
                "mean_f1": float(np.mean([as_float(r.get("f1")) for r in subset])),
                "median_f1": float(np.median([as_float(r.get("f1")) for r in subset])),
                "zero_f1_count": len(subset),
                "mean_fp": float(np.mean([as_float(r.get("fp")) for r in subset])),
                "mean_tp": float(np.mean([as_float(r.get("tp")) for r in subset])),
                "mean_fn": float(np.mean([as_float(r.get("fn")) for r in subset])),
                "mean_oracle_f1": float(np.mean([as_float(r.get("oracle_f1")) for r in subset])),
                "mean_train_normal_count": float(np.mean([as_float(r.get("train_normal_count")) for r in subset])),
                "mean_best_anomaly_rank": float(np.mean([as_float(r.get("best_anomaly_rank")) for r in subset])),
                "top_families": ";".join(f"{fam}:{count}" for fam, count in families.most_common(8)),
            }
        )
    return sorted(out, key=lambda row: row["num_datasets"], reverse=True)


def run_experiment():
    exp93_zero = load_exp93_zero_rows()
    diag_rows = load_exp90_diag()
    review_hits = load_review_hits()
    rows = []
    for name, row in sorted(exp93_zero.items()):
        diag = diag_rows.get(name, {})
        review = review_hits.get(name, {})
        primary, reasons, best_rank = feature_need(row, diag, review)
        review_top1 = review.get("review_top1_strict", {})
        review_top2 = review.get("review_top2_balanced", {})
        review_top3 = review.get("review_top3_broad", {})
        out = dict(row)
        out.update(
            {
                "experiment_id": EXPERIMENT_ID,
                "selector_name": primary,
                "config_name": primary,
                "threshold_method": "diagnosis",
                "score_family": "zero_f1_feature_need_diagnosis",
                "feature_need_primary": primary,
                "feature_need_reasons": reasons,
                "best_anomaly_rank": best_rank,
                "best_anomaly_rank_bucket": topk_bucket(best_rank),
                "exp90_root_cause": diag.get("cause", ""),
                "review_top1_tp": review_top1.get("review_tp", 0),
                "review_top2_tp": review_top2.get("review_tp", 0),
                "review_top3_tp": review_top3.get("review_tp", 0),
                "review_top1_indices": review_top1.get("review_indices", ""),
                "review_top2_indices": review_top2.get("review_indices", ""),
                "review_top3_indices": review_top3.get("review_indices", ""),
            }
        )
        rows.append(out)
    write_csv(results_path(EXPERIMENT_ID), rows)
    summary = summarize(rows)
    write_csv(summary_path(EXPERIMENT_ID), summary)
    STDOUT_LOG.write_text(f"{EXPERIMENT_ID} finished rows={len(rows)}\n{summary[0] if summary else ''}\n")
    print(f"{EXPERIMENT_ID} finished rows={len(rows)}")
    if summary:
        print(summary[0])


if __name__ == "__main__":
    run_experiment()
