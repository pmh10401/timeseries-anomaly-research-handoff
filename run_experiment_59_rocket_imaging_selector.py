import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from run_original_improvement_experiment import DATA_DIR


EXPERIMENT_ID = "experiment_59_rocket_imaging_selector"

CANDIDATES = {
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

LABEL_FREE_ALLOWED_FIELDS = {
    "dataset_name",
    "family",
    "config_name",
    "threshold_method",
    "test_size",
    "train_score_count",
    "q_effective",
    "cap_target",
    "threshold",
    "train_exceed_count",
    "train_exceed_rate",
    "predicted_count",
}
FORBIDDEN_SELECTOR_FIELDS = {
    "tp",
    "fp",
    "fn",
    "f1",
    "auc_roc",
    "auc_pr",
    "oracle_f1",
    "anomaly_count",
    "difficulty_score",
    "difficulty_reasons",
}

RESULTS_PATH = DATA_DIR / f"{EXPERIMENT_ID}_results.csv"
SUMMARY_PATH = DATA_DIR / f"{EXPERIMENT_ID}_summary.csv"
LOG_PATH = DATA_DIR / f"{EXPERIMENT_ID}_stdout.log"


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default=0):
    return int(round(as_float(value, default)))


def read_candidate_rows():
    candidate_rows = {}
    for name, spec in CANDIDATES.items():
        path = spec["path"]
        if not path.exists():
            raise FileNotFoundError(f"Missing candidate result CSV for {name}: {path}")
        selected = {}
        duplicates = Counter()
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("config_name") != spec["config_name"]:
                    continue
                if row.get("threshold_method") != spec["threshold_method"]:
                    continue
                duplicates[row["dataset_name"]] += 1
                selected[row["dataset_name"]] = row
        if not selected:
            raise RuntimeError(f"No rows selected for candidate {name}: {spec}")
        duplicate_names = [dataset for dataset, count in duplicates.items() if count > 1]
        if duplicate_names:
            raise RuntimeError(f"Duplicate selected rows for {name}: {duplicate_names[:10]}")
        candidate_rows[name] = selected
    common = set.intersection(*(set(rows) for rows in candidate_rows.values()))
    if not common:
        raise RuntimeError("No common datasets across selector candidates.")
    for name, rows in candidate_rows.items():
        missing = common.symmetric_difference(set(rows))
        if missing:
            raise RuntimeError(f"Candidate coverage mismatch for {name}: {len(missing)} datasets differ.")
        for row in rows.values():
            if as_int(row.get("cap_target")) and as_int(row.get("train_exceed_count")) > as_int(row.get("cap_target")):
                raise RuntimeError(f"train_exceed_count exceeds cap_target for {name}/{row['dataset_name']}")
            if as_float(row.get("q_effective")) and as_float(row.get("train_exceed_rate")) > as_float(row.get("q_effective")) + 1e-9:
                raise RuntimeError(f"train_exceed_rate exceeds q_effective for {name}/{row['dataset_name']}")
    return candidate_rows, sorted(common)


def candidate_row(candidates, name, dataset_name):
    return candidates[name][dataset_name]


def label_free_candidates(candidates):
    view = {}
    for candidate_name, rows in candidates.items():
        view[candidate_name] = {}
        for dataset_name, row in rows.items():
            forbidden = FORBIDDEN_SELECTOR_FIELDS & set(row)
            safe_row = {key: row.get(key, "") for key in LABEL_FREE_ALLOWED_FIELDS}
            safe_row["_forbidden_fields_removed"] = "|".join(sorted(forbidden))
            view[candidate_name][dataset_name] = safe_row
    return view


def none_row(reference):
    anomaly_count = as_int(reference.get("anomaly_count"))
    test_size = as_int(reference.get("test_size"))
    row = dict(reference)
    row.update(
        {
            "selected_candidate": "none",
            "candidate_config_name": "",
            "candidate_threshold_method": "",
            "selector_reason": "no_label_free_candidate_passed_guard",
            "predicted_count": "0",
            "tp": "0",
            "fp": "0",
            "fn": str(anomaly_count),
            "f1": "0.0",
            "auc_roc": "0.5" if test_size > anomaly_count else "0.0",
            "auc_pr": str(anomaly_count / test_size) if test_size else "0.0",
            "oracle_f1": "0.0",
            "train_exceed_count": "0",
            "train_exceed_rate": "0.0",
        }
    )
    return row


def operational_budget(row, rate=0.02):
    test_size = max(1, as_int(row.get("test_size")))
    return max(1, int(math.ceil(test_size * rate)))


def predicted_count(row):
    return as_int(row.get("predicted_count"))


def train_exceed_rate(row):
    return as_float(row.get("train_exceed_rate"))


def alert_rate(row):
    return predicted_count(row) / max(1, as_int(row.get("test_size")))


def passes_guard(row, rate=0.02, train_rate=0.025):
    count = predicted_count(row)
    return 0 < count <= operational_budget(row, rate) and train_exceed_rate(row) <= train_rate


def choose_label_free_confidence(candidates, dataset_name):
    rocket = candidate_row(candidates, "rocket_exp40", dataset_name)
    exp55 = candidate_row(candidates, "exp55_best", dataset_name)
    exp56 = candidate_row(candidates, "exp56_best", dataset_name)
    if passes_guard(rocket, rate=0.03, train_rate=0.035):
        return "rocket_exp40", "rocket_guard_pass"
    imaging = [name for name, row in [("exp55_best", exp55), ("exp56_best", exp56)] if passes_guard(row)]
    if imaging:
        best = min(imaging, key=lambda name: (train_exceed_rate(candidate_row(candidates, name, dataset_name)), predicted_count(candidate_row(candidates, name, dataset_name))))
        return best, "rocket_weak_imaging_guard_pass"
    if predicted_count(rocket) > 0 and alert_rate(rocket) <= 0.05:
        return "rocket_exp40", "rocket_nonzero_fallback"
    return "none", "no_candidate_passed_confidence"


def choose_agreement_count(candidates, dataset_name):
    rows = {name: candidate_row(candidates, name, dataset_name) for name in CANDIDATES}
    nonzero = {name for name, row in rows.items() if predicted_count(row) > 0}
    if "rocket_exp40" in nonzero and ({"exp55_best", "exp56_best"} & nonzero):
        return "rocket_exp40", "rocket_agrees_with_imaging"
    if {"exp55_best", "exp56_best"} <= nonzero and "rocket_exp40" not in nonzero:
        best = min(["exp55_best", "exp56_best"], key=lambda name: predicted_count(rows[name]))
        return best, "imaging_pair_agrees_rocket_zero"
    if nonzero == {"rocket_exp40"}:
        return "rocket_exp40", "rocket_only_nonzero"
    return "none", "no_count_agreement"


def choose_fp_guarded(candidates, dataset_name):
    guarded = []
    for name in CANDIDATES:
        row = candidate_row(candidates, name, dataset_name)
        if passes_guard(row):
            budget = operational_budget(row)
            score = min(predicted_count(row), budget) / (budget + 1.0)
            score -= 5.0 * train_exceed_rate(row)
            score -= alert_rate(row)
            if name == "rocket_exp40":
                score += 0.10
            guarded.append((score, -predicted_count(row), name))
    if not guarded:
        return "none", "fp_guard_rejected_all"
    return max(guarded)[2], "best_label_free_fp_guard_score"


def family_prior_maps(candidates, datasets, objective):
    family_datasets = defaultdict(list)
    for dataset_name in datasets:
        family = candidate_row(candidates, "rocket_exp40", dataset_name)["family"]
        family_datasets[family].append(dataset_name)
    priors = {}
    for dataset_name in datasets:
        family = candidate_row(candidates, "rocket_exp40", dataset_name)["family"]
        siblings = [name for name in family_datasets[family] if name != dataset_name]
        if not siblings:
            priors[dataset_name] = ("rocket_exp40", "family_prior_no_sibling_default_rocket")
            continue
        scores = {}
        for candidate_name in CANDIDATES:
            vals = []
            for sibling in siblings:
                row = candidate_row(candidates, candidate_name, sibling)
                if objective == "f1":
                    vals.append(as_float(row["f1"]))
                elif objective == "operational":
                    vals.append(as_float(row["f1"]) - 0.03 * as_float(row["fp"]))
                else:
                    raise ValueError(objective)
            scores[candidate_name] = float(np.mean(vals))
        best = max(scores, key=lambda name: (scores[name], name == "rocket_exp40"))
        priors[dataset_name] = (best, f"leave_one_recipe_family_prior_{objective}")
    return priors


def choose_oracle(candidates, dataset_name):
    best = max(CANDIDATES, key=lambda name: (as_float(candidate_row(candidates, name, dataset_name)["f1"]), -as_float(candidate_row(candidates, name, dataset_name)["fp"])))
    return best, "label_leaking_upper_bound_not_operational"


def materialize_selection(strategy_name, selected_name, reason, candidates, dataset_name):
    reference = candidate_row(candidates, "rocket_exp40", dataset_name)
    if selected_name == "none":
        selected = none_row(reference)
    else:
        selected = dict(candidate_row(candidates, selected_name, dataset_name))
        selected["selected_candidate"] = selected_name
        selected["candidate_config_name"] = selected.get("config_name", "")
        selected["candidate_threshold_method"] = selected.get("threshold_method", "")
        selected["selector_reason"] = reason
    selected.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "selector_name": strategy_name,
            "config_name": strategy_name,
            "threshold_method": "selector",
            "score_family": "selector",
        }
    )
    return selected


def selected_rows(candidates, datasets):
    prior_f1 = family_prior_maps(candidates, datasets, "f1")
    prior_operational = family_prior_maps(candidates, datasets, "operational")
    label_free = label_free_candidates(candidates)
    strategies = {
        "rocket_default": lambda d: ("rocket_exp40", "baseline_default_rocket"),
        "exp55_always": lambda d: ("exp55_best", "baseline_always_exp55"),
        "exp56_always": lambda d: ("exp56_best", "baseline_always_exp56"),
        "label_free_confidence_v1": lambda d: choose_label_free_confidence(label_free, d),
        "agreement_count_v1": lambda d: choose_agreement_count(label_free, d),
        "fp_guarded_v1": lambda d: choose_fp_guarded(label_free, d),
        "family_prior_loorecipe_f1": lambda d: prior_f1[d],
        "family_prior_loorecipe_operational": lambda d: prior_operational[d],
        "oracle_candidate_upper_bound": lambda d: choose_oracle(candidates, d),
    }
    rows = []
    for dataset_name in datasets:
        for strategy_name, chooser in strategies.items():
            selected_name, reason = chooser(dataset_name)
            rows.append(materialize_selection(strategy_name, selected_name, reason, candidates, dataset_name))
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    extra = sorted({key for row in rows for key in row} - set(fieldnames))
    fieldnames.extend(extra)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    out = []
    for strategy_name in sorted({row["selector_name"] for row in rows}):
        subset = [row for row in rows if row["selector_name"] == strategy_name]
        families = defaultdict(list)
        for row in subset:
            families[row["family"]].append(as_float(row["f1"]))
        selected_counts = Counter(row["selected_candidate"] for row in subset)
        validation_mode = selector_validation_mode(strategy_name)
        out.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "selector_name": strategy_name,
                "config_name": strategy_name,
                "threshold_method": "selector",
                "selector_validation_mode": validation_mode,
                "operational_candidate": "0" if validation_mode == "labeled_upper_bound" else "1",
                "num_datasets": len(subset),
                "num_families": len(families),
                "mean_auc_roc": float(np.mean([as_float(row["auc_roc"]) for row in subset])),
                "mean_auc_pr": float(np.mean([as_float(row["auc_pr"]) for row in subset])),
                "mean_f1": float(np.mean([as_float(row["f1"]) for row in subset])),
                "median_f1": float(np.median([as_float(row["f1"]) for row in subset])),
                "p25_f1": float(np.percentile([as_float(row["f1"]) for row in subset], 25)),
                "zero_f1_count": sum(1 for row in subset if as_float(row["f1"]) == 0.0),
                "ge_0_5_count": sum(1 for row in subset if as_float(row["f1"]) >= 0.5),
                "family_macro_f1": float(np.mean([np.mean(vals) for vals in families.values()])),
                "mean_predicted_count": float(np.mean([as_float(row["predicted_count"]) for row in subset])),
                "mean_anomaly_count": float(np.mean([as_float(row["anomaly_count"]) for row in subset])),
                "mean_tp": float(np.mean([as_float(row["tp"]) for row in subset])),
                "mean_fp": float(np.mean([as_float(row["fp"]) for row in subset])),
                "mean_fn": float(np.mean([as_float(row["fn"]) for row in subset])),
                "mean_train_exceed_rate": float(np.mean([as_float(row["train_exceed_rate"]) for row in subset])),
                "mean_oracle_f1": float(np.mean([as_float(row["oracle_f1"]) for row in subset])),
                "selected_rocket_exp40": selected_counts.get("rocket_exp40", 0),
                "selected_exp55_best": selected_counts.get("exp55_best", 0),
                "selected_exp56_best": selected_counts.get("exp56_best", 0),
                "selected_none": selected_counts.get("none", 0),
            }
        )
    return sorted(out, key=lambda row: (int(row["operational_candidate"]), row["mean_f1"], -row["mean_fp"]), reverse=True)


def selector_validation_mode(strategy_name):
    if strategy_name == "oracle_candidate_upper_bound":
        return "labeled_upper_bound"
    if strategy_name.startswith("family_prior_loorecipe"):
        return "historical_labeled_prior"
    if strategy_name in {"label_free_confidence_v1", "agreement_count_v1", "fp_guarded_v1"}:
        return "label_free"
    return "fixed_baseline"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ROCKET + imaging selector strategies.")
    parser.add_argument("--dataset-limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    candidates, datasets = read_candidate_rows()
    if args.dataset_limit:
        datasets = datasets[: args.dataset_limit]
    rows = selected_rows(candidates, datasets)
    write_csv(RESULTS_PATH, rows)
    summary = summarize(rows)
    write_csv(SUMMARY_PATH, summary)
    with LOG_PATH.open("w") as f:
        f.write(f"{EXPERIMENT_ID} finished. datasets={len(datasets)} rows={len(rows)}\n")
        if summary:
            best = summary[0]
            f.write(
                "best={config_name} meanF1={mean_f1:.4f} medianF1={median_f1:.4f} "
                "fp={mean_fp:.2f} zero={zero_f1_count}\n".format(**best)
            )
    print(f"{EXPERIMENT_ID} finished. datasets={len(datasets)} rows={len(rows)}")
    if summary:
        print(summary[0])


if __name__ == "__main__":
    main()
