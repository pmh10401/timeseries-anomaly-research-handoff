from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


DATA_DIR = Path("/Users/minho/Documents/Dataset")
OUTPUT_DIR = Path("outputs/exp137_strict_train_only_execution")
EXPERIMENT_ID = "experiment_138_policy_train_only_audit"
EXP137_DETAIL = DATA_DIR / "experiment_137_operational_triage_results.csv"
EXP137_SUMMARY = DATA_DIR / "experiment_137_operational_triage_summary.csv"
EXP133_DETAIL = DATA_DIR / "experiment_133_block_b_confidence_tiers_results.csv"
EXP135_DETAIL = DATA_DIR / "experiment_135_block_c_review_confirmation_results.csv"
EXP87_DETAIL = DATA_DIR / "experiment_87_exp84_index_diagnostics_results.csv"
EXP89_DETAIL = DATA_DIR / "experiment_89_74d_with_exp84_candidate_results.csv"
EXP90_DETAIL = DATA_DIR / "experiment_90_zero_f1_repair_selector_results.csv"
EXP133_CONFIG = "tiered_all_validated_exp93_hard_alerts"
EXP135_CONFIG = "review_tail1pct_all_standard_and_block_c"
EXP87_CONFIG = "aeon_mrh_mr1024_hk4_g32_prune1024_stable_tail_local_gap_knn3"
EXP89_SELECTOR = "exp84_four_model_fg_plus_noalert_repair"
EXP90_SELECTOR = "noalert_top1_train_safe_repair"


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_indices(value):
    text = str(value or "").strip()
    return {int(item) for item in text.split()} if text else set()


def format_indices(indices):
    return " ".join(str(index) for index in sorted(indices))


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rebuild_tiers(row133, row135, test_size):
    hard = parse_indices(row133.get("high_confidence_indices"))
    standard_review = parse_indices(row133.get("standard_confidence_indices")) - hard
    priority_review = parse_indices(row135.get("review_candidate_indices")) - hard - standard_review
    all_indices = hard | standard_review | priority_review
    invalid = sorted(index for index in all_indices if index < 0 or index >= int(test_size))
    if invalid:
        raise ValueError(f"out-of-bounds indices: {invalid[:10]}")
    return {
        "hard": hard,
        "standard_review": standard_review,
        "priority_review": priority_review,
        "no_alert": set(range(int(test_size))) - all_indices,
    }


def family_neutral_exp84_source(family_guard_row, cap2_row):
    if not family_guard_row or not cap2_row:
        raise ValueError("family_guard and count_cap_2pct rows are both required")
    return {
        "dataset_name": cap2_row["dataset_name"],
        "selected_indices": cap2_row.get("selected_indices", ""),
        "top_score_indices": cap2_row.get("top_score_indices", ""),
        "source_threshold_method": "count_cap_2pct",
        "policy_uses_family_name": 0,
        "policy_uses_test_length": 0,
        "replaced_threshold_method": family_guard_row.get("threshold_method", ""),
    }


def summarize_rows(rows):
    total = lambda key: int(sum(as_float(row.get(key)) for row in rows))
    mean = lambda key: sum(as_float(row.get(key)) for row in rows) / max(1, len(rows))
    hard_tp = total("hard_tp")
    hard_fp = total("hard_fp")
    standard_tp = total("standard_review_tp")
    standard_fp = total("standard_review_fp")
    priority_tp = total("priority_review_tp")
    priority_fp = total("priority_review_fp")
    return {
        "num_datasets": len(rows),
        "hard_total_alerts": total("hard_alert_count"),
        "hard_total_tp": hard_tp,
        "hard_total_fp": hard_fp,
        "hard_alert_precision": hard_tp / max(1, hard_tp + hard_fp),
        "mean_hard_f1": mean("hard_f1"),
        "mean_f1_scope": "autonomous_hard_alert",
        "standard_review_total_candidates": total("standard_review_count"),
        "standard_review_total_tp": standard_tp,
        "standard_review_total_fp": standard_fp,
        "priority_review_total_candidates": total("priority_review_count"),
        "priority_review_total_tp": priority_tp,
        "priority_review_total_fp": priority_fp,
        "mean_combined_f1": mean("combined_f1"),
        "combined_metric_scope": "human_assisted_diagnostic_only",
    }


def a0_validate(detail_rows, summary_row):
    errors = []
    for row in detail_rows:
        test_size = int(as_float(row.get("test_size")))
        lanes = {
            "hard": parse_indices(row.get("hard_alert_indices")),
            "standard": parse_indices(row.get("standard_review_indices")),
            "priority": parse_indices(row.get("priority_review_indices")),
        }
        if lanes["hard"] & lanes["standard"] or lanes["hard"] & lanes["priority"] or lanes["standard"] & lanes["priority"]:
            errors.append((row["dataset_name"], "lane_overlap"))
        all_indices = lanes["hard"] | lanes["standard"] | lanes["priority"]
        if any(index < 0 or index >= test_size for index in all_indices):
            errors.append((row["dataset_name"], "out_of_bounds"))
        expected = {
            "hard_alert_count": len(lanes["hard"]),
            "standard_review_count": len(lanes["standard"]),
            "priority_review_count": len(lanes["priority"]),
        }
        for key, value in expected.items():
            if int(as_float(row.get(key))) != value:
                errors.append((row["dataset_name"], key))
    observed = summarize_rows(detail_rows)
    expected_keys = [
        "hard_total_alerts",
        "hard_total_tp",
        "hard_total_fp",
        "standard_review_total_candidates",
        "standard_review_total_tp",
        "standard_review_total_fp",
        "priority_review_total_candidates",
        "priority_review_total_tp",
        "priority_review_total_fp",
    ]
    summary_mismatches = {
        key: (observed[key], as_float(summary_row.get(key)))
        for key in expected_keys
        if observed[key] != int(as_float(summary_row.get(key)))
    }
    return observed, errors, summary_mismatches


def a1_replay(detail_rows, exp133_rows, exp135_rows):
    exp133 = {row["dataset_name"]: row for row in exp133_rows if row.get("config_name") == EXP133_CONFIG}
    exp135 = {row["dataset_name"]: row for row in exp135_rows if row.get("config_name") == EXP135_CONFIG}
    replay_rows = []
    for row in detail_rows:
        name = row["dataset_name"]
        tiers = rebuild_tiers(exp133[name], exp135[name], int(as_float(row["test_size"])))
        expected = {
            "hard_alert_indices": format_indices(tiers["hard"]),
            "standard_review_indices": format_indices(tiers["standard_review"]),
            "priority_review_indices": format_indices(tiers["priority_review"]),
        }
        matches = {key: expected[key] == str(row.get(key, "")).strip() for key in expected}
        replay_rows.append(
            {
                "dataset_name": name,
                "test_size": int(as_float(row["test_size"])),
                **expected,
                "hard_match": int(matches["hard_alert_indices"]),
                "standard_match": int(matches["standard_review_indices"]),
                "priority_match": int(matches["priority_review_indices"]),
                "all_match": int(all(matches.values())),
            }
        )
    return replay_rows


def b1_source_audit(exp87_rows, exp89_rows, exp90_rows):
    sources = {}
    for row in exp87_rows:
        if row.get("config_name") != EXP87_CONFIG:
            continue
        if row.get("threshold_method") not in {"family_guard_v1", "count_cap_2pct"}:
            continue
        sources[(row["dataset_name"], row["threshold_method"])] = row
    exp89 = {row["dataset_name"]: row for row in exp89_rows if row.get("selector_name") == EXP89_SELECTOR}
    exp90 = {row["dataset_name"]: row for row in exp90_rows if row.get("selector_name") == EXP90_SELECTOR}
    names = sorted(
        name
        for name in {key[0] for key in sources}
        if (name, "family_guard_v1") in sources and (name, "count_cap_2pct") in sources
    )
    rows = []
    for name in names:
        fg = sources[(name, "family_guard_v1")]
        cap2 = sources[(name, "count_cap_2pct")]
        neutral = family_neutral_exp84_source(fg, cap2)
        fg_indices = parse_indices(fg.get("selected_indices"))
        neutral_indices = parse_indices(neutral["selected_indices"])
        union = fg_indices | neutral_indices
        rows.append(
            {
                "dataset_name": name,
                "family": fg.get("family", ""),
                "fg_selected_indices": format_indices(fg_indices),
                "neutral_selected_indices": format_indices(neutral_indices),
                "fg_count": len(fg_indices),
                "neutral_count": len(neutral_indices),
                "exact_match": int(fg_indices == neutral_indices),
                "jaccard": len(fg_indices & neutral_indices) / len(union) if union else 1.0,
                "source_threshold_method": neutral["source_threshold_method"],
                "policy_uses_family_name": neutral["policy_uses_family_name"],
                "policy_uses_test_length": neutral["policy_uses_test_length"],
                "exp89_used_exp84_candidate": int(as_float(exp89.get(name, {}).get("used_exp84_candidate"))),
                "exp89_exp84_only_promoted_count": int(as_float(exp89.get(name, {}).get("exp84_only_promoted_count"))),
                "exp90_repair_source": exp90.get(name, {}).get("repair_source", ""),
                "exp90_repair_added_count": int(as_float(exp90.get(name, {}).get("repair_added_count"))),
            }
        )
    return rows


def run(output_dir=OUTPUT_DIR):
    detail_rows = read_rows(EXP137_DETAIL)
    summary_rows = read_rows(EXP137_SUMMARY)
    if len(summary_rows) != 1:
        raise ValueError(f"expected one Exp137 summary row, got {len(summary_rows)}")
    observed, errors, summary_mismatches = a0_validate(detail_rows, summary_rows[0])
    replay_rows = a1_replay(detail_rows, read_rows(EXP133_DETAIL), read_rows(EXP135_DETAIL))
    b1_rows = b1_source_audit(read_rows(EXP87_DETAIL), read_rows(EXP89_DETAIL), read_rows(EXP90_DETAIL))
    contract = {
        "experiment_id": EXPERIMENT_ID,
        "scope": "A0 file integrity, A1 final-lane replay, B1 common-support source audit",
        "retrospective_only": True,
        "autonomous_metric_scope": "autonomous_hard_alert",
        "human_assisted_metric_scope": "human_assisted_diagnostic_only",
        "input_sha256": {str(path): file_sha256(path) for path in [EXP137_DETAIL, EXP137_SUMMARY, EXP133_DETAIL, EXP135_DETAIL, EXP87_DETAIL, EXP89_DETAIL, EXP90_DETAIL]},
        "a0_summary": observed,
        "a0_lane_errors": errors,
        "a0_summary_mismatches": summary_mismatches,
        "a1_all_match_datasets": sum(row["all_match"] for row in replay_rows),
        "a1_total_datasets": len(replay_rows),
        "b1_common_support_datasets": len(b1_rows),
        "b1_changed_source_datasets": sum(not row["exact_match"] for row in b1_rows),
        "next_gate": "C and D1 require a pre-registered operational alert-budget unit; no budget policy was selected here.",
    }
    output_dir = Path(output_dir)
    write_rows(output_dir / "a1_final_lane_replay.csv", replay_rows)
    write_rows(output_dir / "b1_common_support_source_audit.csv", b1_rows)
    (output_dir / "evaluation_contract_a0_a1_b1.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    if errors or summary_mismatches or contract["a1_all_match_datasets"] != len(replay_rows):
        raise SystemExit("A0/A1 validation failed; see output contract")
    return contract


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    result = run(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
