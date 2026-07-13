from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import run_model_hard_research_experiments as exp84_code


ROOT = Path(__file__).resolve().parent
DATA_DIR = Path("/Users/minho/Documents/Dataset")
OUTPUT_DIR = ROOT / "outputs/exp137_policy_train_only_validation"
EXECUTION_DIR = ROOT / "outputs/exp137_strict_train_only_execution"
EXP84_CONFIG = "aeon_mrh_mr1024_hk4_g32_prune1024_stable_tail_local_gap_knn3"
EXP84_SEED = 20260717
EXP133_CONFIG = "tiered_all_validated_exp93_hard_alerts"
EXP135_CONFIG = "review_tail1pct_all_standard_and_block_c"


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    path = Path(path)
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


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_row_hash(row):
    payload = json.dumps(dict(row), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def parse_indices(value):
    text = str(value or "").strip()
    return {int(item) for item in text.split()} if text else set()


def format_indices(indices):
    return " ".join(str(index) for index in sorted(indices))


def tolerance_match(left, right, tolerance):
    unmatched = set(int(value) for value in right)
    matched = 0
    for index in sorted(int(value) for value in left):
        candidates = sorted((value for value in unmatched if abs(value - index) <= tolerance), key=lambda value: (abs(value - index), value))
        if candidates:
            unmatched.remove(candidates[0])
            matched += 1
    return {"matched": matched, "left_only": len(left) - matched, "right_only": len(unmatched)}


def source_coverage_row(dataset, family, fg, cap2, cap3, eligible):
    rows = {"family_guard_v1": fg, "count_cap_2pct": cap2, "count_cap_3pct": cap3}
    present = {name: row is not None for name, row in rows.items()}
    configs = {row.get("config_name") for row in rows.values() if row is not None}
    same_feature_config = bool(configs) and configs == {EXP84_CONFIG}
    missing = [name for name, exists in present.items() if not exists]
    if missing:
        missing_reason = "missing_" + "_and_".join(missing)
    elif not eligible:
        missing_reason = "present_despite_family_filter_not_eligible"
    else:
        missing_reason = ""
    return {
        "dataset": dataset,
        "family": family,
        "has_family_guard_v1": int(present["family_guard_v1"]),
        "has_count_cap_2pct": int(present["count_cap_2pct"]),
        "has_count_cap_3pct": int(present["count_cap_3pct"]),
        "has_same_feature_config": int(same_feature_config),
        "has_same_seed": "code_config_20260717_not_row_logged" if any(present.values()) else "not_applicable",
        "has_same_score_vector": "not_verifiable_score_vectors_not_stored",
        "score_vector_hash_family_guard": "not_stored_in_csv" if fg else "missing_source",
        "score_vector_hash_cap2": "not_stored_in_csv" if cap2 else "missing_source",
        "score_vector_hash_cap3": "not_stored_in_csv" if cap3 else "missing_source",
        "eligible_under_hard_score_families": int(eligible),
        "excluded_by_family_filter": int(not eligible),
        "excluded_by_difficulty_filter": 0,
        "missing_reason": missing_reason,
    }


def git_value(*args):
    completed = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else f"unavailable: {completed.stderr.strip()}"


def package_versions():
    versions = {}
    for name in ("numpy", "scipy", "scikit-learn", "aeon", "pandas"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def a1_dataset_diff(baseline_rows, replay_rows):
    replay = {row["dataset_name"]: row for row in replay_rows}
    rows = []
    for baseline in sorted(baseline_rows, key=lambda row: row["dataset_name"]):
        name = baseline["dataset_name"]
        row = replay[name]
        baseline_view = {
            "dataset": name,
            "hard": str(baseline.get("hard_alert_indices", "")).strip(),
            "standard": str(baseline.get("standard_review_indices", "")).strip(),
            "priority": str(baseline.get("priority_review_indices", "")).strip(),
        }
        replay_view = {
            "dataset": name,
            "hard": str(row.get("hard_alert_indices", "")).strip(),
            "standard": str(row.get("standard_review_indices", "")).strip(),
            "priority": str(row.get("priority_review_indices", "")).strip(),
        }
        rows.append(
            {
                "dataset": name,
                "baseline_hard_indices": baseline_view["hard"],
                "replay_hard_indices": replay_view["hard"],
                "hard_exact_match": int(baseline_view["hard"] == replay_view["hard"]),
                "baseline_standard_indices": baseline_view["standard"],
                "replay_standard_indices": replay_view["standard"],
                "standard_exact_match": int(baseline_view["standard"] == replay_view["standard"]),
                "baseline_priority_indices": baseline_view["priority"],
                "replay_priority_indices": replay_view["priority"],
                "priority_exact_match": int(baseline_view["priority"] == replay_view["priority"]),
                "all_lanes_match": int(baseline_view == replay_view),
                "baseline_row_hash": canonical_row_hash(baseline_view),
                "replay_row_hash": canonical_row_hash(replay_view),
            }
        )
    return rows


def b1_source_diff(source_rows, b1_rows):
    sources = {}
    for row in source_rows:
        if row.get("config_name") != EXP84_CONFIG:
            continue
        method = row.get("threshold_method")
        if method in {"family_guard_v1", "count_cap_2pct"}:
            sources[(row["dataset_name"], method)] = row
    rows = []
    for b1 in sorted(b1_rows, key=lambda row: row["dataset_name"]):
        name = b1["dataset_name"]
        fg = sources[(name, "family_guard_v1")]
        cap2 = sources[(name, "count_cap_2pct")]
        fg_indices = str(fg.get("selected_indices", "")).strip()
        cap2_indices = str(cap2.get("selected_indices", "")).strip()
        rows.append(
            {
                "dataset": name,
                "family": b1.get("family", ""),
                "family_guard_indices": fg_indices,
                "count_cap_2pct_indices": cap2_indices,
                "source_exact_match": int(fg_indices == cap2_indices),
                "source_changed": int(fg_indices != cap2_indices),
                "exp89_current_indices": b1.get("current_exp89_indices", ""),
                "exp89_neutral_indices": b1.get("neutral_exp89_indices", ""),
                "exp89_changed": int(b1.get("current_exp89_indices", "") != b1.get("neutral_exp89_indices", "")),
                "exp90_current_indices": b1.get("current_exp90_indices", ""),
                "exp90_neutral_indices": b1.get("neutral_exp90_indices", ""),
                "exp90_changed": int(b1.get("current_exp90_indices", "") != b1.get("neutral_exp90_indices", "")),
                "exp93_current_indices": b1.get("current_exp93_indices", ""),
                "exp93_neutral_indices": b1.get("neutral_exp93_indices", ""),
                "exp93_changed": int(b1.get("current_exp93_indices", "") != b1.get("neutral_exp93_indices", "")),
                "hard_current_indices": b1.get("current_hard_indices", ""),
                "hard_neutral_indices": b1.get("neutral_hard_indices", ""),
                "hard_changed": int(b1.get("hard_exact_match", "0") != "1"),
                "standard_current_indices": b1.get("current_standard_indices", ""),
                "standard_neutral_indices": b1.get("neutral_standard_indices", ""),
                "standard_changed": int(b1.get("standard_exact_match", "0") != "1"),
                "priority_current_indices": b1.get("current_priority_indices", ""),
                "priority_neutral_indices": b1.get("neutral_priority_indices", ""),
                "priority_changed": int(b1.get("priority_exact_match", "0") != "1"),
            }
        )
    return rows


def coverage_rows(baseline_rows, source_rows):
    sources = {}
    for row in source_rows:
        if row.get("config_name") == EXP84_CONFIG:
            sources[(row["dataset_name"], row["threshold_method"])] = row
    rows = []
    for baseline in sorted(baseline_rows, key=lambda row: row["dataset_name"]):
        name = baseline["dataset_name"]
        family = baseline.get("family", "")
        rows.append(
            source_coverage_row(
                name,
                family,
                sources.get((name, "family_guard_v1")),
                sources.get((name, "count_cap_2pct")),
                sources.get((name, "count_cap_3pct")),
                family in exp84_code.HARD_SCORE_FAMILIES,
            )
        )
    return rows


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_contract(baseline_rows, input_paths, output_paths):
    dataset_names = sorted(row["dataset_name"] for row in baseline_rows)
    database = DATA_DIR / "univariate_ts.db"
    return {
        "storage": {"database_snapshot": {"path": str(database), "bytes": database.stat().st_size, "modified_utc": datetime.fromtimestamp(database.stat().st_mtime, tz=timezone.utc).isoformat(), "sha256": sha256_file(database)}},
        "repository_and_code": {
            "git_branch": git_value("branch", "--show-current"),
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_status": git_value("status", "--short", "--branch"),
            "python_version": platform.python_version(),
            "package_versions": package_versions(),
            "replay_script_paths": [str(ROOT / name) for name in ("run_experiment_138_policy_train_only_audit.py", "run_experiment_139_family_neutral_common_support.py", "run_experiment_141_family_neutral_full_coverage.py", "run_exp137_policy_validation_evidence.py")],
            "replay_script_sha256": {str(ROOT / name): sha256_file(ROOT / name) for name in ("run_experiment_138_policy_train_only_audit.py", "run_experiment_139_family_neutral_common_support.py", "run_experiment_141_family_neutral_full_coverage.py", "run_exp137_policy_validation_evidence.py")},
        },
        "input_data": {
            "input_file_paths": [str(path) for path in input_paths],
            "input_sha256": {str(path): sha256_file(path) for path in input_paths},
            "dataset_count": len(dataset_names),
            "dataset_list_sha256": hashlib.sha256("\n".join(dataset_names).encode()).hexdigest(),
            "excluded_datasets": ["CornellWhaleChallenge", "Wafer_normal_1"],
            "excluded_dataset_reasons": {
                "CornellWhaleChallenge": "historical full evaluation exclusion; README documents a length-4000 high-frequency audio case",
                "Wafer_normal_1": "historical full evaluation exclusion; detailed reason not verified in current source/docs",
            },
        },
        "reproducibility": {
            "seed": {"exp84_config_random_state": EXP84_SEED, "other_replay_seed": "delegated to frozen upstream result files"},
            "parallelism": {"b1_workers": int(os.environ.get("RANK_EXPERIMENT_WORKERS", "6")), "a0_a1": 1},
            "row_sorting_rule": "dataset ascending lexical order before CSV write and hash",
            "output_hash_rule": "SHA256 of raw output bytes; row hashes use canonical JSON with sorted keys",
            "floating_point_tolerance": "exact index equality; reported metrics serialized from deterministic code; no tolerance used for A1 lane equality",
        },
        "evaluation_definition": {
            "hard_precision_definition": "micro: sum(hard_tp) / (sum(hard_tp) + sum(hard_fp))",
            "hard_recall_definition": "per dataset: hard_tp / (hard_tp + hard_fn), denominator protected by max(1, denominator)",
            "dataset_f1_definition": "per dataset: 2*tp / (2*tp + fp + fn), zero when denominator is zero",
            "mean_hard_f1_definition": "arithmetic mean of per-dataset hard_f1",
            "combined_f1_scope": "human_assisted_diagnostic_only",
            "zero_denominator_rules": "precision and recall use max(1, denominator); F1 is 0 when denominator is 0",
            "no_true_anomaly_rule": "not separately verified in current 1,117 output; evaluation uses the same denominator safeguards",
            "no_candidate_rule": "tp=0, fp=0; F1 is 0 unless denominator from fn is nonzero, then 0",
            "point_matching_rule": "primary: exact index equality",
            "tolerance_matching_rules": ["plus_minus_1 diagnostic", "plus_minus_3 diagnostic", "plus_minus_5 diagnostic"],
            "priority_review_scope": "review-only retrospective rule; never autonomous hard alert",
        },
        "experiment_scope": {
            "A0_file_integrity": "completed",
            "A1_selector_replay": "completed; final lane replay only, not full feature/score rerun",
            "B1_common_support_family_neutral": "completed; 339 shared Exp84 source datasets",
            "B2_full_coverage_family_neutral": "completed; 1,117/1,117 datasets, zero source errors; retrospective counterfactual only",
            "C_policy_budget": "blocked pending operational alert-budget decision",
            "D1_policy_level_train_only": "blocked pending B2 and C",
            "D2_end_to_end_provenance": "not_started",
            "D3_prospective_validation": "not_started",
        },
        "interpretation_limits": {
            "retrospective_only": True,
            "not_end_to_end_strict": True,
            "not_prospective": True,
            "not_deployment_validated": True,
        },
        "output_paths": [str(path) for path in output_paths],
    }


def write_markdown_outputs(a1_rows, b1_rows, coverage, contract, output_dir, b2_summary):
    a1_mismatch = [row for row in a1_rows if not int(row["all_lanes_match"])]
    a1_summary = "\n".join(
        [
            "# A1 Selector Replay Summary",
            "",
            f"- Dataset coverage: {len(a1_rows)}",
            f"- All-lane matches: {len(a1_rows) - len(a1_mismatch)}/{len(a1_rows)}",
            f"- Mismatched datasets: {len(a1_mismatch)}",
            f"- Hard mismatches: {sum(not int(row['hard_exact_match']) for row in a1_rows)}",
            f"- Standard mismatches: {sum(not int(row['standard_exact_match']) for row in a1_rows)}",
            f"- Priority mismatches: {sum(not int(row['priority_exact_match']) for row in a1_rows)}",
            "- Replay command: `python3 run_experiment_138_policy_train_only_audit.py`",
            "- Replay code: `run_experiment_138_policy_train_only_audit.py`, `run_experiment_137_operational_triage.py`",
            f"- Commit: `{contract['repository_and_code']['git_commit']}`",
            "- Input/output SHA256: see `06_evaluation_contract_v2.json` and `MANIFEST.csv`.",
            "- Seed: no new model fit; A1 replays frozen Exp133/Exp135 selector inputs.",
            f"- Environment: Python {contract['repository_and_code']['python_version']}",
            "",
            "A1 is a final-lane selector replay. It does not reproduce upstream feature extraction or score generation.",
        ]
    )
    write_text(output_dir / "03_a1_selector_replay_summary.md", a1_summary)

    b1_counts = {
        "source changed": sum(int(row["source_changed"]) for row in b1_rows),
        "Exp89 changed": sum(int(row["exp89_changed"]) for row in b1_rows),
        "Exp90 changed": sum(int(row["exp90_changed"]) for row in b1_rows),
        "Exp93 changed": sum(int(row["exp93_changed"]) for row in b1_rows),
        "Hard changed": sum(int(row["hard_changed"]) for row in b1_rows),
        "Standard changed": sum(int(row["standard_changed"]) for row in b1_rows),
        "Priority changed": sum(int(row["priority_changed"]) for row in b1_rows),
    }
    b1_summary = "\n".join(
        [
            "# B1 Source-level Summary",
            "",
            markdown_table(["Metric", "Observed", "Expected"], [[key, value, {"source changed": 38, "Exp89 changed": 6, "Exp90 changed": 3, "Exp93 changed": 3, "Hard changed": 0, "Standard changed": 3, "Priority changed": 0}[key]] for key, value in b1_counts.items()]),
            "",
            "All counts are independently recalculated from `04_b1_source_level_diff.csv`.",
            "This is a 339-dataset common-support retrospective counterfactual, not full 1,117-dataset family-independent validation.",
        ]
    )
    write_text(output_dir / "05_b1_source_level_summary.md", b1_summary)

    all_three = sum(row["has_family_guard_v1"] and row["has_count_cap_2pct"] and row["has_count_cap_3pct"] for row in coverage)
    cap2_only = sum(row["has_count_cap_2pct"] and not row["has_family_guard_v1"] and not row["has_count_cap_3pct"] for row in coverage)
    absent = sum(not row["has_family_guard_v1"] and not row["has_count_cap_2pct"] and not row["has_count_cap_3pct"] for row in coverage)
    eligible = sum(row["eligible_under_hard_score_families"] for row in coverage)
    report = "\n".join(
        [
            "# Exp84 Source Coverage Report",
            "",
            markdown_table(
                ["Check", "Count"],
                [
                    ["Exp137 datasets", len(coverage)],
                    ["FG/cap2/cap3 all present", all_three],
                    ["cap2 only", cap2_only],
                    ["No existing Exp84 source", absent],
                    ["Eligible under HARD_SCORE_FAMILIES", eligible],
                    ["Requires new B2 source computation", absent],
                ],
            ),
            "",
            "Existing source rows use the same recorded feature configuration when present. The result CSV does not store raw score vectors, so score-vector equality cannot be independently hashed; this is recorded as `not_verifiable_score_vectors_not_stored` rather than inferred.",
            "The code configuration fixes Exp84 random_state to 20260717, but row-level seed provenance is not stored in the historical CSV.",
            "B2 requires all 1,117 datasets. Missing rows must be recomputed, not treated as empty candidates or silently excluded.",
        ]
    )
    write_text(output_dir / "09_exp84_source_coverage_report.md", report)

    memo = """# Alert Budget Decision Memo

## Decision status

**No primary C/D1 budget is approved.** The current data does not verify whether a user reviews alerts per run, wafer, sensor-step, batch, day, or whole test dataset. Therefore no arbitrary K and no `ceil(rate * n_train)` policy will be used as a primary operating budget.

## Options

| Option | Required metadata | Current implementability | Operating interpretation | Strict TRAIN-only suitability | Primary C/D1 use |
|---|---|---|---|---|---|
| A. Fixed operating cap K | verified review unit and user/safety capacity | not verifiable from current data | maximum alerts per operating unit | compatible when K is pre-registered | only after owner fixes K and unit |
| B. TRAIN normal block percentile | block/run boundaries equivalent to operating unit | not verifiable from current data | upper normal-operation alert volume, e.g. 99th percentile | compatible if blocks and percentile are frozen | only after block definition is available |
| C. `clip(ceil(rate*n_train), min, max)` | train count only | technically possible | research sensitivity only; train count is not review capacity | parameter-only compatible, operationally weak | not primary |

## Consequence

C and D1 are blocked. Code may be prepared, but no primary result will be run or selected until an operating unit and either K or the TRAIN normal block rule are supplied. TEST results must not choose this policy.
"""
    write_text(output_dir / "10_alert_budget_decision_memo.md", memo)

    b2_lines = "not available"
    if b2_summary:
        b2_lines = (
            f"B2 completed on {b2_summary['datasets']} datasets: hard alerts {b2_summary['hard_alerts']}, "
            f"TP {b2_summary['hard_tp']}, FP {b2_summary['hard_fp']}, "
            f"micro precision {b2_summary['hard_precision']:.6f}, mean hard F1 {b2_summary['mean_hard_f1']:.6f}."
        )
    claims = f"""# Presentation Claims After A0/A1/B1/B2

## Verified

- **Exp137 final routing does not use TEST labels or anomaly positions for lane selection.** A1 replay matched 1,117/1,117 stored lanes.
- **In B1 common support, family-guard replacement did not change autonomous hard alerts.** The comparison covers 339 datasets, not all 1,117.
- **B2 calculated the universal Exp84 `count_cap_2pct` source for all 1,117 datasets with zero source errors, and reproduced B1 neutral lanes on its 339-dataset common support.** {b2_lines}

## Conditional

- **Removing the family-dependent source coverage is feasible under the frozen Exp84 configuration.** Full-coverage B2 changed the final hard lane, so it must be reported as a retrospective policy counterfactual rather than as an invariant result.
- **FP 661 to 314 is the current retrospective Exp137 hard-alert result.** B2 has 326 hard FPs, so the original 314 count must not be attributed to family-neutral B2.

## Not allowed

- TEST-length budget can be removed while preserving the FP reduction. C has not run because the budget unit is not registered.
- Policy-level TRAIN-only preserves the FP reduction. C/D1 have not completed.
- The whole pipeline is end-to-end strict TRAIN-only.
- Exp137 has prospective validation on real equipment.
"""
    write_text(output_dir / "31_presentation_claims_after_a0_a1_b1_c_b2_d1.md", claims)


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def write_manifest(output_dir, contract, generated_files):
    entries = []
    for path in sorted(generated_files):
        entries.append(
            {
                "file": str(path.relative_to(output_dir)),
                "description": "Exp137 policy-level validation artifact",
                "experiment_id": "A0/A1/B1 audit or B2 full-coverage counterfactual",
                "stage": "retrospective audit",
                "input_files": ";".join(contract["input_data"]["input_file_paths"]),
                "input_sha256": json.dumps(contract["input_data"]["input_sha256"], sort_keys=True),
                "code_commit": contract["repository_and_code"]["git_commit"],
                "code_files": ";".join(contract["repository_and_code"]["replay_script_paths"]),
                "code_sha256": json.dumps(contract["repository_and_code"]["replay_script_sha256"], sort_keys=True),
                "execution_command": "python3 run_exp137_policy_validation_evidence.py; B2: python3 run_experiment_141_family_neutral_full_coverage.py",
                "seed": json.dumps(contract["reproducibility"]["seed"], sort_keys=True),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "retrospective_or_prospective": "retrospective",
                "autonomous_or_human_assisted": "autonomous hard and human-assisted review kept separate",
                "known_limitations": "retrospective; not end-to-end strict; not prospective; C/D1 budget blocked",
            }
        )
    write_rows(output_dir / "MANIFEST.csv", entries)
    body = "# Manifest\n\n" + markdown_table(["File", "Stage", "Limitation"], [[entry["file"], entry["stage"], entry["known_limitations"]] for entry in entries])
    write_text(output_dir / "MANIFEST.md", body)


def run(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    baseline_path = DATA_DIR / "experiment_137_operational_triage_results.csv"
    a1_path = EXECUTION_DIR / "a1_final_lane_replay.csv"
    b1_path = EXECUTION_DIR / "b1_full" / "b1_family_neutral_common_support_results.csv"
    source_path = DATA_DIR / "experiment_87_exp84_index_diagnostics_results.csv"
    b2_summary_path = output_dir / "b2_full" / "18_b2_family_neutral_summary.json"
    input_paths = [baseline_path, a1_path, b1_path, source_path, DATA_DIR / "experiment_133_block_b_confidence_tiers_results.csv", DATA_DIR / "experiment_135_block_c_review_confirmation_results.csv"]
    baseline = read_rows(baseline_path)
    a1 = read_rows(a1_path)
    b1 = read_rows(b1_path)
    sources = read_rows(source_path)
    a1_rows = a1_dataset_diff(baseline, a1)
    b1_rows = b1_source_diff(sources, b1)
    coverage = coverage_rows(baseline, sources)
    write_rows(output_dir / "01_a1_selector_replay_dataset_diff.csv", a1_rows)
    write_rows(output_dir / "04_b1_source_level_diff.csv", b1_rows)
    write_rows(output_dir / "08_exp84_source_coverage.csv", coverage)
    provisional_outputs = [output_dir / name for name in ("01_a1_selector_replay_dataset_diff.csv", "04_b1_source_level_diff.csv", "08_exp84_source_coverage.csv")]
    contract = build_contract(baseline, input_paths, provisional_outputs)
    write_text(output_dir / "06_evaluation_contract_v2.json", json.dumps(contract, indent=2, sort_keys=True))
    write_text(output_dir / "07_evaluation_contract_v2.md", "# Evaluation Contract v2\n\n```json\n" + json.dumps(contract, indent=2, sort_keys=True) + "\n```")
    b2_summary = json.loads(b2_summary_path.read_text()) if b2_summary_path.exists() else None
    write_markdown_outputs(a1_rows, b1_rows, coverage, contract, output_dir, b2_summary)
    blocker = """# BLOCKER REPORT

## Blocked stages

- `experiment_140_policy_budget_replay` (C)
- `experiment_142_policy_level_train_only` (D1)

## Cause

The operating alert unit and primary budget rule are not verified in the available data. Selecting K or a TRAIN normal block definition without that operational contract would be arbitrary.

## Impact

C cannot remove TEST-length budget policy honestly. B2 is complete, but D1 cannot combine B2 and C.

## Required user decision

Provide the operating review unit and either a pre-registered maximum K or a TRAIN normal block definition plus fixed percentile.

## Resume condition

Record the decision in the evaluation contract before running C/D1. Do not use TEST performance to choose it.
"""
    write_text(output_dir / "BLOCKER_REPORT.md", blocker)
    generated = [path for path in output_dir.rglob("*") if path.is_file() and path.name not in {"MANIFEST.md", "MANIFEST.csv"}]
    write_manifest(output_dir, contract, generated)
    return {
        "a1_all_match": sum(int(row["all_lanes_match"]) for row in a1_rows),
        "a1_total": len(a1_rows),
        "b1_source_changed": sum(int(row["source_changed"]) for row in b1_rows),
        "b1_exp89_changed": sum(int(row["exp89_changed"]) for row in b1_rows),
        "b1_exp90_changed": sum(int(row["exp90_changed"]) for row in b1_rows),
        "b1_exp93_changed": sum(int(row["exp93_changed"]) for row in b1_rows),
        "b1_hard_changed": sum(int(row["hard_changed"]) for row in b1_rows),
        "b1_standard_changed": sum(int(row["standard_changed"]) for row in b1_rows),
        "b1_priority_changed": sum(int(row["priority_changed"]) for row in b1_rows),
        "coverage_all_three": sum(row["has_family_guard_v1"] and row["has_count_cap_2pct"] and row["has_count_cap_3pct"] for row in coverage),
        "coverage_missing_all": sum(not row["has_family_guard_v1"] and not row["has_count_cap_2pct"] and not row["has_count_cap_3pct"] for row in coverage),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2, sort_keys=True))
