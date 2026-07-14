"""Exp151/152: corrected virtual-run conformal policy experiment.

Why a new experiment number?
----------------------------
Exp149/150 produced useful review-candidate sensitivity data, but their Hard
and Priority lanes were invalid because the code accidentally read Exp93 and
Exp131 rows through ``exp133.load_maps()`` as though they were Exp133 and
Exp135 output rows. Missing columns silently became empty index sets, so every
candidate was routed to Standard review.

V2 intentionally does not reuse old Exp133/Exp135 final indices. Instead it:

1. builds conformal candidates from TRAIN-normal out-of-fold scores;
2. recomputes the independent Block-B score and TRAIN-normal threshold for the
   *new* candidate universe;
3. recomputes Block-C auxiliary evidence for the new Standard-review universe;
4. writes label-free predictions and freezes their SHA-256 hash;
5. loads TEST labels only in a separate evaluation phase.

The local database path is deliberately unchanged:
``/Users/minho/Documents/Dataset/univariate_ts.db``.

The results remain retrospective counterfactual virtual-equipment benchmark
results. No alpha is selected from TEST performance, and Priority review is
human-assisted research evidence rather than an autonomous alert.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np

# The file lives below experiments/exp151_152. Add the repository root before
# importing the historical research modules preserved at repository root.
HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
for import_path in (HERE, REPO_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import run_experiment_131_rocket_block_b_calibration as exp131
import run_experiment_135_block_c_review_confirmation as exp135
import run_experiment_40_original_score_normalization_sweep as exp40
import run_experiment_60_62_rocket_imaging_selector_variants as selector
import run_model_hard_research_experiments as model_hard
from run_balanced_improvement_experiment import density_knn_score_pair
from run_experiment_29_train_normal_threshold_calibration import (
    train_false_positive_stats,
)
from run_experiment_40_original_score_normalization_sweep import count_cap_threshold
from run_original_improvement_experiment import parse_family
from run_rank_ensemble_calibration import align_series_lengths, sanitize_series, z_normalize

import virtual_run_policy_core as core


DATA_DIR = Path("/Users/minho/Documents/Dataset")
DB_PATH = DATA_DIR / "univariate_ts.db"
BASELINE_PATH = DATA_DIR / "experiment_137_operational_triage_results.csv"
EXP87_PATH = DATA_DIR / "experiment_87_exp84_index_diagnostics_results.csv"
EXTERNAL_LIVE_PATH = DATA_DIR / "rank_dashboard_external_live_run.json"

DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/exp151_152_virtual_run_conformal_v2"
CURRENT_EXPERIMENT_ID = "experiment_151_virtual_run_conformal_policy_v2"
B2_EXPERIMENT_ID = "experiment_152_virtual_run_conformal_policy_v2"
CURRENT_VARIANT = "exp151_current_source_v2"
B2_VARIANT = "exp152_b2_source_v2"
WORKERS = int(os.environ.get("RANK_EXPERIMENT_WORKERS", "7"))

BLOCK_B_RATE = 0.015
BLOCK_B_NEIGHBORS = 3
BLOCK_C_RATE = 0.01

SOURCE_CONFIGS = {
    "rocket_exp40": {
        "kind": "rocket",
        "config": next(
            config
            for config in exp40.CONFIGS
            if config["name"] == "rocket_256_knn3_local_gap"
        ),
    },
    "exp55_best": {
        "kind": "imaging",
        "config": selector.IMAGING_CONFIGS["exp55_best"],
    },
    "exp56_best": {
        "kind": "imaging",
        "config": selector.IMAGING_CONFIGS["exp56_best"],
    },
    "exp84": {
        "kind": "aeon",
        "config": next(
            config
            for config in model_hard.EXPERIMENT_SPECS[
                "experiment_87_exp84_index_diagnostics"
            ]["configs"]
            if config["name"] == core.EXPECTED_B2_CONFIG_NAME
        ),
    },
}


def read_rows(path: Path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
    return rows


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unavailable"


def resolve_b2_manifest_path() -> Path:
    candidates = [
        REPO_ROOT
        / "outputs/exp137_policy_train_only_validation/b2_full/16_b2_full_coverage_source_manifest.csv",
        REPO_ROOT
        / "results/exp137_policy_train_only_validation/b2_full/16_b2_full_coverage_source_manifest.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "B2 manifest not found. Checked: " + ", ".join(str(path) for path in candidates)
    )


def evaluation_dataset_names(dataset_limit: int | None = None):
    names = sorted(row["dataset_name"] for row in read_rows(BASELINE_PATH))
    if len(names) != len(set(names)):
        raise ValueError("duplicate dataset names in Exp137 baseline")
    if dataset_limit:
        names = names[: int(dataset_limit)]
    return names


def _load_blob_rows(conn, dataset_name: str, split: str):
    return conn.execute(
        """
        SELECT i.values_blob
        FROM instances i
        JOIN datasets d ON i.dataset_id = d.id
        WHERE d.name = ? AND i.split = ?
        ORDER BY i.instance_index
        """,
        (dataset_name, split),
    ).fetchall()


def load_dataset_series_only(dataset_name: str, db_path: Path = DB_PATH):
    """Load TRAIN/TEST values without reading TEST labels.

    This is the key structural correction for prediction/evaluation separation.
    The historical ``load_dataset_data`` and ``load_original_record`` helpers
    load TEST labels as part of their return value, so V2 does not call them in
    prediction. Target length is derived from TRAIN-normal series only; TEST
    batch size and TEST series-length statistics do not set model parameters.
    """

    conn = sqlite3.connect(str(db_path))
    try:
        meta = conn.execute(
            "SELECT series_length FROM datasets WHERE name = ?", (dataset_name,)
        ).fetchone()
        if meta is None:
            raise KeyError(f"dataset not found: {dataset_name}")
        metadata_len = int(meta[0]) if meta[0] else 0
        train_rows = _load_blob_rows(conn, dataset_name, "TRAIN")
        test_rows = _load_blob_rows(conn, dataset_name, "TEST")
    finally:
        conn.close()

    train_series = [
        sanitize_series(np.frombuffer(row[0], dtype=np.float32)) for row in train_rows
    ]
    test_series = [
        sanitize_series(np.frombuffer(row[0], dtype=np.float32)) for row in test_rows
    ]
    if not train_series:
        raise ValueError(f"no TRAIN instances: {dataset_name}")
    if not test_series:
        raise ValueError(f"no TEST instances: {dataset_name}")

    train_lengths = [len(series) for series in train_series]
    train_target_len = int(round(float(np.median(train_lengths))))
    if train_target_len <= 0:
        train_target_len = int(metadata_len or 16)
    train_target_len = min(max(8, train_target_len), 2048)

    x_train = align_series_lengths(train_series, train_target_len)
    x_test = align_series_lengths(test_series, train_target_len)

    # Some frozen feature helpers expect a historical record dictionary. Every
    # length field below is deliberately TRAIN-derived. The neutral family and
    # dataset aliases prevent hidden family/name routing inside Exp152 scoring.
    scoring_record = {
        "dataset_name": "__virtual_run_dataset__",
        "family": "__family_neutral__",
        "metadata_len": metadata_len,
        "actual_len_median": float(train_target_len),
        "actual_len_max": float(max(train_lengths)),
        "test_actual_len_median": float(train_target_len),
        "train_series": train_series,
        "test_series": test_series,
        "train_target_len": train_target_len,
    }
    return x_train.astype(np.float32), x_test.astype(np.float32), scoring_record


def load_test_labels(dataset_name: str, db_path: Path = DB_PATH) -> np.ndarray:
    """Load TEST labels for the evaluation phase only."""

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            """
            SELECT i.label
            FROM instances i
            JOIN datasets d ON i.dataset_id = d.id
            WHERE d.name = ? AND i.split = 'TEST'
            ORDER BY i.instance_index
            """,
            (dataset_name,),
        ).fetchall()
    finally:
        conn.close()
    try:
        return np.asarray([int(row[0]) for row in rows], dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"non-binary instance labels cannot be evaluated: {dataset_name}"
        ) from exc


def source_score_pair(source_name, x_fit, x_query, x_test, record):
    """Fit one frozen source on TRAIN-normal data and score query/TEST runs."""

    target_len = int(record["train_target_len"])
    if source_name == "rocket_exp40":
        fit_z = z_normalize(x_fit).astype(np.float32)
        query_z = z_normalize(x_query).astype(np.float32)
        test_z = z_normalize(x_test).astype(np.float32)
        config = SOURCE_CONFIGS[source_name]["config"]
        fit_features, query_features = exp40.rocket_feature_pair(
            fit_z, query_z, fit_z.shape[1], config["num_kernels"]
        )
        _, test_features = exp40.rocket_feature_pair(
            fit_z, test_z, fit_z.shape[1], config["num_kernels"]
        )
        fit_for_query, query_features = model_hard.scale_feature_pair(
            fit_features, query_features
        )
        fit_for_test, test_features = model_hard.scale_feature_pair(
            fit_features, test_features
        )
        query_scores = exp40.density_knn_score_pair(
            fit_for_query, query_features, 3, "local_gap"
        )[1]
        test_scores = exp40.density_knn_score_pair(
            fit_for_test, test_features, 3, "local_gap"
        )[1]
        fit_scores = exp40.density_knn_score_pair(
            fit_for_query, fit_for_query, 3, "local_gap"
        )[0]
        return fit_scores, query_scores, test_scores

    if source_name in {"exp55_best", "exp56_best"}:
        config = SOURCE_CONFIGS[source_name]["config"]
        fit_raw = align_series_lengths([row for row in x_fit], target_len)
        query_raw = align_series_lengths([row for row in x_query], target_len)
        test_raw = align_series_lengths([row for row in x_test], target_len)
        fit_z = z_normalize(fit_raw).astype(np.float32)
        query_z = z_normalize(query_raw).astype(np.float32)
        test_z = z_normalize(test_raw).astype(np.float32)
        fit_pre, query_pre = model_hard.prepare_series_pair_for_scale(
            config.get("series_scale", "per_series_z"),
            fit_raw,
            query_raw,
            fit_z,
            query_z,
        )
        fit_pre_for_test, test_pre = model_hard.prepare_series_pair_for_scale(
            config.get("series_scale", "per_series_z"),
            fit_raw,
            test_raw,
            fit_z,
            test_z,
        )
        fit_scores, query_scores = model_hard.score_pair_for_config(
            fit_pre, query_pre, target_len, config, dict(record), {}
        )
        _, test_scores = model_hard.score_pair_for_config(
            fit_pre_for_test, test_pre, target_len, config, dict(record), {}
        )
        return fit_scores, query_scores, test_scores

    if source_name == "exp84":
        fit_z = z_normalize(x_fit).astype(np.float32)
        query_z = z_normalize(x_query).astype(np.float32)
        test_z = z_normalize(x_test).astype(np.float32)
        config = SOURCE_CONFIGS[source_name]["config"]
        fit_scores, query_scores = model_hard.score_pair_for_config(
            fit_z, query_z, fit_z.shape[1], config, dict(record), {}
        )
        _, test_scores = model_hard.score_pair_for_config(
            fit_z, test_z, fit_z.shape[1], config, dict(record), {}
        )
        return np.asarray(fit_scores), np.asarray(query_scores), np.asarray(test_scores)

    raise ValueError(f"unknown source: {source_name}")


def crossfit_source(source_name, x_train, x_test, record):
    n_train = len(x_train)
    if n_train < 5:
        return {
            "train_oof_scores": np.asarray([], dtype=float),
            "test_scores": np.asarray([], dtype=float),
            "method": "unsupported_n_train_lt5",
            "minimum_attainable_p": 1.0,
        }

    oof = np.full(n_train, np.nan, dtype=float)
    for fit_indices, query_indices in core.fold_plan(n_train, seed=core.SEED):
        _, query_scores, _ = source_score_pair(
            source_name,
            x_train[fit_indices],
            x_train[query_indices],
            x_test,
            record,
        )
        oof[query_indices] = np.asarray(query_scores, dtype=float)

    # The full-fit TEST scores are computed only after all OOF TRAIN-normal
    # scores exist. TEST labels are not loaded in this phase.
    _, _, test_scores = source_score_pair(
        source_name, x_train, x_test, x_test, record
    )
    valid = oof[np.isfinite(oof)]
    if len(valid) != n_train:
        raise ValueError(
            f"incomplete OOF scores source={source_name}: {len(valid)}/{n_train}"
        )
    return {
        "train_oof_scores": valid,
        "test_scores": np.asarray(test_scores, dtype=float),
        "method": "deterministic_5fold" if n_train >= 20 else "leave_one_out",
        "minimum_attainable_p": 1.0 / (len(valid) + 1),
    }


def recompute_independent_evidence(x_train, x_test):
    """Recompute Block-B and Block-C for the new conformal candidate universe.

    Exp149/150 incorrectly tried to read final historical lane indices. V2 uses
    the original TRAIN-normal scoring functions directly. Block-B uses the
    second independent ROCKET kernel block with the frozen 1.5% count-cap rate;
    Block-C uses the third block with the frozen 1.0% rate.
    """

    x_train_z = z_normalize(x_train).astype(np.float32)
    x_test_z = z_normalize(x_test).astype(np.float32)

    reference, seed_offset, train_features, test_features = exp131.score_block_b(
        x_train_z, x_test_z
    )
    block_b_train_scores, block_b_test_scores = density_knn_score_pair(
        train_features,
        test_features,
        BLOCK_B_NEIGHBORS,
        "local_gap",
    )
    block_b_threshold, block_b_q, block_b_cap = count_cap_threshold(
        block_b_train_scores, BLOCK_B_RATE
    )
    block_b_indices = set(
        np.flatnonzero(block_b_test_scores > block_b_threshold)
        .astype(int)
        .tolist()
    )
    block_b_exceed_count, block_b_exceed_rate = train_false_positive_stats(
        block_b_train_scores, block_b_threshold
    )

    (
        block_c_indices,
        block_c_test_scores,
        block_c_reference_count,
        block_c_seed_offset,
        block_c_q,
        block_c_cap,
    ) = exp135.block_c_candidates(x_train_z, x_test_z)

    return {
        "block_b_indices": block_b_indices,
        "block_b_threshold": float(block_b_threshold),
        "block_b_q_effective": float(block_b_q),
        "block_b_cap_target": int(block_b_cap),
        "block_b_reference_count": len(reference),
        "block_b_reference_seed_offset": int(seed_offset),
        "block_b_train_exceed_count": int(block_b_exceed_count),
        "block_b_train_exceed_rate": float(block_b_exceed_rate),
        "block_b_test_score_max": float(np.max(block_b_test_scores))
        if len(block_b_test_scores)
        else None,
        "block_c_indices": set(block_c_indices),
        "block_c_q_effective": float(block_c_q),
        "block_c_cap_target": int(block_c_cap),
        "block_c_reference_count": int(block_c_reference_count),
        "block_c_reference_seed_offset": int(block_c_seed_offset),
        "block_c_test_score_max": float(np.max(block_c_test_scores))
        if len(block_c_test_scores)
        else None,
    }


def load_current_exp84_coverage():
    rows = read_rows(EXP87_PATH)
    # Exp87 contains diagnostic rows from more than one configuration. Exp151
    # preserves only the historical coverage of the exact frozen Exp84 source
    # used by Exp89/B2; otherwise unrelated family_guard rows could silently
    # make Exp84 available to additional datasets.
    return {
        row["dataset_name"]
        for row in rows
        if row.get("config_name") == core.EXPECTED_B2_CONFIG_NAME
        and row.get("threshold_method") == "family_guard_v1"
    }


def load_and_validate_b2_contract(expected_names):
    path = resolve_b2_manifest_path()
    expected = set(expected_names)
    # Full B2 manifests contain all 1,117 datasets. Smoke runs intentionally
    # validate only their frozen subset, so filter before enforcing exact
    # coverage. The full run still checks all 1,117 names exactly.
    rows = [
        row for row in read_rows(path)
        if row.get("dataset_name") in expected
    ]
    return path, core.validate_b2_manifest(rows, expected)


def _format_indices(indices) -> str:
    return " ".join(str(int(index)) for index in sorted(set(indices)))


def predict_dataset(
    dataset_name: str,
    current_exp84_coverage: set[str],
    b2_contract_names: set[str],
):
    """Return label-free prediction rows for one dataset.

    ``y_test`` is intentionally absent from the signature. This function can
    only access TRAIN/TEST values, frozen source configuration and TRAIN-normal
    calibration results.
    """

    x_train, x_test, record = load_dataset_series_only(dataset_name)
    n_train, n_test = len(x_train), len(x_test)
    calibration = core.calibration_status(n_train)

    if dataset_name not in b2_contract_names:
        raise ValueError(f"dataset missing from validated B2 contract: {dataset_name}")

    source_scores = {}
    evidence = {
        "block_b_indices": set(),
        "block_c_indices": set(),
        "block_b_threshold": None,
        "block_b_q_effective": None,
        "block_b_cap_target": None,
        "block_b_reference_count": 0,
        "block_b_reference_seed_offset": 0,
        "block_b_train_exceed_count": 0,
        "block_b_train_exceed_rate": 0.0,
        "block_b_test_score_max": None,
        "block_c_q_effective": None,
        "block_c_cap_target": None,
        "block_c_reference_count": 0,
        "block_c_reference_seed_offset": 0,
        "block_c_test_score_max": None,
    }
    if not calibration["abstain"]:
        for source_name in SOURCE_CONFIGS:
            source_scores[source_name] = crossfit_source(
                source_name, x_train, x_test, record
            )
        evidence = recompute_independent_evidence(x_train, x_test)

    variant_specs = (
        (
            CURRENT_VARIANT,
            CURRENT_EXPERIMENT_ID,
            ["rocket_exp40", "exp55_best", "exp56_best"]
            + (["exp84"] if dataset_name in current_exp84_coverage else []),
        ),
        (
            B2_VARIANT,
            B2_EXPERIMENT_ID,
            ["rocket_exp40", "exp55_best", "exp56_best", "exp84"],
        ),
    )

    rows = []
    for variant, experiment_id, available_sources in variant_specs:
        for alpha in core.ALPHAS:
            if calibration["abstain"]:
                p_values = {}
                candidates = set()
                agreement_count = np.zeros(n_test, dtype=int)
                minimum_p = {source: 1.0 for source in available_sources}
            else:
                p_values = {
                    source: core.conformal_p_values(
                        source_scores[source]["train_oof_scores"],
                        source_scores[source]["test_scores"],
                    )
                    for source in available_sources
                }
                candidates, agreement_count = core.agreement_indices(
                    p_values,
                    alpha,
                    required_sources=core.REQUIRED_SOURCE_AGREEMENT,
                )
                minimum_p = {
                    source: source_scores[source]["minimum_attainable_p"]
                    for source in available_sources
                }

            resolution = core.resolution_status(
                minimum_p,
                alpha,
                required_sources=core.REQUIRED_SOURCE_AGREEMENT,
            )
            lanes = core.route_lanes_from_recomputed_evidence(
                candidates,
                evidence["block_b_indices"],
                evidence["block_c_indices"],
                n_test,
            )

            row = {
                "experiment_id": experiment_id,
                "dataset_name": dataset_name,
                "family": parse_family(dataset_name),
                "variant": variant,
                "alpha": alpha,
                "n_train": n_train,
                "n_test": n_test,
                "train_target_len": int(record["train_target_len"]),
                "calibration_status": calibration["status"],
                "abstained": int(calibration["abstain"]),
                "crossfit_method": (
                    "unsupported_n_train_lt5"
                    if calibration["abstain"]
                    else source_scores["rocket_exp40"]["method"]
                ),
                "available_sources": " ".join(available_sources),
                "source_count": len(available_sources),
                "required_source_agreement": core.REQUIRED_SOURCE_AGREEMENT,
                "candidate_indices": _format_indices(candidates),
                "candidate_count": len(candidates),
                "agreement_count_distribution": json.dumps(
                    {
                        str(count): int(np.sum(agreement_count == count))
                        for count in range(len(available_sources) + 1)
                    },
                    sort_keys=True,
                ),
                "block_b_indices": _format_indices(evidence["block_b_indices"]),
                "block_b_count": len(evidence["block_b_indices"]),
                "block_c_indices": _format_indices(evidence["block_c_indices"]),
                "block_c_count": len(evidence["block_c_indices"]),
                "hard_indices": _format_indices(lanes["hard"]),
                "standard_review_indices": _format_indices(
                    lanes["standard_review"]
                ),
                "priority_review_indices": _format_indices(
                    lanes["priority_review"]
                ),
                "hard_count": len(lanes["hard"]),
                "standard_review_count": len(lanes["standard_review"]),
                "priority_review_count": len(lanes["priority_review"]),
                "no_alert_count": len(lanes["no_alert"]),
                "minimum_attainable_p_by_source": json.dumps(
                    minimum_p, sort_keys=True
                ),
                "resolution_eligible_sources": " ".join(
                    resolution["eligible_sources"]
                ),
                "resolution_eligible_source_count": resolution[
                    "eligible_source_count"
                ],
                "resolution_blocks_two_source_agreement": int(
                    resolution["blocks_required_agreement"]
                ),
                "routing_uses_test_labels": 0,
                "routing_uses_test_positions": 0,
                "routing_uses_test_length_for_policy": 0,
                "retrospective_counterfactual": 1,
                "prospective_validated": 0,
                "priority_review_posthoc_research_rule": 1,
                "mean_combined_f1_scope": "human_assisted_diagnostic_only",
                **{
                    key: value
                    for key, value in evidence.items()
                    if key not in {"block_b_indices", "block_c_indices"}
                },
            }
            for source in available_sources:
                row[f"{source}_p_min"] = (
                    float(np.min(p_values[source])) if source in p_values else None
                )
            rows.append(row)
    return rows


def grain_audit(names):
    conn = sqlite3.connect(str(DB_PATH))
    try:
        columns = [
            item[1] for item in conn.execute("PRAGMA table_info(datasets)").fetchall()
        ]
        rows = []
        for name in names:
            dataset_row = conn.execute(
                "SELECT * FROM datasets WHERE name = ?", (name,)
            ).fetchone()
            if dataset_row is None:
                raise KeyError(f"dataset not found: {name}")
            metadata = dict(zip(columns, dataset_row))
            train_labels = [
                row[0]
                for row in conn.execute(
                    """
                    SELECT i.label
                    FROM instances i JOIN datasets d ON i.dataset_id=d.id
                    WHERE d.name=? AND i.split='TRAIN'
                    ORDER BY i.instance_index
                    """,
                    (name,),
                )
            ]
            test_labels = [
                row[0]
                for row in conn.execute(
                    """
                    SELECT i.label
                    FROM instances i JOIN datasets d ON i.dataset_id=d.id
                    WHERE d.name=? AND i.split='TEST'
                    ORDER BY i.instance_index
                    """,
                    (name,),
                )
            ]
            compatible = set(train_labels + test_labels) <= {"0", "1"}
            rows.append(
                {
                    "dataset": name,
                    "train_instance_count": len(train_labels),
                    "test_instance_count": len(test_labels),
                    "train_label_values": " ".join(sorted(set(train_labels))),
                    "test_label_values": " ".join(sorted(set(test_labels))),
                    "values_dtype": "float32",
                    "labels_blob_presence": "not_used_for_instance_policy",
                    "instance_level_binary_label_compatible": int(compatible),
                    "minimum_attainable_conformal_p": 1.0
                    / (len(train_labels) + 1)
                    if train_labels
                    else 1.0,
                    "abstained_n_train_lt5": int(len(train_labels) < 5),
                    "dataset_train_normal_count_metadata": metadata.get(
                        "train_normal_count", ""
                    ),
                    "dataset_test_total_count_metadata": metadata.get(
                        "test_total_count", ""
                    ),
                }
            )
    finally:
        conn.close()
    return rows


def _contract(output_dir, names, b2_path, skip_db_hash=False):
    script_path = Path(__file__).resolve()
    core_path = script_path.with_name("virtual_run_policy_core.py")
    return {
        "experiment_ids": [CURRENT_EXPERIMENT_ID, B2_EXPERIMENT_ID],
        "scope": "retrospective counterfactual virtual wafer-run benchmark",
        "database_path": str(DB_PATH),
        "database_sha256": "skipped_for_smoke_test"
        if skip_db_hash
        else core.sha256_file(DB_PATH),
        "baseline_path": str(BASELINE_PATH),
        "baseline_sha256": core.sha256_file(BASELINE_PATH),
        "exp87_path": str(EXP87_PATH),
        "exp87_sha256": core.sha256_file(EXP87_PATH),
        "b2_manifest_path": str(b2_path),
        "b2_manifest_sha256": core.sha256_file(b2_path),
        "script_sha256": core.sha256_file(script_path),
        "core_sha256": core.sha256_file(core_path),
        "git_commit": git_commit(),
        "seed": core.SEED,
        "alphas": list(core.ALPHAS),
        "alpha_selection_rule": "sensitivity_only_no_alpha_selected",
        "required_source_agreement": core.REQUIRED_SOURCE_AGREEMENT,
        "block_b_rate": BLOCK_B_RATE,
        "block_b_neighbors": BLOCK_B_NEIGHBORS,
        "block_c_rate": BLOCK_C_RATE,
        "dataset_count": len(names),
        "dataset_list_sha256": hashlib.sha256(
            "\n".join(names).encode("utf-8")
        ).hexdigest(),
        "prediction_label_access": "forbidden",
        "evaluation_requires_frozen_prediction_hash": True,
        "hard_metric_scope": "autonomous_hard_alert_only",
        "review_metric_scope": "human_assisted_diagnostic_only",
        "prospective_validated": False,
        "end_to_end_configuration_provenance_verified": False,
        "output_dir": str(output_dir),
    }


def verify_contract_inputs(contract) -> None:
    """Ensure code and frozen inputs did not change after the audit phase."""

    checks = {
        "baseline_sha256": BASELINE_PATH,
        "exp87_sha256": EXP87_PATH,
        "b2_manifest_sha256": Path(contract["b2_manifest_path"]),
        "script_sha256": Path(__file__).resolve(),
        "core_sha256": Path(__file__).resolve().with_name("virtual_run_policy_core.py"),
    }
    for key, path in checks.items():
        actual = core.sha256_file(path)
        expected = contract[key]
        if actual != expected:
            raise ValueError(
                f"frozen input changed after audit: {key} expected={expected} "
                f"actual={actual} path={path}"
            )
    expected_db = contract.get("database_sha256")
    if expected_db and expected_db != "skipped_for_smoke_test":
        actual_db = core.sha256_file(DB_PATH)
        if actual_db != expected_db:
            raise ValueError(
                "database changed after audit: "
                f"expected={expected_db} actual={actual_db} path={DB_PATH}"
            )


def run_audit_phase(
    output_dir: Path,
    dataset_limit: int | None = None,
    skip_db_hash: bool = False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = evaluation_dataset_names(dataset_limit)
    b2_path, _ = load_and_validate_b2_contract(names)
    audit_rows = grain_audit(names)
    write_rows(output_dir / "02_virtual_run_grain_audit.csv", audit_rows)

    incompatible = [
        row["dataset"]
        for row in audit_rows
        if not int(row["instance_level_binary_label_compatible"])
    ]
    if incompatible:
        blocker = (
            "# Exp151/152 blocker\n\n"
            "Instance-level binary label compatibility failed for: "
            + ", ".join(incompatible[:20])
            + "\n"
        )
        (output_dir / "BLOCKER_REPORT_V2.md").write_text(
            blocker, encoding="utf-8"
        )
        raise SystemExit("non-binary instance labels found; audit blocked")

    contract = _contract(output_dir, names, b2_path, skip_db_hash=skip_db_hash)
    contract["n_train_lt5_datasets"] = int(
        sum(int(row["abstained_n_train_lt5"]) for row in audit_rows)
    )
    write_json(output_dir / "01_evaluation_contract.json", contract)
    return contract


def _load_completed_checkpoint(path: Path):
    completed = {}
    if not path.exists():
        return completed
    for payload in read_jsonl(path):
        completed[payload["dataset_name"]] = payload["rows"]
    return completed


def _append_checkpoint(path: Path, dataset_name: str, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"dataset_name": dataset_name, "rows": rows},
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )


def run_prediction_phase(
    output_dir: Path,
    dataset_limit: int | None = None,
    workers: int = WORKERS,
    reset_checkpoint: bool = False,
):
    """Generate and freeze label-free prediction artifacts."""

    output_dir = Path(output_dir)
    contract_path = output_dir / "01_evaluation_contract.json"
    if not contract_path.exists():
        raise FileNotFoundError("run --phase audit before prediction")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    verify_contract_inputs(contract)
    names = evaluation_dataset_names(dataset_limit)
    expected_list_hash = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
    if contract["dataset_list_sha256"] != expected_list_hash:
        raise ValueError("dataset list differs from frozen audit contract")

    current_coverage = load_current_exp84_coverage()
    _, b2_contract = load_and_validate_b2_contract(names)
    b2_names = set(b2_contract)

    checkpoint = output_dir / "prediction_checkpoint_v2.jsonl"
    if reset_checkpoint and checkpoint.exists():
        checkpoint.unlink()
    completed = _load_completed_checkpoint(checkpoint)
    pending = [name for name in names if name not in completed]
    errors = []

    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        futures = {
            executor.submit(
                predict_dataset, name, current_coverage, b2_names
            ): name
            for name in pending
        }
        for done, future in enumerate(as_completed(futures), 1):
            name = futures[future]
            try:
                rows = future.result()
                if len(rows) != len(core.ALPHAS) * 2:
                    raise ValueError(
                        f"unexpected prediction rows dataset={name}: {len(rows)}"
                    )
                completed[name] = rows
                _append_checkpoint(checkpoint, name, rows)
            except Exception as exc:
                errors.append({"dataset": name, "error": repr(exc)})
            completed_count = len(completed)
            if completed_count % 10 == 0 or completed_count == len(names):
                print(
                    f"Prediction progress: [{completed_count:4d}/{len(names):4d}] "
                    f"errors={len(errors)}",
                    flush=True,
                )

    if errors:
        write_rows(output_dir / "prediction_errors.csv", errors)
        raise SystemExit(
            f"{len(errors)} prediction errors; see prediction_errors.csv"
        )
    if set(completed) != set(names):
        raise ValueError("prediction coverage incomplete after worker completion")

    all_rows = [
        row
        for name in sorted(names)
        for row in completed[name]
    ]
    all_rows.sort(
        key=lambda row: (row["variant"], float(row["alpha"]), row["dataset_name"])
    )
    exp151_path = output_dir / "03_exp151_current_source_predictions.jsonl"
    exp152_path = output_dir / "04_exp152_b2_source_predictions.jsonl"
    write_jsonl(
        exp151_path, [row for row in all_rows if row["variant"] == CURRENT_VARIANT]
    )
    write_jsonl(
        exp152_path, [row for row in all_rows if row["variant"] == B2_VARIANT]
    )

    manifest = {
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_count": len(names),
        "rows_per_variant": len(names) * len(core.ALPHAS),
        "exp151_prediction_file": exp151_path.name,
        "exp151_prediction_sha256": core.sha256_file(exp151_path),
        "exp152_prediction_file": exp152_path.name,
        "exp152_prediction_sha256": core.sha256_file(exp152_path),
        "checkpoint_sha256": core.sha256_file(checkpoint),
        "evaluation_contract_sha256": core.sha256_file(contract_path),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
    }
    write_json(output_dir / "05_prediction_manifest.json", manifest)
    return manifest


def _parse_indices(value) -> set[int]:
    if value is None:
        return set()
    text = str(value).strip()
    if not text:
        return set()
    return {int(token) for token in text.replace(",", " ").split()}


def _tier_metrics(y_test, indices):
    selected = set(indices)
    truth = set(np.flatnonzero(np.asarray(y_test, dtype=int) == 1).tolist())
    tp = len(selected & truth)
    fp = len(selected - truth)
    fn = len(truth - selected)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    denominator = 2 * tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / denominator if denominator else 0.0,
    }


def evaluate_prediction_row(prediction_row, y_test):
    n_test = int(prediction_row["n_test"])
    if len(y_test) != n_test:
        raise ValueError(
            f"TEST label length mismatch dataset={prediction_row['dataset_name']}: "
            f"prediction={n_test} labels={len(y_test)}"
        )
    hard = _parse_indices(prediction_row.get("hard_indices"))
    standard = _parse_indices(prediction_row.get("standard_review_indices"))
    priority = _parse_indices(prediction_row.get("priority_review_indices"))
    if hard & standard or hard & priority or standard & priority:
        raise ValueError(
            f"lane overlap dataset={prediction_row['dataset_name']}"
        )
    combined = hard | standard | priority
    output = dict(prediction_row)
    for prefix, indices in (
        ("hard", hard),
        ("standard_review", standard),
        ("priority_review", priority),
        ("combined", combined),
    ):
        output.update(
            {f"{prefix}_{key}": value for key, value in _tier_metrics(y_test, indices).items()}
        )
    return output


def aggregate(rows):
    rows = list(rows)

    def total(key):
        return int(sum(float(row.get(key, 0) or 0) for row in rows))

    def mean(key):
        return float(np.mean([float(row.get(key, 0) or 0) for row in rows]))

    hard_tp, hard_fp = total("hard_tp"), total("hard_fp")
    standard_tp, standard_fp = total("standard_review_tp"), total(
        "standard_review_fp"
    )
    priority_tp, priority_fp = total("priority_review_tp"), total(
        "priority_review_fp"
    )
    candidate_counts = np.asarray(
        [float(row.get("candidate_count", 0) or 0) for row in rows], dtype=float
    )
    return {
        "datasets": len(rows),
        "hard_alerts": total("hard_count"),
        "hard_tp": hard_tp,
        "hard_fp": hard_fp,
        "hard_precision": hard_tp / max(1, hard_tp + hard_fp),
        "mean_hard_recall": mean("hard_recall"),
        "mean_hard_f1": mean("hard_f1"),
        "standard_review_candidates": total("standard_review_count"),
        "standard_review_tp": standard_tp,
        "standard_review_fp": standard_fp,
        "standard_review_precision": standard_tp
        / max(1, standard_tp + standard_fp),
        "priority_review_candidates": total("priority_review_count"),
        "priority_review_tp": priority_tp,
        "priority_review_fp": priority_fp,
        "priority_review_precision": priority_tp
        / max(1, priority_tp + priority_fp),
        "mean_combined_f1": mean("combined_f1"),
        "mean_combined_f1_scope": "human_assisted_diagnostic_only",
        "candidate_mean": float(np.mean(candidate_counts)),
        "candidate_median": float(np.median(candidate_counts)),
        "candidate_p90": float(np.percentile(candidate_counts, 90)),
        "candidate_p95": float(np.percentile(candidate_counts, 95)),
        "candidate_max": int(np.max(candidate_counts)),
        "candidate_zero_datasets": int(np.sum(candidate_counts == 0)),
        "abstained_datasets": int(
            sum(int(row.get("abstained", 0)) for row in rows)
        ),
        "resolution_blocked_datasets": int(
            sum(
                int(row.get("resolution_blocks_two_source_agreement", 0))
                for row in rows
            )
        ),
        "retrospective_counterfactual": 1,
        "prospective_validated": 0,
    }


def _summary_markdown(experiment_id, summaries):
    return (
        f"# {experiment_id}\n\n"
        "All pre-registered alpha values are reported together. No TEST metric "
        "selected alpha. Hard alert metrics are autonomous; Standard/Priority "
        "review and combined F1 are human-assisted diagnostics. Results are "
        "retrospective counterfactual virtual-equipment benchmark results, not "
        "prospective equipment validation.\n\n```json\n"
        + json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n"
    )


def _write_presentation_claims(path: Path, summaries):
    lines = [
        "# Exp151/152 presentation claims",
        "",
        "## 검증 완료 문구",
        "",
        "- Prediction 파일은 TEST label 없이 생성되고 SHA-256으로 동결된 뒤 별도 평가되었다.",
        "- 새 conformal 후보에 대해 Block-B와 Block-C evidence를 다시 계산했다.",
        "- 모든 alpha는 민감도 결과로 함께 보고했으며 하나를 운영값으로 선택하지 않았다.",
        "- Hard alert는 autonomous, review lane과 combined F1은 human-assisted 진단 지표다.",
        "",
        "## 조건부 문구",
        "",
        "- Exp152는 B2와 동일한 family-independent Exp84 configuration 계약을 확인한 policy-level 결과다.",
        "- 성능 수치는 공개 시계열 기반 가상 설비의 retrospective counterfactual 결과다.",
        "",
        "## 사용 금지 문구",
        "",
        "- 가장 좋은 alpha가 확정됐다.",
        "- 실제 반도체 설비에서 false alarm이 보장된다.",
        "- 전체 feature/configuration 선택 이력까지 end-to-end strict TRAIN-only로 검증됐다.",
        "- Priority review가 autonomous alert로 검증됐다.",
        "",
        "## Alpha별 결과",
        "",
        "```json",
        json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation_phase(output_dir: Path, dataset_limit: int | None = None):
    """Verify frozen predictions, then load labels and compute metrics."""

    output_dir = Path(output_dir)
    manifest_path = output_dir / "05_prediction_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("run --phase predict before evaluation")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract_path = output_dir / "01_evaluation_contract.json"
    if core.sha256_file(contract_path) != manifest["evaluation_contract_sha256"]:
        raise ValueError("evaluation contract changed after prediction freeze")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    verify_contract_inputs(contract)
    exp151_prediction_path = output_dir / manifest["exp151_prediction_file"]
    exp152_prediction_path = output_dir / manifest["exp152_prediction_file"]
    core.verify_frozen_prediction(
        exp151_prediction_path, manifest["exp151_prediction_sha256"]
    )
    core.verify_frozen_prediction(
        exp152_prediction_path, manifest["exp152_prediction_sha256"]
    )

    prediction_rows = read_jsonl(exp151_prediction_path) + read_jsonl(
        exp152_prediction_path
    )
    names = evaluation_dataset_names(dataset_limit)
    labels = {name: load_test_labels(name) for name in names}
    evaluated = [
        evaluate_prediction_row(row, labels[row["dataset_name"]])
        for row in prediction_rows
    ]
    evaluated.sort(
        key=lambda row: (row["variant"], float(row["alpha"]), row["dataset_name"])
    )

    exp151_rows = [row for row in evaluated if row["variant"] == CURRENT_VARIANT]
    exp152_rows = [row for row in evaluated if row["variant"] == B2_VARIANT]
    write_rows(output_dir / "06_exp151_current_source_results.csv", exp151_rows)
    write_rows(output_dir / "09_exp152_b2_source_results.csv", exp152_rows)

    comparison = []
    for variant, experiment_id, rows in (
        (CURRENT_VARIANT, CURRENT_EXPERIMENT_ID, exp151_rows),
        (B2_VARIANT, B2_EXPERIMENT_ID, exp152_rows),
    ):
        summaries = []
        for alpha in core.ALPHAS:
            subset = [row for row in rows if float(row["alpha"]) == alpha]
            summary = aggregate(subset)
            summary.update(
                {
                    "experiment_id": experiment_id,
                    "variant": variant,
                    "alpha": alpha,
                    "scope": "retrospective counterfactual virtual wafer-run benchmark",
                }
            )
            summaries.append(summary)
            comparison.append(summary)
        if variant == CURRENT_VARIANT:
            write_rows(output_dir / "07_exp151_current_source_summary.csv", summaries)
            (output_dir / "08_exp151_current_source_summary.md").write_text(
                _summary_markdown(experiment_id, summaries), encoding="utf-8"
            )
        else:
            write_rows(output_dir / "10_exp152_b2_source_summary.csv", summaries)
            (output_dir / "11_exp152_b2_source_summary.md").write_text(
                _summary_markdown(experiment_id, summaries), encoding="utf-8"
            )

    write_rows(output_dir / "12_alpha_sensitivity_comparison.csv", comparison)
    write_rows(
        output_dir / "13_dataset_level_candidate_diff.csv",
        [
            {
                "dataset": name,
                "alpha": alpha,
                "exp151_candidates": next(
                    row["candidate_indices"]
                    for row in exp151_rows
                    if row["dataset_name"] == name and float(row["alpha"]) == alpha
                ),
                "exp152_candidates": next(
                    row["candidate_indices"]
                    for row in exp152_rows
                    if row["dataset_name"] == name and float(row["alpha"]) == alpha
                ),
                "exp151_hard": next(
                    row["hard_indices"]
                    for row in exp151_rows
                    if row["dataset_name"] == name and float(row["alpha"]) == alpha
                ),
                "exp152_hard": next(
                    row["hard_indices"]
                    for row in exp152_rows
                    if row["dataset_name"] == name and float(row["alpha"]) == alpha
                ),
            }
            for name in names
            for alpha in core.ALPHAS
        ],
    )
    _write_presentation_claims(output_dir / "15_presentation_claims.md", comparison)
    write_output_manifest(output_dir)
    return comparison


def write_output_manifest(output_dir: Path):
    output_dir = Path(output_dir)
    manifest_rows = []
    for path in sorted(output_dir.glob("*")):
        if not path.is_file() or path.name in {"MANIFEST.csv", "MANIFEST.md"}:
            continue
        manifest_rows.append(
            {
                "file": path.name,
                "sha256": core.sha256_file(path),
                "retrospective_or_prospective": "retrospective_counterfactual",
                "autonomous_or_human_assisted": "hard_autonomous_review_human_assisted",
            }
        )
    write_rows(output_dir / "MANIFEST.csv", manifest_rows)
    lines = [
        "# Exp151/152 manifest",
        "",
        "Hard alert metrics are autonomous. Standard/Priority review and combined F1 are human-assisted diagnostics.",
        "All results are retrospective counterfactual virtual-equipment benchmark artifacts.",
        "",
    ] + [f"- `{row['file']}`: `{row['sha256']}`" for row in manifest_rows]
    (output_dir / "MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", required=True, choices=("audit", "predict", "evaluate")
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--reset-checkpoint", action="store_true")
    parser.add_argument(
        "--skip-db-hash",
        action="store_true",
        help="Smoke-test convenience only. Do not use for the full audit.",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    if args.phase == "audit":
        run_audit_phase(
            output_dir,
            dataset_limit=args.dataset_limit,
            skip_db_hash=args.skip_db_hash,
        )
    elif args.phase == "predict":
        run_prediction_phase(
            output_dir,
            dataset_limit=args.dataset_limit,
            workers=args.workers,
            reset_checkpoint=args.reset_checkpoint,
        )
    else:
        run_evaluation_phase(output_dir, dataset_limit=args.dataset_limit)


if __name__ == "__main__":
    main()
