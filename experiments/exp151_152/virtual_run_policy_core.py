"""Pure, label-free policy helpers for Exp151/152.

This module deliberately contains no database access and no TEST-label access.
Keeping these functions pure makes the core policy easy to test and prevents
prediction logic from silently depending on evaluation labels.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from sklearn.model_selection import KFold

SEED = 20260717
ALPHAS = (0.005, 0.01, 0.02, 0.05)
REQUIRED_SOURCE_AGREEMENT = 2
EXPECTED_B2_CONFIG_NAME = (
    "aeon_mrh_mr1024_hk4_g32_prune1024_stable_tail_local_gap_knn3"
)


def sha256_file(path: Path) -> str:
    """Return a stable SHA-256 digest for a frozen artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_prediction(path: Path, expected_sha256: str) -> None:
    """Refuse evaluation if the prediction artifact changed after freezing."""

    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            "prediction artifact hash mismatch: "
            f"expected={expected_sha256} actual={actual} path={path}"
        )


def conformal_p_value(train_oof_scores: Iterable[float], score: float) -> float:
    """Compute the pre-registered one-sided empirical conformal p-value.

    Only TRAIN-normal out-of-fold scores enter the reference distribution.
    The +1 correction preserves a finite-sample lower bound of 1/(n+1).
    """

    scores = np.asarray(list(train_oof_scores), dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0 or not np.isfinite(score):
        return 1.0
    return float((1 + np.sum(scores >= float(score))) / (len(scores) + 1))


def conformal_p_values(
    train_oof_scores: Iterable[float], test_scores: Iterable[float]
) -> np.ndarray:
    """Vector form of :func:`conformal_p_value`.

    Each TEST run is scored independently. Appending more TEST runs therefore
    cannot change the p-value of an already-scored run.
    """

    return np.asarray(
        [conformal_p_value(train_oof_scores, score) for score in test_scores],
        dtype=float,
    )


def fold_plan(n_train: int, seed: int = SEED):
    """Return the frozen cross-fitting plan.

    - 5 <= n_train < 20: leave-one-out, matching the pre-registration.
    - n_train >= 20: deterministic five-fold cross-fitting.
    - n_train < 2: no valid fold plan.
    """

    n_train = int(n_train)
    if n_train < 2:
        return []
    indices = np.arange(n_train)
    if n_train < 20:
        return [(np.delete(indices, i), np.asarray([i])) for i in range(n_train)]
    splitter = KFold(n_splits=5, shuffle=True, random_state=int(seed))
    return [
        (fit.astype(int), query.astype(int))
        for fit, query in splitter.split(indices)
    ]


def agreement_indices(
    p_values_by_source: Mapping[str, np.ndarray],
    alpha: float,
    required_sources: int = REQUIRED_SOURCE_AGREEMENT,
):
    """Select TEST instances supported by at least ``required_sources`` sources."""

    names = sorted(p_values_by_source)
    if not names:
        return set(), np.zeros(0, dtype=int)
    lengths = {len(np.asarray(p_values_by_source[name])) for name in names}
    if len(lengths) != 1:
        raise ValueError(f"source p-value length mismatch: {sorted(lengths)}")
    matrix = np.vstack(
        [np.asarray(p_values_by_source[name], dtype=float) <= float(alpha) for name in names]
    )
    counts = matrix.sum(axis=0)
    selected = set(
        np.flatnonzero(counts >= int(required_sources)).astype(int).tolist()
    )
    return selected, counts


def route_lanes_from_recomputed_evidence(
    candidates,
    block_b_indices,
    block_c_indices,
    test_size: int,
):
    """Route new conformal candidates using newly recomputed evidence.

    Block-B is an independent cross-check and therefore controls autonomous
    Hard alerts. Block-C is auxiliary evidence applied only to candidates that
    did not become Hard; those candidates remain human-assisted review items.

    The function intentionally does *not* accept old Exp133/Exp135 output
    indices. Those indices were produced for the old Exp93 candidate universe
    and cannot validly confirm a new conformal candidate universe.
    """

    candidates = set(int(i) for i in candidates)
    block_b_indices = set(int(i) for i in block_b_indices)
    block_c_indices = set(int(i) for i in block_c_indices)
    universe = set(range(int(test_size)))

    invalid = sorted((candidates | block_b_indices | block_c_indices) - universe)
    if invalid:
        raise ValueError(
            f"out-of-bounds indices for test_size={test_size}: {invalid[:10]}"
        )

    hard = candidates & block_b_indices
    standard_before_priority = candidates - hard
    priority_review = standard_before_priority & block_c_indices
    standard_review = standard_before_priority - priority_review
    no_alert = universe - candidates

    return {
        "hard": hard,
        "standard_review": standard_review,
        "priority_review": priority_review,
        "no_alert": no_alert,
    }


def calibration_status(n_train: int):
    """Return the pre-registered low-TRAIN handling status.

    Earlier Exp149/150 documentation called this "review-only" while the code
    actually emitted no candidates. V2 removes that ambiguity: fewer than five
    TRAIN-normal runs means calibration abstention and no autonomous/review
    candidate indices for that dataset.
    """

    if int(n_train) < 5:
        return {
            "status": "insufficient_calibration_n_train_lt5",
            "abstain": True,
            "review_only": False,
        }
    return {"status": "calibrated", "abstain": False, "review_only": False}


def resolution_status(
    minimum_p_by_source: Mapping[str, float],
    alpha: float,
    required_sources: int = REQUIRED_SOURCE_AGREEMENT,
):
    """Describe whether conformal resolution can support source agreement.

    Exp149/150 marked a dataset limited when *any* source could not reach alpha.
    That was too strict because the policy requires only two agreeing sources.
    V2 counts how many sources can reach alpha and blocks only when fewer than
    the required number are eligible.
    """

    eligible = sorted(
        name
        for name, minimum_p in minimum_p_by_source.items()
        if float(minimum_p) <= float(alpha)
    )
    return {
        "eligible_sources": eligible,
        "eligible_source_count": len(eligible),
        "required_source_count": int(required_sources),
        "blocks_required_agreement": len(eligible) < int(required_sources),
    }


def validate_b2_manifest(rows, expected_datasets):
    """Validate the frozen B2 full-coverage contract before Exp152 runs.

    Exp152 recomputes Exp84 scores, but it must prove that it is using the same
    B2 configuration and seed and that the B2 coverage contract contains every
    evaluation dataset. A manifest row is used only as a contract assertion;
    its TEST-derived selected indices are never reused.
    """

    selected = {}
    for row in rows:
        if row.get("threshold_method") != "count_cap_2pct":
            continue
        name = row.get("dataset_name", "")
        if not name:
            continue
        if row.get("config_name") != EXPECTED_B2_CONFIG_NAME:
            raise ValueError(
                f"B2 config mismatch dataset={name}: {row.get('config_name')}"
            )
        if int(float(row.get("random_state", -1))) != SEED:
            raise ValueError(
                f"B2 seed mismatch dataset={name}: {row.get('random_state')}"
            )
        if int(float(row.get("source_uses_family_name", 1))) != 0:
            raise ValueError(f"B2 family dependence found dataset={name}")
        if int(float(row.get("source_uses_test_length", 1))) != 0:
            raise ValueError(f"B2 TEST-length dependence found dataset={name}")
        if name in selected:
            raise ValueError(f"duplicate B2 count_cap_2pct row: {name}")
        selected[name] = row

    expected = set(expected_datasets)
    actual = set(selected)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "B2 coverage mismatch: "
            f"expected={len(expected)} actual={len(actual)} "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    return selected
