import numpy as np
import pytest

from run_experiment_60_62_rocket_imaging_selector_variants import (
    family_adjusted_large_budget,
    rank_ensemble_guarded_indices,
    rank_ensemble_margin_indices,
    score_rank_bundle,
)


def make_bundle(scores):
    return {"test_scores": np.asarray(scores, dtype=np.float64), "indices": set()}


def test_rocket_guard_keeps_rank_candidates_that_are_also_rocket_top():
    bundles = {
        "rocket_exp40": make_bundle([0.99, 0.10, 0.80, 0.20, 0.05]),
        "exp55_best": make_bundle([0.01, 0.95, 0.90, 0.20, 0.05]),
        "exp56_best": make_bundle([0.01, 0.94, 0.85, 0.20, 0.05]),
    }

    selected = rank_ensemble_guarded_indices(bundles, count=2, guard="rocket_top", guard_count=2)

    assert selected == {2}


def test_two_model_guard_keeps_candidates_seen_by_at_least_two_models():
    bundles = {
        "rocket_exp40": make_bundle([0.99, 0.10, 0.80, 0.20, 0.05]),
        "exp55_best": make_bundle([0.01, 0.95, 0.90, 0.20, 0.05]),
        "exp56_best": make_bundle([0.01, 0.94, 0.85, 0.20, 0.05]),
    }

    selected = rank_ensemble_guarded_indices(bundles, count=2, guard="two_model_top", guard_count=2)

    assert selected == {1, 2}


def test_score_rank_bundle_reports_gap_between_selected_boundary_and_next():
    bundle = score_rank_bundle(make_bundle([0.9, 0.7, 0.6, 0.1]), count=2)

    assert bundle["indices"] == {0, 1}
    assert bundle["rank_margin"] == pytest.approx(0.1)


def test_margin_guard_reduces_budget_when_rank_boundary_is_ambiguous():
    bundles = {
        "rocket_exp40": make_bundle([0.99, 0.97, 0.96, 0.10]),
        "exp55_best": make_bundle([0.98, 0.97, 0.96, 0.10]),
        "exp56_best": make_bundle([0.97, 0.96, 0.95, 0.10]),
    }

    selected = rank_ensemble_margin_indices(bundles, count=3, min_margin=2.0)

    assert selected == {0, 1}


def test_family_adjusted_budget_is_more_conservative_for_high_fp_families():
    assert family_adjusted_large_budget("FordA", default_count=8, mode="conservative") == 5
    assert family_adjusted_large_budget("ScreenType", default_count=8, mode="conservative") == 6
    assert family_adjusted_large_budget("UnknownFamily", default_count=8, mode="conservative") == 8
