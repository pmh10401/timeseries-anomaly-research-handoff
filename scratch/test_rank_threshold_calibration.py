import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "run_rank_threshold_calibration.py"


def load_module():
    spec = importlib.util.spec_from_file_location("rank_threshold_calibration", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prior_q_predicts_expected_tail_count():
    module = load_module()
    scores = np.linspace(0.0, 1.0, 100)

    preds = module.predict_by_strategy(scores, strategy="prior_q_02")

    assert preds.sum() == 2
    assert np.all(preds[-2:] == 1)
    assert preds[-3] == 0


def test_min_k_strategy_prevents_single_hit_for_larger_tests():
    module = load_module()
    scores = np.linspace(0.0, 1.0, 100)

    preds = module.predict_by_strategy(scores, strategy="prior_q_02_min_k_3")

    assert preds.sum() == 3
    assert np.all(preds[-3:] == 1)


def test_gap_strategy_cuts_after_largest_upper_tail_gap():
    module = load_module()
    scores = np.array([0.01, 0.02, 0.03, 0.40, 0.41, 0.42, 0.90, 0.91])

    preds = module.predict_by_strategy(scores, strategy="tail_gap")

    assert preds.sum() == 2
    assert np.all(preds[-2:] == 1)


if __name__ == "__main__":
    test_prior_q_predicts_expected_tail_count()
    test_min_k_strategy_prevents_single_hit_for_larger_tests()
    test_gap_strategy_cuts_after_largest_upper_tail_gap()
    print("rank threshold calibration smoke test passed")
