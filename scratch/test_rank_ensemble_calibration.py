import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "run_rank_ensemble_calibration.py"


def load_module():
    spec = importlib.util.spec_from_file_location("rank_ensemble", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rank_ensemble_rewards_consensus_outlier():
    module = load_module()
    scores = {
        "multi_aug": np.array([0.1, 0.2, 10.0], dtype=np.float32),
        "fused": np.array([0.1, 0.3, 9.0], dtype=np.float32),
        "hybrid": np.array([0.2, 0.4, 8.0], dtype=np.float32),
    }

    ensemble = module.weighted_rank_ensemble(scores, {"multi_aug": 0.5, "fused": 0.3, "hybrid": 0.2})

    assert int(np.argmax(ensemble)) == 2
    assert ensemble[2] > ensemble[1] > ensemble[0]


def test_evaluate_scores_uses_train_evt_threshold():
    module = load_module()
    train_scores = np.array([0.0] * 20 + [0.10, 0.12, 0.14, 0.16], dtype=np.float32)
    test_scores = np.array([0.01, 0.02, 0.03, 0.17, 0.18], dtype=np.float32)
    y_test = np.array([0, 0, 0, 1, 1], dtype=np.int64)

    metrics = module.evaluate_scores(y_test, test_scores, train_scores)

    assert metrics["threshold_method"] in {"evt_gpd", "adaptive_fallback", "percentile_fallback"}
    assert metrics["f1_evt"] == 1.0


def test_weighted_reference_rank_ensemble_uses_train_distribution():
    module = load_module()
    train_scores = {
        "multi_aug": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        "fused": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        "hybrid": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
    }
    test_scores = {
        "multi_aug": np.array([0.5, 3.5], dtype=np.float32),
        "fused": np.array([0.5, 3.5], dtype=np.float32),
        "hybrid": np.array([0.5, 3.5], dtype=np.float32),
    }

    train_ensemble, test_ensemble = module.weighted_reference_rank_ensemble(
        train_scores,
        test_scores,
        {"multi_aug": 1 / 3, "fused": 1 / 3, "hybrid": 1 / 3},
    )

    assert np.allclose(train_ensemble, np.array([0.25, 0.5, 0.75, 1.0]))
    assert np.allclose(test_ensemble, np.array([0.25, 1.0]))


if __name__ == "__main__":
    test_rank_ensemble_rewards_consensus_outlier()
    test_evaluate_scores_uses_train_evt_threshold()
    test_weighted_reference_rank_ensemble_uses_train_distribution()
    print("rank ensemble smoke test passed")
