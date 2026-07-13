import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "run_rank_ensemble_calibration_v2.py"
sys.path.insert(0, str(ROOT))


def load_module():
    spec = importlib.util.spec_from_file_location("rank_ensemble_v2", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hybrid_scores_route_short_to_recon_prob_and_long_to_mse():
    module = load_module()
    mse_train = np.array([1.0, 2.0], dtype=np.float32)
    mse_test = np.array([10.0, 20.0], dtype=np.float32)
    prob_train = np.array([3.0, 4.0], dtype=np.float32)
    prob_test = np.array([30.0, 40.0], dtype=np.float32)

    short_train, short_test, short_route = module.select_hybrid_scores_by_length(
        seq_len=149,
        mse_scores=(mse_train, mse_test),
        recon_prob_scores=(prob_train, prob_test),
    )
    long_train, long_test, long_route = module.select_hybrid_scores_by_length(
        seq_len=150,
        mse_scores=(mse_train, mse_test),
        recon_prob_scores=(prob_train, prob_test),
    )

    assert short_route == "Recon_Probability_NLL"
    assert np.array_equal(short_train, prob_train)
    assert np.array_equal(short_test, prob_test)
    assert long_route == "MSE_Reconstruction"
    assert np.array_equal(long_train, mse_train)
    assert np.array_equal(long_test, mse_test)


if __name__ == "__main__":
    test_hybrid_scores_route_short_to_recon_prob_and_long_to_mse()
    print("rank ensemble v2 smoke test passed")
