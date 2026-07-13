import numpy as np

from run_model_hard_research_experiments import aeon_feature_pair


def test_aeon_multirocket_handles_single_train_case():
    train = np.random.default_rng(1).normal(size=(1, 64)).astype(np.float32)
    test = np.random.default_rng(2).normal(size=(5, 64)).astype(np.float32)
    config = {
        "kind": "aeon_multirocket",
        "num_kernels": 256,
        "random_state": 1,
    }
    train_features, test_features = aeon_feature_pair(train, test, config, {})
    assert train_features.shape[0] == 1
    assert test_features.shape[0] == 5


def test_aeon_multirocket_hydra_handles_single_train_case():
    train = np.random.default_rng(3).normal(size=(1, 64)).astype(np.float32)
    test = np.random.default_rng(4).normal(size=(5, 64)).astype(np.float32)
    config = {
        "kind": "aeon_multirocket_hydra",
        "num_kernels": 256,
        "hydra_kernels": 4,
        "hydra_groups": 8,
        "random_state": 2,
    }
    train_features, test_features = aeon_feature_pair(train, test, config, {})
    assert train_features.shape[0] == 1
    assert test_features.shape[0] == 5
