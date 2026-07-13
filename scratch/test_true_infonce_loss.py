import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "run_all_adaptive_cnn_true_infonce_multi_aug.py"


def load_module():
    spec = importlib.util.spec_from_file_location("true_infonce_multi_aug", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_true_infonce_rewards_matched_pairs():
    module = load_module()
    embeddings = torch.eye(4, dtype=torch.float32)

    aligned_loss = module.true_infonce_loss(embeddings, embeddings, temperature=0.2)
    shuffled_loss = module.true_infonce_loss(embeddings, embeddings.roll(1, dims=0), temperature=0.2)

    assert aligned_loss.item() < shuffled_loss.item()


def test_sanitize_series_interpolates_nan_values():
    module = load_module()
    series = module.sanitize_series(torch.tensor([float("nan"), 1.0, float("nan"), 3.0, float("nan")]).numpy())

    assert series.tolist() == [1.0, 1.0, 2.0, 3.0, 3.0]
    assert not torch.isnan(torch.tensor(series)).any()


if __name__ == "__main__":
    test_true_infonce_rewards_matched_pairs()
    test_sanitize_series_interpolates_nan_values()
    print("true_infonce_loss smoke test passed")
