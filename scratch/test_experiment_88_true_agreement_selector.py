import pandas as pd

from run_experiment_88_true_agreement_exp84_selector import (
    first_top_index,
    parse_indices,
    train_safe,
)


def row(**kwargs):
    defaults = {
        "top_score_indices": "7 3 9",
        "train_exceed_rate": 0.01,
        "top1_threshold_margin": 0.1,
        "top1_top2_margin": 0.02,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


def test_parse_indices_handles_blank_nan_and_numbers():
    assert parse_indices("") == set()
    assert parse_indices(float("nan")) == set()
    assert parse_indices("1 4 9") == {1, 4, 9}
    assert parse_indices("1.0 4.0") == {1, 4}


def test_first_top_index_preserves_score_rank_order():
    assert first_top_index(row(top_score_indices="7 3 9")) == 7
    assert first_top_index(row(top_score_indices="2")) == 2
    assert first_top_index(row(top_score_indices="")) is None


def test_train_safe_uses_train_exceed_rate_only():
    assert train_safe(row(train_exceed_rate=0.014), 0.015)
    assert not train_safe(row(train_exceed_rate=0.016), 0.015)
    assert not train_safe(None, 0.015)
