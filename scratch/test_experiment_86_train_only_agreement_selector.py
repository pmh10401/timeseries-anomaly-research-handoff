import pandas as pd

from run_experiment_86_train_only_agreement_exp84_selector import (
    SPECIALIST_FAMILY_GUARD,
    base_overlap_any,
    base_strong_overlap,
    choose_source,
    train_safe,
)


def row(**kwargs):
    defaults = {
        "dataset_name": "AnyFamily_normal_0",
        "family": "AnyFamily",
        "experiment_id": "source",
        "config_name": "cfg",
        "threshold_method": "selector",
        "test_size": 50,
        "predicted_count": 1,
        "train_exceed_rate": 0.0,
        "rocket_exp55_or_exp56_overlap_any": 0,
        "rocket_exp55_overlap": 0,
        "rocket_exp56_overlap": 0,
        "exp55_exp56_overlap": 0,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


def test_train_safe_uses_only_train_exceed_rate():
    assert train_safe(row(train_exceed_rate=0.01), 0.015)
    assert not train_safe(row(train_exceed_rate=0.02), 0.015)
    assert not train_safe(None, 0.015)


def test_overlap_proxy_detects_any_and_strong_agreement():
    weak = row(rocket_exp55_overlap=1)
    strong = row(rocket_exp55_overlap=2)
    broad = row(rocket_exp55_overlap=1, rocket_exp56_overlap=1)

    assert base_overlap_any(weak)
    assert not base_strong_overlap(weak)
    assert base_strong_overlap(strong)
    assert base_strong_overlap(broad)


def test_agreement_selector_uses_exp84_without_family_prior():
    base = row(f1=0.2, rocket_exp55_overlap=1)
    review = row(f1=0.3)
    exp84 = row(f1=0.9, threshold_method=SPECIALIST_FAMILY_GUARD, train_exceed_rate=0.01)
    by_threshold = {(str(exp84["dataset_name"]), SPECIALIST_FAMILY_GUARD): exp84}

    selected, source_name, reason = choose_source(
        "exp84_agree_proxy_te015_fg_else_primary", base, review, by_threshold
    )

    assert selected["f1"] == 0.9
    assert source_name == "exp84_family_guard_agree_proxy"
    assert "overlap" in reason


def test_agreement_selector_falls_back_when_overlap_missing():
    base = row(f1=0.2)
    review = row(f1=0.3)
    exp84 = row(f1=0.9, threshold_method=SPECIALIST_FAMILY_GUARD, train_exceed_rate=0.01)
    by_threshold = {(str(exp84["dataset_name"]), SPECIALIST_FAMILY_GUARD): exp84}

    selected, source_name, reason = choose_source(
        "exp84_agree_proxy_te015_fg_else_primary", base, review, by_threshold
    )

    assert selected["f1"] == 0.2
    assert source_name == "exp74d_primary"
    assert "fallback" in reason
