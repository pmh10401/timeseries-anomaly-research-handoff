import pandas as pd

from run_experiment_85_exp84_hard_specialist_selector import (
    SPECIALIST_CAP3,
    SPECIALIST_FAMILY_GUARD,
    choose_source,
    is_operationally_guarded,
)


def row(**kwargs):
    defaults = {
        "dataset_name": "InlineSkate_normal_0",
        "family": "InlineSkate",
        "experiment_id": "source",
        "config_name": "cfg",
        "threshold_method": "selector",
        "test_size": 50,
        "predicted_count": 1,
        "train_exceed_rate": 0.0,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


def test_gain_family_selector_uses_exp84_cap3_for_known_gain_family():
    base = row(f1=0.0)
    review = row(f1=0.1)
    exp84 = row(f1=1.0, threshold_method=SPECIALIST_CAP3)
    by_threshold = {(str(exp84["dataset_name"]), SPECIALIST_CAP3): exp84}

    selected, source_name, reason = choose_source(
        "gain_family_exp84_cap3_else_primary", base, review, by_threshold
    )

    assert selected["f1"] == 1.0
    assert source_name == "exp84_cap3_gain_family"
    assert "research prior" in reason


def test_gain_family_selector_falls_back_for_non_gain_family():
    base = row(dataset_name="Crop_normal_0", family="Crop", f1=0.5)
    review = row(dataset_name="Crop_normal_0", family="Crop", f1=0.6)
    exp84 = row(dataset_name="Crop_normal_0", family="Crop", f1=1.0, threshold_method=SPECIALIST_CAP3)
    by_threshold = {(str(exp84["dataset_name"]), SPECIALIST_CAP3): exp84}

    selected, source_name, reason = choose_source(
        "gain_family_exp84_cap3_else_primary", base, review, by_threshold
    )

    assert selected["f1"] == 0.5
    assert source_name == "exp74d_primary"
    assert "fallback" in reason


def test_operational_guard_rejects_large_predicted_rate():
    assert is_operationally_guarded(row(test_size=50, predicted_count=3, train_exceed_rate=0.01))
    assert not is_operationally_guarded(row(test_size=50, predicted_count=4, train_exceed_rate=0.01))
    assert not is_operationally_guarded(row(test_size=50, predicted_count=1, train_exceed_rate=0.02))


def test_hard_guarded_selector_uses_guarded_row_only_when_safe():
    base = row(f1=0.2)
    review = row(f1=0.3)
    safe = row(f1=0.9, threshold_method=SPECIALIST_FAMILY_GUARD, predicted_count=2, train_exceed_rate=0.01)
    unsafe = row(f1=1.0, threshold_method=SPECIALIST_FAMILY_GUARD, predicted_count=5, train_exceed_rate=0.01)

    selected, source_name, _ = choose_source(
        "hard_exp84_guarded_else_primary",
        base,
        review,
        {(str(safe["dataset_name"]), SPECIALIST_FAMILY_GUARD): safe},
    )
    assert selected["f1"] == 0.9
    assert source_name == "exp84_family_guard_guarded"

    selected, source_name, _ = choose_source(
        "hard_exp84_guarded_else_primary",
        base,
        review,
        {(str(unsafe["dataset_name"]), SPECIALIST_FAMILY_GUARD): unsafe},
    )
    assert selected["f1"] == 0.2
    assert source_name == "exp74d_primary"
