import numpy as np
import pytest

from run_model_hard_research_experiments import threshold_selection_diagnostics


def test_threshold_selection_diagnostics_records_selected_and_top_indices():
    scores = np.asarray([0.1, 0.9, 0.4, 0.8], dtype=np.float64)

    diag = threshold_selection_diagnostics(scores, threshold=0.5, top_n=3)

    assert diag["selected_indices"] == "1 3"
    assert diag["top_score_indices"] == "1 3 2"
    assert diag["selected_score_max"] == 0.9
    assert diag["selected_score_min"] == 0.8
    assert diag["top1_score"] == 0.9
    assert diag["top2_score"] == 0.8
    assert diag["top1_top2_margin"] == pytest.approx(0.1)
    assert diag["top1_threshold_margin"] == pytest.approx(0.4)


def test_threshold_selection_diagnostics_handles_empty_scores():
    diag = threshold_selection_diagnostics([], threshold=0.5)

    assert diag["selected_indices"] == ""
    assert diag["top_score_indices"] == ""
    assert diag["top1_score"] == ""
