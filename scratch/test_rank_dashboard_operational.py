import sys
import sqlite3
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serve_rank_dashboard as dash


def test_summarize_by_config_includes_operational_metrics():
    rows = [
        {
            "dataset_name": "A",
            "config_name": "rocket",
            "threshold_method": "count_cap_2pct",
            "f1": "0.5",
            "auc_roc": "0.8",
            "auc_pr": "0.4",
            "predicted_count": "3",
            "tp": "2",
            "fp": "1",
            "fn": "1",
            "train_exceed_rate": "0.005",
        },
        {
            "dataset_name": "B",
            "config_name": "rocket",
            "threshold_method": "count_cap_2pct",
            "f1": "1.0",
            "auc_roc": "1.0",
            "auc_pr": "0.9",
            "predicted_count": "1",
            "tp": "1",
            "fp": "0",
            "fn": "0",
            "train_exceed_rate": "0.010",
        },
    ]
    summary = dash.summarize_by_config(rows)
    best = summary[0]
    assert best["strategy"] == "rocket::count_cap_2pct"
    assert best["mean_fp"] == 0.5
    assert best["mean_tp"] == 1.5
    assert round(best["alert_precision"], 4) == 0.75
    assert best["mean_train_exceed_rate"] == 0.0075
    assert best["operational_tier"] == "운영 기본 후보"


def test_parse_log_error_summary_groups_reasons_and_families():
    text = "\n".join(
        [
            "2026-07-07 01:00:00,000 [ERROR] Error evaluating dataset GestureMidAirD1_normal_1: setting an array element with a sequence.",
            "2026-07-07 01:00:01,000 [ERROR] Error evaluating dataset MelbournePedestrian_normal_1: Input contains NaN.",
            "2026-07-07 01:00:02,000 [ERROR] Error evaluating Yoga_normal_1: out of memory",
        ]
    )
    errors = dash.parse_error_events(text.splitlines())
    summary = dash.error_summary(errors)
    assert summary["total_errors"] == 3
    assert summary["by_reason"]["variable_length_inhomogeneous_shape"] == 1
    assert summary["by_reason"]["nan_or_inf"] == 1
    assert summary["by_reason"]["memory"] == 1
    assert summary["by_family"]["GestureMidAirD1"] == 1
    assert summary["by_family"]["Yoga"] == 1


def test_coverage_snapshot_reports_done_and_error_gap():
    rows = [{"dataset_name": "A"}, {"dataset_name": "B"}, {"dataset_name": "B"}]
    errors = [{"dataset_name": "C", "family": "C", "reason": "nan_or_inf"}]
    coverage = dash.coverage_snapshot(rows, expected=5, errors=errors)
    assert coverage["datasets_done"] == 2
    assert coverage["error_datasets"] == 1
    assert coverage["remaining_or_missing"] == 2
    assert coverage["coverage_percent"] == 40.0


def test_recent_from_rows_uses_latest_unique_dataset_as_fallback():
    rows = [
        {"dataset_name": "A", "sequence_length": "100", "train_count": "20"},
        {"dataset_name": "A", "sequence_length": "100", "train_count": "20"},
        {"dataset_name": "B", "sequence_length": "80", "train_score_count": "15"},
    ]
    recent = dash.recent_from_rows(rows, limit=2)
    assert [item["name"] for item in recent] == ["A", "B"]
    assert recent[-1]["length"] == 80
    assert recent[-1]["train_size"] == 15


def test_family_trouble_spots_rank_problem_families_for_strategy():
    rows = [
        {"dataset_name": "A_normal_1", "config_name": "rocket", "threshold_method": "cap2", "f1": "0", "tp": "0", "fp": "2", "fn": "1", "predicted_count": "2"},
        {"dataset_name": "A_normal_2", "config_name": "rocket", "threshold_method": "cap2", "f1": "0.5", "tp": "1", "fp": "1", "fn": "1", "predicted_count": "2"},
        {"dataset_name": "B_normal_1", "config_name": "rocket", "threshold_method": "cap2", "f1": "1", "tp": "1", "fp": "0", "fn": "0", "predicted_count": "1"},
        {"dataset_name": "A_normal_1", "config_name": "rocket", "threshold_method": "cap3", "f1": "1", "tp": "1", "fp": "0", "fn": "0", "predicted_count": "1"},
    ]
    spots = dash.family_trouble_spots(rows, "rocket::cap2", limit=2)
    assert spots[0]["family"] == "A"
    assert spots[0]["datasets"] == 2
    assert spots[0]["zero_f1_count"] == 1
    assert spots[0]["total_fp"] == 3
    assert spots[0]["alert_precision"] == 1 / 4


def test_throughput_snapshot_uses_recent_progress_events():
    lines = [
        "2026-07-07 08:00:00,000 [INFO] Progress: [  10/100] rows=90 | best=x meanF1=0.1",
        "2026-07-07 08:05:00,000 [INFO] Progress: [  30/100] rows=270 | best=x meanF1=0.2",
    ]
    events = dash.parse_progress_events(lines)
    snapshot = dash.throughput_snapshot(events, done=30, expected=100, elapsed=600)
    assert snapshot["recent_datasets_per_minute"] == 4.0
    assert snapshot["average_datasets_per_minute"] == 3.0
    assert round(snapshot["recent_eta_seconds"]) == 1050
    assert snapshot["last_progress"]["done"] == 30


def test_experiment_compare_includes_running_and_delta_to_best():
    completed = [
        {"id": "done1", "label": "Done 1", "best_strategy": "a", "mean_f1": 0.5, "mean_fp": 1.0},
        {"id": "done2", "label": "Done 2", "best_strategy": "b", "mean_f1": 0.7, "mean_fp": 2.0},
    ]
    active_summary = [{"strategy": "active", "mean_f1": 0.6, "mean_fp": 0.5}]
    rows = dash.experiment_compare_snapshot("active_exp", "Active Exp", active_summary, completed, is_running=True)
    assert rows[0]["status"] == "running"
    assert rows[0]["id"] == "active_exp"
    assert rows[0]["delta_vs_best_f1"] == -0.1
    assert rows[1]["id"] == "done2"
    assert rows[1]["delta_vs_best_f1"] == 0.0


def test_metric_glossary_has_plain_korean_examples():
    glossary = dash.metric_glossary_snapshot()
    by_id = {item["id"]: item for item in glossary}
    assert "mean_fp" in by_id
    assert "alert_precision" in by_id
    assert "train_exceed" in by_id
    assert "정상" in by_id["mean_fp"]["description"]
    assert "예:" in by_id["mean_fp"]["example"]
    assert by_id["alert_precision"]["direction"]
    assert by_id["train_exceed"]["group"] == "운영 지표"


def test_metric_help_ui_is_collapsible_and_click_targeted():
    html = dash.INDEX_HTML
    assert '<details class="panel metric-help-panel"' in html
    assert "<summary" in html
    assert "open>" not in html.split('<details class="panel metric-help-panel"', 1)[1].split("</details>", 1)[0]
    assert 'id="metricHelpPopover"' in html
    assert "showMetricHelp" in html
    assert "hideMetricHelp" in html


def test_resource_metric_uses_compact_font_class():
    html = dash.INDEX_HTML
    assert 'class="value resource-value" id="cpu"' in html
    assert ".resource-value" in html
    assert "font-size: 20px" in html


def test_experiment_compare_does_not_hide_lower_ranked_completed_runs():
    html = dash.INDEX_HTML
    compare_block = html.split("const compareRows = document.getElementById('compareRows');", 1)[1].split("setText('compareInfo'", 1)[0]
    assert ".slice(0, 10)" not in compare_block


def test_dashboard_registers_new_queued_experiments():
    assert "experiment_43_explanation_space_transforms" in dash.EXPERIMENTS
    assert "experiment_44_classical_embedding_baselines" in dash.EXPERIMENTS
    assert dash.EXPERIMENTS["experiment_43_explanation_space_transforms"]["detail_csv"].name == "experiment_43_explanation_space_transforms_results.csv"
    assert dash.EXPERIMENTS["experiment_44_classical_embedding_baselines"]["summary_csv"].name == "experiment_44_classical_embedding_baselines_summary.csv"


def test_process_tree_snapshot_aggregates_child_processes():
    ps_output = "\n".join(
        [
            "100 1 10.0 1.0 python",
            "101 100 40.0 2.0 python",
            "102 101 50.0 3.0 python",
            "200 1 90.0 4.0 other",
        ]
    )
    snapshot = dash.process_tree_snapshot(100, ps_output=ps_output)
    assert snapshot["pid"] == 100
    assert snapshot["process_count"] == 3
    assert snapshot["cpu"] == 100.0
    assert snapshot["mem"] == 6.0
    assert snapshot["child_pids"] == [101, 102]


def test_resource_sampling_does_not_raise_when_process_creation_is_temporarily_unavailable():
    original = dash.subprocess.check_output

    def unavailable(*args, **kwargs):
        raise BlockingIOError(35, "Resource temporarily unavailable")

    dash.subprocess.check_output = unavailable
    try:
        assert dash.process_tree_snapshot(100) is None
        assert dash.system_cpu_snapshot()["available"] is False
    finally:
        dash.subprocess.check_output = original


def test_system_cpu_snapshot_parses_top_output():
    top_output = "CPU usage: 12.34% user, 5.66% sys, 82.00% idle"
    snapshot = dash.system_cpu_snapshot(top_output=top_output)
    assert snapshot["cpu_percent"] == 18.0
    assert snapshot["cpu_user_percent"] == 12.34
    assert snapshot["cpu_system_percent"] == 5.66
    assert snapshot["source"] == "top"


def test_gpu_snapshot_handles_unavailable_sampler():
    snapshot = dash.gpu_snapshot(nvidia_output="", powermetrics_output="", errors=["powermetrics requires sudo"])
    assert snapshot["available"] is False
    assert "requires sudo" in snapshot["note"]


def test_completed_snapshot_exposes_operational_columns():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        detail = tmp_path / "detail.csv"
        summary = tmp_path / "summary.csv"
        detail.write_text("dataset_name,config_name,threshold_method,f1,tp,fp,fn,predicted_count,train_exceed_rate\nA,c,m,1,1,0,0,1,0.0\n")
        summary.write_text(
            "config_name,threshold_method,mean_f1,median_f1,mean_fp,mean_tp,alert_precision,mean_train_exceed_rate,mean_predicted_count,num_datasets\n"
            "c,m,1.0,1.0,0.0,1.0,1.0,0.0,1.0,1\n"
        )
        original = dash.EXPERIMENTS.copy()
        try:
            dash.EXPERIMENTS.clear()
            dash.EXPERIMENTS["x"] = {
                "label": "X",
                "detail_csv": detail,
                "summary_csv": summary,
                "stdout_log": tmp_path / "out.log",
            }
            snapshot = dash.completed_result_snapshot({"completed": ["x"], "history": [{"id": "x", "finished_at": "now"}]})
        finally:
            dash.EXPERIMENTS.clear()
            dash.EXPERIMENTS.update(original)
        assert snapshot[0]["mean_fp"] == 0.0
        assert snapshot[0]["alert_precision"] == 1.0
        assert snapshot[0]["mean_train_exceed_rate"] == 0.0


def test_unified_experiment_items_include_external_planned_and_blocked_states():
    state = {"queue": ["queued_exp"], "completed": ["done_exp"], "running": None}
    external = {"id": "live_exp", "label": "Live", "status": "running", "expected_datasets": 10}
    plan = [
        {"id": "planned_exp", "label": "Planned", "status": "planned"},
        {"id": "blocked_exp", "label": "Blocked", "status": "blocked", "blocked_reason": "missing grain"},
    ]

    items = dash.unified_experiment_items(state, external, plan)
    by_id = {item["id"]: item for item in items}

    assert by_id["live_exp"]["status"] == "running"
    assert by_id["queued_exp"]["status"] == "queued"
    assert by_id["planned_exp"]["status"] == "planned"
    assert by_id["blocked_exp"]["blocked_reason"] == "missing grain"


def test_unified_experiment_items_preserve_completed_history_order():
    items = dash.unified_experiment_items({"completed": ["z_old", "a_new"]}, {}, [])
    completed = [item["id"] for item in items if item["status"] == "completed"]

    assert completed == ["z_old", "a_new"]


def test_unified_experiment_items_mark_bundle_complete_after_external_completion():
    plan = [{"id": "audit", "status": "planned", "runner_id": "bundle"}]
    items = dash.unified_experiment_items({}, {"id": "bundle", "status": "completed"}, plan)

    assert items[0]["status"] == "completed"


def test_dataset_context_snapshot_explains_benchmark_grain():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "sample.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE datasets (
                id INTEGER PRIMARY KEY, name TEXT, train_normal_count INTEGER,
                test_normal_count INTEGER, test_anomaly_count INTEGER, test_total_count INTEGER
            );
            CREATE TABLE instances (
                id INTEGER PRIMARY KEY, dataset_id INTEGER, split TEXT,
                instance_index INTEGER, label TEXT, values_blob BLOB, labels_blob BLOB
            );
            INSERT INTO datasets VALUES (1, 'Demo', 3, 9, 1, 10);
            """
        )
        conn.commit()
        conn.close()

        context = dash.dataset_context_snapshot(db_path)

        assert context["dataset_count"] == 1
        assert context["train_normal_instances"] == 3
        assert context["test_anomaly_instances"] == 1
        assert context["candidate_index_meaning"] == "TEST time-series instance index"
        assert context["production_run_mapping_verified"] is False


def test_gpt_handoff_contains_queue_metrics_and_dataset_caveat():
    status = {
        "active_experiment": "exp_demo",
        "progress_percent": 25.0,
        "unified_experiments": [{"id": "next", "status": "planned"}],
        "runtime_health": {"heartbeat": {}},
    }
    context = {
        "dataset_count": 1119,
        "evaluation_dataset_count": 1117,
        "candidate_index_meaning": "TEST time-series instance index",
        "production_run_mapping_verified": False,
    }

    handoff = dash.gpt_handoff_snapshot(status, context)
    markdown = dash.gpt_handoff_markdown(handoff)

    assert handoff["schema_version"] == "1.0"
    assert handoff["current_status"]["active_experiment"] == "exp_demo"
    assert "TEST time-series instance index" in markdown
    assert "production run mapping is not verified" in markdown


def test_gpt_handoff_markdown_limits_completed_experiments_but_json_keeps_them():
    items = [{"id": f"done_{index}", "status": "completed"} for index in range(15)]
    handoff = dash.gpt_handoff_snapshot(
        {"active_experiment": "x", "progress_percent": 0, "unified_experiments": items},
        {"candidate_index_meaning": "TEST time-series instance index", "production_run_mapping_verified": False},
    )

    markdown = dash.gpt_handoff_markdown(handoff)

    assert len(handoff["experiment_work_items"]) == 15
    assert markdown.count(": completed") == 10
    assert "5 additional completed experiments omitted" in markdown


def test_dashboard_has_gpt_export_and_dataset_context_controls():
    html = dash.INDEX_HTML
    assert 'id="copyGptHandoff"' in html
    assert 'id="downloadGptJson"' in html
    assert 'id="datasetContext"' in html


if __name__ == "__main__":
    test_summarize_by_config_includes_operational_metrics()
    test_parse_log_error_summary_groups_reasons_and_families()
    test_coverage_snapshot_reports_done_and_error_gap()
    test_recent_from_rows_uses_latest_unique_dataset_as_fallback()
    test_family_trouble_spots_rank_problem_families_for_strategy()
    test_throughput_snapshot_uses_recent_progress_events()
    test_experiment_compare_includes_running_and_delta_to_best()
    test_metric_glossary_has_plain_korean_examples()
    test_metric_help_ui_is_collapsible_and_click_targeted()
    test_resource_metric_uses_compact_font_class()
    test_experiment_compare_does_not_hide_lower_ranked_completed_runs()
    test_dashboard_registers_new_queued_experiments()
    test_process_tree_snapshot_aggregates_child_processes()
    test_system_cpu_snapshot_parses_top_output()
    test_gpu_snapshot_handles_unavailable_sampler()
    test_completed_snapshot_exposes_operational_columns()
    print("rank dashboard operational tests passed")
