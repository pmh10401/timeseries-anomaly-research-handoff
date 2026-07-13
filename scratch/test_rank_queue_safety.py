import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_rank_experiments_sequential as queue


def test_reconcile_running_state_clears_dead_pid_without_touching_queue():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        old_state_path = queue.STATE_PATH
        old_log_path = queue.LOG_PATH
        try:
            queue.STATE_PATH = tmp_path / "state.json"
            queue.LOG_PATH = tmp_path / "queue.log"
            state = {
                "completed": [],
                "queue": ["experiment_44_classical_embedding_baselines"],
                "history": [],
                "running": {
                    "id": "experiment_44_classical_embedding_baselines",
                    "pid": 99999998,
                    "child_pid": 99999999,
                    "started_at": "2026-07-07 00:00:00",
                },
            }
            queue.save_state(state)

            reconciled = queue.reconcile_running_state(queue.load_state())

            assert reconciled["running"] is None
            assert reconciled["queue"] == ["experiment_44_classical_embedding_baselines"]
            assert "STALE running state cleared" in queue.LOG_PATH.read_text()
        finally:
            queue.STATE_PATH = old_state_path
            queue.LOG_PATH = old_log_path


def test_reconcile_running_state_keeps_live_pid():
    state = {
        "completed": [],
        "queue": [],
        "history": [],
        "running": {"id": "x", "pid": os.getpid(), "started_at": "2026-07-07 00:00:00"},
    }

    reconciled = queue.reconcile_running_state(dict(state))

    assert reconciled["running"] == state["running"]


if __name__ == "__main__":
    test_reconcile_running_state_clears_dead_pid_without_touching_queue()
    test_reconcile_running_state_keeps_live_pid()
    print("rank queue safety tests passed")
