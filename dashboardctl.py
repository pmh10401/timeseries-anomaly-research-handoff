#!/usr/bin/env python3
"""Single-process controller for the read-only rank experiment dashboard."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
DATA_DIR = Path("/Users/minho/Documents/Dataset")
PID_PATH = DATA_DIR / "rank_dashboard_server.pid"
LOG_PATH = DATA_DIR / "rank_dashboard_server.log"
SERVER_SCRIPT = ROOT / "serve_rank_dashboard.py"
PORT = int(os.environ.get("RANK_DASHBOARD_PORT", "8765"))


def read_pid():
    try:
        return int(PID_PATH.read_text().strip())
    except (OSError, ValueError):
        return None


def command_for_pid(pid):
    try:
        return subprocess.check_output(["ps", "-p", str(pid), "-o", "command="], text=True).strip()
    except subprocess.CalledProcessError:
        return ""


def dashboard_pid():
    pid = read_pid()
    if not pid:
        return None
    command = command_for_pid(pid)
    return pid if "serve_rank_dashboard.py" in command else None


def status():
    pid = dashboard_pid()
    payload = {"pid": pid, "port": PORT, "reachable": False}
    try:
        with urlopen(f"http://127.0.0.1:{PORT}/api/status", timeout=10) as response:
            body = json.loads(response.read())
        payload.update(
            {
                "reachable": True,
                "running": body.get("running"),
                "active_experiment": body.get("active_experiment"),
                "runtime_health": body.get("runtime_health"),
            }
        )
    except Exception as exc:
        payload["error"] = str(exc)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["reachable"] else 1


def stop():
    pid = dashboard_pid()
    if not pid:
        print("dashboard not running")
        return 0
    os.kill(pid, signal.SIGTERM)
    for _ in range(30):
        if not dashboard_pid():
            break
        time.sleep(0.1)
    if dashboard_pid():
        raise SystemExit(f"dashboard pid {pid} did not stop")
    print(f"dashboard stopped pid={pid}")
    return 0


def start():
    pid = dashboard_pid()
    if pid:
        print(f"dashboard already running pid={pid}")
        return 0
    with LOG_PATH.open("a") as log_file:
        subprocess.Popen(
            [sys.executable, str(SERVER_SCRIPT)],
            cwd=str(ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    for _ in range(30):
        if dashboard_pid():
            print(f"dashboard started pid={dashboard_pid()} port={PORT}")
            return 0
        time.sleep(0.1)
    raise SystemExit("dashboard failed to start; inspect rank_dashboard_server.log")


def main():
    parser = argparse.ArgumentParser(description="Control the local rank experiment dashboard")
    parser.add_argument("command", choices=("start", "stop", "restart", "status"))
    args = parser.parse_args()
    if args.command == "start":
        return start()
    if args.command == "stop":
        return stop()
    if args.command == "restart":
        stop()
        return start()
    return status()


if __name__ == "__main__":
    raise SystemExit(main())
