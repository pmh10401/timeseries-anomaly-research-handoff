import importlib.util
import sqlite3
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "run_experiment_33_evalset_reconstruction_validation.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_module():
    spec = importlib.util.spec_from_file_location("experiment_33", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_blob(value):
    return np.asarray([value, value + 0.25], dtype=np.float32).tobytes()


def build_fixture_db(path):
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            univariate BOOLEAN NOT NULL,
            equal_length BOOLEAN NOT NULL,
            series_length INTEGER,
            has_missing BOOLEAN NOT NULL,
            train_normal_count INTEGER NOT NULL,
            train_anomaly_count INTEGER NOT NULL,
            train_total_count INTEGER NOT NULL,
            test_normal_count INTEGER NOT NULL,
            test_anomaly_count INTEGER NOT NULL,
            test_total_count INTEGER NOT NULL,
            total_normal_count INTEGER NOT NULL,
            total_anomaly_count INTEGER NOT NULL,
            total_count INTEGER NOT NULL
        );
        CREATE TABLE instances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER NOT NULL,
            split TEXT NOT NULL,
            instance_index INTEGER NOT NULL,
            label TEXT NOT NULL,
            values_blob BLOB NOT NULL,
            labels_blob BLOB
        );
        """
    )
    conn.execute(
        """
        INSERT INTO datasets (
            name, univariate, equal_length, series_length, has_missing,
            train_normal_count, train_anomaly_count, train_total_count,
            test_normal_count, test_anomaly_count, test_total_count,
            total_normal_count, total_anomaly_count, total_count
        )
        VALUES ('Toy_normal_A', 1, 1, 2, 0, 2, 0, 2, 52, 2, 54, 54, 2, 56)
        """
    )
    dataset_id = conn.execute("SELECT id FROM datasets").fetchone()[0]

    rows = []
    # Train normals.
    rows.append((dataset_id, "TRAIN", 0, "0", make_blob(1), b"\x00\x00"))
    rows.append((dataset_id, "TRAIN", 1, "0", make_blob(2), b"\x00\x00"))
    # Test normals: two exact train overlaps, one duplicate, and 49 clean normals.
    rows.append((dataset_id, "TEST", 0, "0", make_blob(1), b"\x00\x00"))
    rows.append((dataset_id, "TEST", 1, "0", make_blob(2), b"\x00\x00"))
    rows.append((dataset_id, "TEST", 2, "0", make_blob(10), b"\x00\x00"))
    rows.append((dataset_id, "TEST", 3, "0", make_blob(10), b"\x00\x00"))
    for offset in range(49):
        rows.append((dataset_id, "TEST", 4 + offset, "0", make_blob(100 + offset), b"\x00\x00"))
    rows.append((dataset_id, "TEST", 53, "1", make_blob(900), b"\x01\x01"))
    rows.append((dataset_id, "TEST", 54, "1", make_blob(901), b"\x01\x01"))
    conn.executemany(
        """
        INSERT INTO instances (dataset_id, split, instance_index, label, values_blob, labels_blob)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()


def test_strict_manifest_removes_train_overlap_and_test_duplicates(tmp_path):
    module = load_module()
    db_path = tmp_path / "fixture.sqlite"
    build_fixture_db(db_path)

    manifests = module.build_manifests(str(db_path), seed=20260707, dataset_limit=None)
    strict = [row for row in manifests if row["manifest_variant"] == "strict_unbalanced"]

    assert len(strict) == 1
    assert strict[0]["clean_test_normal_count"] == 50
    assert strict[0]["clean_test_anomaly_count"] == 2
    assert strict[0]["removed_train_overlap_normals"] == 2
    assert strict[0]["removed_test_duplicate_normals"] == 1
    assert strict[0]["is_eligible"] == 1


def test_balanced_manifest_keeps_exact_2pct_ratio(tmp_path):
    module = load_module()
    db_path = tmp_path / "fixture.sqlite"
    build_fixture_db(db_path)

    manifests = module.build_manifests(str(db_path), seed=20260707, dataset_limit=None)
    balanced = [row for row in manifests if row["manifest_variant"] == "balanced_2pct"]

    assert len(balanced) == 1
    assert balanced[0]["clean_test_anomaly_count"] == 1
    assert balanced[0]["clean_test_normal_count"] == 49
    assert balanced[0]["clean_test_total_count"] == 50
    assert balanced[0]["anomaly_rate"] == 0.02


def test_manifest_generation_is_deterministic(tmp_path):
    module = load_module()
    db_path = tmp_path / "fixture.sqlite"
    build_fixture_db(db_path)

    first = module.build_manifests(str(db_path), seed=20260707, dataset_limit=None)
    second = module.build_manifests(str(db_path), seed=20260707, dataset_limit=None)

    assert first == second


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        test_strict_manifest_removes_train_overlap_and_test_duplicates(tmp_path)
        test_balanced_manifest_keeps_exact_2pct_ratio(tmp_path)
        test_manifest_generation_is_deterministic(tmp_path)
    print("experiment 33 evalset reconstruction tests passed")
