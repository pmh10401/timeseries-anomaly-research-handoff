import argparse
import csv
import hashlib
import logging
import os
import random
import sqlite3
import warnings
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import scipy.stats
import torch
import torch.nn as nn
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from run_all_adaptive_cnn_multi_aug_contrastive import (
    AdaptiveConvVAE,
    augment_multi_phase,
)
from run_experiment_29_train_normal_threshold_calibration import train_false_positive_stats
from run_rank_ensemble_calibration import align_series_lengths, sanitize_series, z_normalize
from run_rank_threshold_calibration import top_k_oracle_f1


DATA_DIR = Path("/Users/minho/Documents/Dataset")
DB_PATH = DATA_DIR / "univariate_ts.db"
RNG_SEED = 20260707


def choose_acceleration_device(preferred="auto", mps_available=None, cuda_available=None):
    preferred = (preferred or "auto").lower()
    mps_available = torch.backends.mps.is_available() if mps_available is None else bool(mps_available)
    cuda_available = torch.cuda.is_available() if cuda_available is None else bool(cuda_available)
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "mps":
        return torch.device("mps") if mps_available else torch.device("cpu")
    if preferred == "cuda":
        return torch.device("cuda") if cuda_available else torch.device("cpu")
    if mps_available:
        return torch.device("mps")
    if cuda_available:
        return torch.device("cuda")
    return torch.device("cpu")


device = choose_acceleration_device(os.environ.get("MULTI_AUG_ROBUST_DEVICE", "auto"))


EXPERIMENT_SPECS = {
    "experiment_41_multi_aug_robust_baseline": {
        "label": "Experiment 41 - Multi-Aug robust baseline",
        "detail": DATA_DIR / "experiment_41_multi_aug_robust_baseline_results.csv",
        "summary": DATA_DIR / "experiment_41_multi_aug_robust_baseline_summary.csv",
        "log": DATA_DIR / "experiment_41_multi_aug_robust_baseline.log",
        "include_operational_metrics": False,
        "thresholds": [
            {"name": "percentile", "kind": "percentile"},
            {"name": "adaptive", "kind": "adaptive_distribution"},
            {"name": "evt", "kind": "evt_pot"},
        ],
    },
    "experiment_42_multi_aug_robust_operational": {
        "label": "Experiment 42 - Multi-Aug robust operational thresholds",
        "detail": DATA_DIR / "experiment_42_multi_aug_robust_operational_results.csv",
        "summary": DATA_DIR / "experiment_42_multi_aug_robust_operational_summary.csv",
        "log": DATA_DIR / "experiment_42_multi_aug_robust_operational.log",
        "include_operational_metrics": True,
        "thresholds": [
            {"name": "percentile", "kind": "percentile"},
            {"name": "adaptive", "kind": "adaptive_distribution"},
            {"name": "evt", "kind": "evt_pot"},
            {"name": "count_cap_1pct", "kind": "count_cap", "rate": 0.01},
            {"name": "count_cap_2pct", "kind": "count_cap", "rate": 0.02},
            {"name": "count_cap_3pct", "kind": "count_cap", "rate": 0.03},
            {"name": "adaptive_v0", "kind": "adaptive_count_cap"},
        ],
    },
}


def get_spec(exp_id):
    if exp_id not in EXPERIMENT_SPECS:
        raise SystemExit(f"Unknown robust Multi-Aug experiment: {exp_id}")
    spec = dict(EXPERIMENT_SPECS[exp_id])
    spec["id"] = exp_id
    return spec


def stable_seed(dataset_name):
    digest = hashlib.sha256(dataset_name.encode("utf-8")).hexdigest()
    return RNG_SEED + int(digest[:8], 16) % 1_000_000


def set_dataset_seed(dataset_name):
    seed = stable_seed(dataset_name)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_family(dataset_name):
    if "_normal_" in dataset_name:
        return dataset_name.rsplit("_normal_", 1)[0]
    return dataset_name


def sanitize_and_align_series(series, target_len):
    cleaned = [sanitize_series(np.asarray(item, dtype=np.float32)) for item in series]
    return align_series_lengths(cleaned, target_len).astype(np.float32)


def prepare_arrays(train_series, test_series, target_len):
    return (
        sanitize_and_align_series(train_series, target_len),
        sanitize_and_align_series(test_series, target_len),
    )


def load_dataset_record(dataset_name, db_path=DB_PATH):
    conn = sqlite3.connect(str(db_path))
    meta = conn.execute("SELECT series_length FROM datasets WHERE name = ?", (dataset_name,)).fetchone()
    metadata_len = int(meta[0]) if meta and meta[0] else 0
    train_rows = conn.execute(
        """
        SELECT i.values_blob
        FROM instances i
        JOIN datasets d ON i.dataset_id = d.id
        WHERE d.name = ? AND i.split = 'TRAIN'
        ORDER BY i.instance_index
        """,
        (dataset_name,),
    ).fetchall()
    test_rows = conn.execute(
        """
        SELECT i.values_blob, i.label
        FROM instances i
        JOIN datasets d ON i.dataset_id = d.id
        WHERE d.name = ? AND i.split = 'TEST'
        ORDER BY i.instance_index
        """,
        (dataset_name,),
    ).fetchall()
    conn.close()
    train_series = [np.frombuffer(row[0], dtype=np.float32) for row in train_rows]
    test_series = [np.frombuffer(row[0], dtype=np.float32) for row in test_rows]
    y_test = np.array([int(row[1]) for row in test_rows], dtype=np.int64)
    lengths = [len(item) for item in train_series + test_series]
    actual_median = int(round(float(np.median(lengths)))) if lengths else metadata_len
    target_len = metadata_len or actual_median
    target_len = max(8, int(target_len))
    return {
        "dataset_name": dataset_name,
        "family": parse_family(dataset_name),
        "metadata_len": metadata_len,
        "target_len": target_len,
        "actual_len_median": actual_median,
        "actual_len_min": int(min(lengths)) if lengths else target_len,
        "actual_len_max": int(max(lengths)) if lengths else target_len,
        "train_series": train_series,
        "test_series": test_series,
        "y_test": y_test,
    }


def prepare_multi_aug_dataset(X, target_size=500):
    n_rows = len(X)
    if n_rows < target_size:
        indices = np.random.choice(n_rows, target_size, replace=True)
        X_orig = X[indices]
    else:
        X_orig = X[:target_size]
    X_aug = np.array([augment_multi_phase(row) for row in X_orig], dtype=np.float32)
    return X_orig.astype(np.float32), X_aug


def train_multi_aug_scores(X_train, X_test, seq_len, epochs=10, batch_size=128, target_size=500, run_device=None):
    run_device = run_device or device
    original_train_size = len(X_train)
    max_beta = 0.15 / seq_len
    gamma = 0.8 / np.log(original_train_size + 2)
    X_train = z_normalize(X_train).astype(np.float32)
    X_test = z_normalize(X_test).astype(np.float32)
    X_train_orig, X_train_aug = prepare_multi_aug_dataset(X_train, target_size=target_size)
    X_train_orig = np.expand_dims(X_train_orig, axis=1)
    X_train_aug = np.expand_dims(X_train_aug, axis=1)
    X_test_ch = np.expand_dims(X_test, axis=1)
    train_dataset = TensorDataset(
        torch.tensor(X_train_orig, dtype=torch.float32),
        torch.tensor(X_train_aug, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = AdaptiveConvVAE(latent_dim=128, seq_len=seq_len).to(run_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(epochs):
        current_beta = min(max_beta, (epoch / max(1, epochs - 3)) * max_beta)
        for x_orig, x_aug in train_loader:
            x_orig = x_orig.to(run_device)
            x_aug = x_aug.to(run_device)
            optimizer.zero_grad()
            recon_orig, mu_orig, logvar_orig = model(x_orig)
            z_orig = model.reparameterize(mu_orig, logvar_orig)
            _, mu_aug, logvar_aug = model(x_aug)
            z_aug = model.reparameterize(mu_aug, logvar_aug)
            recon_loss = nn.functional.mse_loss(recon_orig, x_orig, reduction="mean")
            kl_loss = -0.5 * torch.mean(1 + logvar_orig - mu_orig.pow(2) - logvar_orig.exp())
            contrastive_loss = torch.mean(1.0 - nn.functional.cosine_similarity(z_orig, z_aug, dim=1))
            loss = recon_loss + current_beta * kl_loss + gamma * contrastive_loss
            loss.backward()
            optimizer.step()

    def score_array(X):
        X_ch = np.expand_dims(X, axis=1) if X.ndim == 2 else X
        scores = []
        loader = DataLoader(TensorDataset(torch.tensor(X_ch, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
        model.eval()
        with torch.no_grad():
            for (x_batch,) in loader:
                x_batch = x_batch.to(run_device)
                recon, mu, logvar = model(x_batch)
                recon_errs = ((x_batch - recon) ** 2).mean(dim=(1, 2))
                kl_errs = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                scores.extend((recon_errs + max_beta * kl_errs).cpu().numpy())
        return np.asarray(scores, dtype=np.float64)

    return score_array(X_train), score_array(X_test_ch)


def clean_scores(scores):
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return np.array([0.0], dtype=np.float64)
    return scores


def count_cap_threshold(scores, rate):
    scores = clean_scores(scores)
    cap = int(np.floor(float(rate) * len(scores)))
    cap = max(0, min(cap, len(scores) - 1))
    threshold = float(np.sort(scores)[len(scores) - cap - 1])
    return threshold, cap / max(1, len(scores)), cap


def original_thresholds(train_scores, q_target):
    train_scores = clean_scores(train_scores)
    percentile = float(np.percentile(train_scores, 100 * (1.0 - q_target)))
    train_skew = scipy.stats.skew(train_scores)
    if train_skew > 1.2:
        try:
            shape, loc, scale = scipy.stats.lognorm.fit(train_scores, floc=0)
            adaptive = float(scipy.stats.lognorm.ppf(1.0 - q_target, shape, loc, scale))
        except Exception:
            adaptive = percentile
    elif train_skew < 0.2:
        mu_fit, std_fit = scipy.stats.norm.fit(train_scores)
        adaptive = float(scipy.stats.norm.ppf(1.0 - q_target, mu_fit, std_fit))
    else:
        try:
            a_fit, loc_fit, scale_fit = scipy.stats.gamma.fit(train_scores, floc=0)
            adaptive = float(scipy.stats.gamma.ppf(1.0 - q_target, a_fit, loc_fit, scale_fit))
        except Exception:
            adaptive = percentile
    if not np.isfinite(adaptive) or adaptive <= 0:
        adaptive = percentile
    t = float(np.percentile(train_scores, 90))
    excesses = train_scores[train_scores > t] - t
    n = len(train_scores)
    nt = len(excesses)
    if nt > 10 and np.std(excesses) > 1e-12 and len(np.unique(np.round(excesses, 12))) > 2:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                c_fit, _, scale_fit = scipy.stats.genpareto.fit(excesses, floc=0)
            prob_excess = np.clip(1.0 - (q_target * n / nt), 0.90, 0.9999)
            evt = float(t + scipy.stats.genpareto.ppf(prob_excess, c_fit, loc=0, scale=scale_fit))
        except (Exception, RuntimeWarning):
            evt = adaptive
    else:
        evt = adaptive
    if not np.isfinite(evt) or evt <= 0:
        evt = adaptive
    return {"percentile": percentile, "adaptive": adaptive, "evt": evt}


def threshold_for_method(method, train_scores, q_target, test_size, originals=None):
    originals = originals or original_thresholds(train_scores, q_target)
    kind = method["kind"]
    if kind in {"percentile", "adaptive_distribution", "evt_pot"}:
        return originals[method["name"]], q_target, ""
    if kind == "count_cap":
        threshold, effective, cap = count_cap_threshold(train_scores, method["rate"])
        return threshold, effective, cap
    if kind == "adaptive_count_cap":
        rate = 0.03 if test_size > 50 else 0.02
        threshold, effective, cap = count_cap_threshold(train_scores, rate)
        return threshold, effective, cap
    raise ValueError(f"Unknown threshold method: {method}")


def score_metrics(y_true, test_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, test_scores)
    return {
        "auc_roc": roc_auc_score(y_true, test_scores),
        "auc_pr": auc(recall, precision),
        "oracle_f1": top_k_oracle_f1(y_true, test_scores),
    }


def evaluate_threshold(y_true, test_scores, threshold):
    preds = (test_scores > threshold).astype(np.int64)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    return {
        "predicted_count": int(preds.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "f1": f1_score(y_true, preds, zero_division=0),
        "f1_evt": f1_score(y_true, preds, zero_division=0),
    }


def run_dataset(dataset_name, spec, args):
    set_dataset_seed(dataset_name)
    record = load_dataset_record(dataset_name, args.db_path)
    y_test = record["y_test"]
    if len(record["train_series"]) == 0 or len(record["test_series"]) == 0 or len(np.unique(y_test)) < 2:
        return []
    X_train, X_test = prepare_arrays(record["train_series"], record["test_series"], record["target_len"])
    train_scores, test_scores = train_multi_aug_scores(
        X_train,
        X_test,
        record["target_len"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        target_size=args.target_size,
        run_device=choose_acceleration_device(args.device),
    )
    metrics = score_metrics(y_test, test_scores)
    q_target = max(0.001, 1.0 / max(1, len(y_test)))
    originals = original_thresholds(train_scores, q_target)
    rows = []
    for method in spec["thresholds"]:
        threshold, q_effective, cap_target = threshold_for_method(
            method,
            train_scores,
            q_target,
            len(y_test),
            originals=originals,
        )
        train_exceed_count, train_exceed_rate = train_false_positive_stats(train_scores, threshold)
        rows.append(
            {
                "experiment_id": spec["id"],
                "dataset_name": dataset_name,
                "family": record["family"],
                "config_name": "multi_aug_robust_vae",
                "threshold_method": method["name"],
                "threshold_family": method["kind"],
                "sequence_length": record["target_len"],
                "metadata_len": record["metadata_len"],
                "actual_len_min": record["actual_len_min"],
                "actual_len_median": record["actual_len_median"],
                "actual_len_max": record["actual_len_max"],
                "train_count": len(record["train_series"]),
                "test_size": len(y_test),
                "anomaly_count": int(np.sum(y_test)),
                "target_q": q_target,
                "q_effective": q_effective,
                "cap_target": cap_target,
                "threshold": threshold,
                "train_exceed_count": train_exceed_count,
                "train_exceed_rate": train_exceed_rate,
                **metrics,
                **evaluate_threshold(y_test, test_scores, threshold),
            }
        )
    return rows


def load_dataset_names(db_path=DB_PATH):
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        """
        SELECT name FROM datasets
        WHERE name NOT IN ('CornellWhaleChallenge', 'Wafer_normal_1')
        ORDER BY name
        """
    ).fetchall()
    conn.close()
    return [row[0] for row in rows]


def append_rows(path, rows, fieldnames):
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def read_existing_detail_rows(path):
    if not path.exists() or path.stat().st_size == 0:
        return [], None
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def completed_dataset_names(rows, exp_id=None):
    completed = set()
    for row in rows:
        if exp_id is not None and row.get("experiment_id", exp_id) != exp_id:
            continue
        dataset_name = row.get("dataset_name")
        if dataset_name:
            completed.add(dataset_name)
    return completed


def assert_detail_dataset_coverage(path, expected_dataset_names, exp_id, logger):
    disk_rows, _ = read_existing_detail_rows(path)
    disk_dataset_names = completed_dataset_names(disk_rows, exp_id)
    missing = [name for name in expected_dataset_names if name not in disk_dataset_names]
    missing_path = path.with_name(f"{path.stem}_missing_datasets.csv")
    if missing:
        with missing_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["experiment_id", "dataset_name"])
            writer.writeheader()
            writer.writerows({"experiment_id": exp_id, "dataset_name": name} for name in missing)
        logger.warning(
            "%s detail CSV is missing %d/%d datasets after run; wrote %s. Queue will continue.",
            exp_id,
            len(missing),
            len(expected_dataset_names),
            missing_path,
        )
    elif missing_path.exists():
        missing_path.unlink()
    return disk_rows


def repair_missing_datasets(exp_id, spec, args, expected_dataset_names, fieldnames, logger, max_attempts=1):
    detail_rows, existing_fieldnames = read_existing_detail_rows(spec["detail"])
    fieldnames = fieldnames or existing_fieldnames
    for attempt in range(1, int(max_attempts) + 1):
        completed = completed_dataset_names(detail_rows, exp_id)
        missing = [name for name in expected_dataset_names if name not in completed]
        if not missing:
            break
        logger.warning(
            "%s repairing %d missing datasets before queue continues. attempt=%d",
            exp_id,
            len(missing),
            attempt,
        )
        for dataset_name in missing:
            try:
                rows = run_dataset(dataset_name, spec, args)
            except Exception as exc:
                logger.error("Repair failed for dataset %s: %s", dataset_name, exc, exc_info=True)
                rows = []
            if not rows:
                logger.warning("Repair produced no rows for dataset %s", dataset_name)
                continue
            detail_rows.extend(rows)
            fieldnames = fieldnames or list(rows[0].keys())
            append_rows(spec["detail"], rows, fieldnames)
    return read_existing_detail_rows(spec["detail"])[0]


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summary = []
    keys = sorted({(row["config_name"], row["threshold_method"]) for row in rows})
    for config_name, method in keys:
        subset = [row for row in rows if row["config_name"] == config_name and row["threshold_method"] == method]
        f1s = [float(row["f1"]) for row in subset]
        by_family = defaultdict(list)
        for row in subset:
            by_family[row["family"]].append(float(row["f1"]))
        family_means = [float(np.mean(values)) for values in by_family.values()]
        summary.append(
            {
                "config_name": config_name,
                "threshold_method": method,
                "num_datasets": len(subset),
                "num_families": len(by_family),
                "mean_auc_roc": float(np.mean([float(row["auc_roc"]) for row in subset])),
                "mean_auc_pr": float(np.mean([float(row["auc_pr"]) for row in subset])),
                "mean_f1": float(np.mean(f1s)),
                "median_f1": float(np.median(f1s)),
                "zero_f1_count": sum(1 for value in f1s if value == 0),
                "family_macro_f1": float(np.mean(family_means)) if family_means else 0.0,
                "mean_predicted_count": float(np.mean([int(row["predicted_count"]) for row in subset])),
                "mean_tp": float(np.mean([int(row["tp"]) for row in subset])),
                "mean_fp": float(np.mean([int(row["fp"]) for row in subset])),
                "mean_fn": float(np.mean([int(row["fn"]) for row in subset])),
                "mean_train_exceed_rate": float(np.mean([float(row["train_exceed_rate"]) for row in subset])),
                "mean_oracle_f1": float(np.mean([float(row["oracle_f1"]) for row in subset])),
            }
        )
    return sorted(summary, key=lambda row: (row["mean_f1"], row["mean_auc_pr"]), reverse=True)


def make_logger(exp_id, log_path):
    logger = logging.getLogger(exp_id)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(logging.FileHandler(log_path))
    logger.addHandler(logging.StreamHandler())
    for handler in logger.handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    return logger


def run_experiment(exp_id, args):
    spec = get_spec(exp_id)
    logger = make_logger(exp_id, spec["log"])
    if not args.keep_existing:
        for path in [spec["detail"], spec["summary"]]:
            if path.exists():
                path.unlink()
    detail_rows = []
    fieldnames = None
    if args.keep_existing:
        detail_rows, fieldnames = read_existing_detail_rows(spec["detail"])

    dataset_names = load_dataset_names(args.db_path)
    if args.dataset_limit is not None:
        dataset_names = dataset_names[: args.dataset_limit]
    expected_dataset_names = list(dataset_names)
    if args.keep_existing and detail_rows:
        completed_names = completed_dataset_names(detail_rows, exp_id)
        dataset_names = [name for name in dataset_names if name not in completed_names]
    run_device = choose_acceleration_device(args.device)
    logger.info(
        "Starting %s on %d datasets. device=%s epochs=%d existing_rows=%d",
        exp_id,
        len(dataset_names),
        run_device,
        args.epochs,
        len(detail_rows),
    )
    if not dataset_names and detail_rows:
        final_rows = assert_detail_dataset_coverage(spec["detail"], expected_dataset_names, exp_id, logger)
        write_csv(spec["summary"], summarize(final_rows))
        logger.info("%s resume found no remaining datasets.", exp_id)
        return
    for idx, dataset_name in enumerate(dataset_names, 1):
        try:
            rows = run_dataset(dataset_name, spec, args)
        except Exception as exc:
            logger.error("Error evaluating dataset %s: %s", dataset_name, exc, exc_info=True)
            rows = []
        if not rows:
            logger.warning("No rows produced for dataset %s", dataset_name)
        if rows:
            detail_rows.extend(rows)
            fieldnames = fieldnames or list(rows[0].keys())
            append_rows(spec["detail"], rows, fieldnames)
        if idx % 25 == 0 or idx == len(dataset_names):
            summary_rows = summarize(detail_rows) if detail_rows else []
            write_csv(spec["summary"], summary_rows)
            best = summary_rows[0] if summary_rows else None
            if best:
                logger.info(
                    "Progress: [%4d/%4d] rows=%d | best=%s meanF1=%.4f medianF1=%.4f fp=%.2f",
                    idx,
                    len(dataset_names),
                    len(detail_rows),
                    f"{best['config_name']}/{best['threshold_method']}",
                    best["mean_f1"],
                    best["median_f1"],
                    best["mean_fp"],
                )
    final_rows = repair_missing_datasets(exp_id, spec, args, expected_dataset_names, fieldnames, logger)
    final_rows = assert_detail_dataset_coverage(spec["detail"], expected_dataset_names, exp_id, logger)
    write_csv(spec["summary"], summarize(final_rows))
    logger.info("%s finished.", exp_id)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run robust Multi-Aug VAE experiments.")
    parser.add_argument("experiment_id", choices=sorted(EXPERIMENT_SPECS))
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--dataset-limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("MULTI_AUG_ROBUST_EPOCHS", "10")))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target-size", type=int, default=500)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default=os.environ.get("MULTI_AUG_ROBUST_DEVICE", "auto"))
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args(argv)


def main_for_experiment(exp_id, argv=None):
    args = parse_args([exp_id] + list(argv or []))
    run_experiment(exp_id, args)


def main(argv=None):
    args = parse_args(argv)
    run_experiment(args.experiment_id, args)


if __name__ == "__main__":
    main()
