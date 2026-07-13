#!/usr/bin/env python3
import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score

from run_balanced_improvement_experiment import count_cap_threshold
from run_experiment_29_train_normal_threshold_calibration import train_false_positive_stats
from run_model_hard_research_experiments import HARD_SCORE_FAMILIES, read_difficulty_rows
from run_original_improvement_experiment import DATA_DIR, DB_PATH, load_original_record, target_len_for_record
from run_rank_ensemble_calibration import align_series_lengths, z_normalize
from run_rank_threshold_calibration import top_k_oracle_f1


EXPERIMENT_ID = "experiment_79_vae_epoch_sweep_guarded"
DETAIL_PATH = DATA_DIR / f"{EXPERIMENT_ID}_results.csv"
SUMMARY_PATH = DATA_DIR / f"{EXPERIMENT_ID}_summary.csv"
LOG_PATH = DATA_DIR / f"{EXPERIMENT_ID}_stdout.log"
EPOCHS = [10, 30, 60]
THRESHOLDS = [("count_cap_1pct", 0.01), ("count_cap_2pct", 0.02), ("count_cap_3pct", 0.03)]


def torch_setup():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return torch, nn, DataLoader, TensorDataset, device


class ConvAutoEncoderFactory:
    def __init__(self, nn):
        self.nn = nn

    def build(self):
        nn = self.nn

        class ConvAutoEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv1d(1, 16, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.AvgPool1d(2),
                    nn.Conv1d(16, 32, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.AvgPool1d(2),
                )
                self.decoder = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
                    nn.Conv1d(32, 16, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
                    nn.Conv1d(16, 1, kernel_size=5, padding=2),
                )

            def forward(self, x):
                out = self.decoder(self.encoder(x))
                if out.shape[-1] < x.shape[-1]:
                    out = nn.functional.pad(out, (0, x.shape[-1] - out.shape[-1]))
                return out[..., : x.shape[-1]]

        return ConvAutoEncoder()


def target_datasets(limit=None):
    rows = [
        row
        for row in read_difficulty_rows()
        if row["family"] in HARD_SCORE_FAMILIES and int(row["train_count"]) >= 30
    ]
    rows = sorted(rows, key=lambda row: (row["family"], row["dataset_name"]))
    if limit is not None:
        rows = rows[:limit]
    return rows


def score_metrics(y_true, test_scores):
    try:
        auc_roc = roc_auc_score(y_true, test_scores)
    except ValueError:
        auc_roc = 0.5
    precision, recall, _ = precision_recall_curve(y_true, test_scores)
    return {
        "auc_roc": float(auc(recall, precision) * 0 + auc_roc),
        "auc_pr": float(auc(recall, precision)),
        "oracle_f1": float(top_k_oracle_f1(y_true, test_scores)),
    }


def evaluate_threshold(y_true, test_scores, threshold, metrics):
    preds = (np.asarray(test_scores) > threshold).astype(np.int64)
    return {
        "predicted_count": int(preds.sum()),
        "tp": int(((preds == 1) & (y_true == 1)).sum()),
        "fp": int(((preds == 1) & (y_true == 0)).sum()),
        "fn": int(((preds == 0) & (y_true == 1)).sum()),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        **metrics,
    }


def reconstruction_scores(torch, DataLoader, TensorDataset, model, X, batch_size, device):
    tensor = torch.tensor(X[:, None, :], dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)
    scores = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            err = ((recon - batch) ** 2).mean(dim=(1, 2))
            scores.extend(err.detach().cpu().numpy().tolist())
    return np.asarray(scores, dtype=np.float64)


def train_and_score(record, epochs, batch_size):
    torch, nn, DataLoader, TensorDataset, device = torch_setup()
    target_len = min(max(32, target_len_for_record(record, "actual_median")), 512)
    X_train = align_series_lengths(record["train_series"], target_len)
    X_test = align_series_lengths(record["test_series"], target_len)
    X_train = z_normalize(X_train).astype(np.float32)
    X_test = z_normalize(X_test).astype(np.float32)
    factory = ConvAutoEncoderFactory(nn)
    model = factory.build().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    tensor = torch.tensor(X_train[:, None, :], dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(int(epochs)):
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = ((recon - batch) ** 2).mean()
            loss.backward()
            optimizer.step()
    train_scores = reconstruction_scores(torch, DataLoader, TensorDataset, model, X_train, batch_size, device)
    test_scores = reconstruction_scores(torch, DataLoader, TensorDataset, model, X_test, batch_size, device)
    return train_scores, test_scores, target_len, str(device)


def run_dataset(row, batch_size):
    record = load_original_record(row["dataset_name"], DB_PATH)
    y_test = np.asarray(record["y_test"], dtype=np.int64)
    if len(record["train_series"]) == 0 or len(record["test_series"]) == 0 or len(np.unique(y_test)) < 2:
        return []
    rows = []
    for epochs in EPOCHS:
        train_scores, test_scores, target_len, device = train_and_score(record, epochs, batch_size)
        metrics = score_metrics(y_test, test_scores)
        for method, rate in THRESHOLDS:
            threshold, q_effective, cap_target = count_cap_threshold(train_scores, rate)
            train_exceed_count, train_exceed_rate = train_false_positive_stats(train_scores, threshold)
            rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "dataset_name": record["dataset_name"],
                    "family": record["family"],
                    "config_name": f"conv_ae_epoch_{epochs}",
                    "selector_name": f"conv_ae_epoch_{epochs}",
                    "score_family": "conv_autoencoder_reconstruction",
                    "threshold_method": method,
                    "device": device,
                    "sequence_length": target_len,
                    "train_count": len(record["train_series"]),
                    "test_size": len(y_test),
                    "anomaly_count": int(np.sum(y_test)),
                    "train_score_count": len(train_scores),
                    "threshold": threshold,
                    "q_effective": q_effective,
                    "cap_target": cap_target,
                    "train_exceed_count": train_exceed_count,
                    "train_exceed_rate": train_exceed_rate,
                    **evaluate_threshold(y_test, test_scores, threshold, metrics),
                }
            )
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    out = []
    keys = sorted({(row["config_name"], row["threshold_method"]) for row in rows})
    for config_name, method in keys:
        subset = [row for row in rows if row["config_name"] == config_name and row["threshold_method"] == method]
        by_family = defaultdict(list)
        for row in subset:
            by_family[row["family"]].append(float(row["f1"]))
        f1s = [float(row["f1"]) for row in subset]
        out.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "config_name": config_name,
                "selector_name": config_name,
                "score_family": subset[0]["score_family"],
                "threshold_method": method,
                "num_datasets": len(subset),
                "num_families": len(by_family),
                "mean_auc_roc": float(np.mean([float(row["auc_roc"]) for row in subset])),
                "mean_auc_pr": float(np.mean([float(row["auc_pr"]) for row in subset])),
                "mean_f1": float(np.mean(f1s)),
                "median_f1": float(np.median(f1s)),
                "p25_f1": float(np.percentile(f1s, 25)),
                "zero_f1_count": sum(1 for value in f1s if value == 0),
                "ge_0_5_count": sum(1 for value in f1s if value >= 0.5),
                "family_macro_f1": float(np.mean([np.mean(values) for values in by_family.values()])),
                "mean_predicted_count": float(np.mean([int(row["predicted_count"]) for row in subset])),
                "mean_anomaly_count": float(np.mean([int(row["anomaly_count"]) for row in subset])),
                "mean_tp": float(np.mean([int(row["tp"]) for row in subset])),
                "mean_fp": float(np.mean([int(row["fp"]) for row in subset])),
                "mean_fn": float(np.mean([int(row["fn"]) for row in subset])),
                "mean_train_exceed_rate": float(np.mean([float(row["train_exceed_rate"]) for row in subset])),
                "mean_oracle_f1": float(np.mean([float(row["oracle_f1"]) for row in subset])),
            }
        )
    return sorted(out, key=lambda item: (item["mean_f1"], item["mean_auc_pr"]), reverse=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 79 VAE/AE epoch sweep guarded probe")
    parser.add_argument("--dataset-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    for path in [DETAIL_PATH, SUMMARY_PATH]:
        if path.exists():
            path.unlink()
    targets = target_datasets(args.dataset_limit)
    rows = []
    with LOG_PATH.open("w") as log:
        log.write(f"{EXPERIMENT_ID} starting datasets={len(targets)} epochs={EPOCHS}\n")
        for pos, row in enumerate(targets, 1):
            try:
                rows.extend(run_dataset(row, args.batch_size))
            except Exception as exc:
                log.write(f"ERROR {row['dataset_name']}: {exc}\n")
            if pos % 5 == 0 or pos == len(targets):
                write_csv(DETAIL_PATH, rows)
                write_csv(SUMMARY_PATH, summarize(rows))
                msg = f"{EXPERIMENT_ID} progress {pos}/{len(targets)} rows={len(rows)}"
                print(msg, flush=True)
                log.write(msg + "\n")
    write_csv(DETAIL_PATH, rows)
    write_csv(SUMMARY_PATH, summarize(rows))
    print(f"{EXPERIMENT_ID} finished. datasets={len(targets)} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
