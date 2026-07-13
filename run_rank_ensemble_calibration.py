import csv
import logging
import os
import sqlite3

import numpy as np
import scipy.stats
import torch
import torch.nn as nn
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler("/Users/minho/Documents/Dataset/rank_ensemble_calibration_run.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("RankEnsembleCalibration")

DB_PATH = "/Users/minho/Documents/Dataset/univariate_ts.db"
DETAIL_OUT_PATH = "/Users/minho/Documents/Dataset/vae_results_rank_ensemble_calibration_train_evt.csv"
SUMMARY_OUT_PATH = "/Users/minho/Documents/Dataset/vae_results_rank_ensemble_calibration_train_evt_summary.csv"

WEIGHT_CONFIGS = {
    "ma50_fused30_hybrid20": {"multi_aug": 0.50, "fused": 0.30, "hybrid": 0.20},
    "equal": {"multi_aug": 1 / 3, "fused": 1 / 3, "hybrid": 1 / 3},
    "ma40_fused40_hybrid20": {"multi_aug": 0.40, "fused": 0.40, "hybrid": 0.20},
    "ma60_fused20_hybrid20": {"multi_aug": 0.60, "fused": 0.20, "hybrid": 0.20},
    "ma45_fused25_hybrid30": {"multi_aug": 0.45, "fused": 0.25, "hybrid": 0.30},
}

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class AdaptiveConvVAE(nn.Module):
    def __init__(self, latent_dim=128, seq_len=152):
        super().__init__()
        self.latent_dim = latent_dim
        if seq_len >= 256:
            self.kernel_size = 15
            self.padding = 7
            self.channels = [1, 16, 32, 64, 128]
            self.trans_channels = [128, 64, 32, 16, 1]
        elif seq_len >= 128:
            self.kernel_size = 9
            self.padding = 4
            self.channels = [1, 16, 32, 64]
            self.trans_channels = [64, 32, 16, 1]
        elif seq_len >= 64:
            self.kernel_size = 5
            self.padding = 2
            self.channels = [1, 16, 32]
            self.trans_channels = [32, 16, 1]
        else:
            self.kernel_size = 3
            self.padding = 1
            self.channels = [1, 16]
            self.trans_channels = [16, 1]

        self.last_channel = self.channels[-1]
        self.num_layers = len(self.channels) - 1

        enc_layers = []
        for i in range(self.num_layers):
            enc_layers.append(
                nn.Conv1d(
                    self.channels[i],
                    self.channels[i + 1],
                    kernel_size=self.kernel_size,
                    stride=2,
                    padding=self.padding,
                )
            )
            enc_layers.append(nn.ReLU(True))
        self.enc_conv = nn.Sequential(*enc_layers)

        with torch.no_grad():
            dummy_output = self.enc_conv(torch.zeros(1, 1, seq_len))
            self.conv_out_len = dummy_output.size(2)
            self.flat_dim = self.last_channel * self.conv_out_len

        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, self.flat_dim)

        dec_layers = []
        trans_kernel = self.kernel_size + 1
        for i in range(self.num_layers):
            dec_layers.append(
                nn.ConvTranspose1d(
                    self.trans_channels[i],
                    self.trans_channels[i + 1],
                    kernel_size=trans_kernel,
                    stride=2,
                    padding=self.padding,
                )
            )
            if i < self.num_layers - 1:
                dec_layers.append(nn.ReLU(True))
        self.dec_conv = nn.Sequential(*dec_layers)

    def encode(self, x):
        h = self.enc_conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z, target_len):
        h = self.dec_fc(z)
        h = h.view(h.size(0), self.last_channel, self.conv_out_len)
        x_recon = self.dec_conv(h)
        if x_recon.size(2) != target_len:
            x_recon = nn.functional.interpolate(x_recon, size=target_len, mode="linear", align_corners=False)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, x.size(2)), mu, logvar


def sanitize_series(x):
    x = np.asarray(x, dtype=np.float32)
    finite_mask = np.isfinite(x)
    if finite_mask.all():
        return x
    if not finite_mask.any():
        return np.zeros_like(x, dtype=np.float32)
    idx = np.arange(len(x))
    cleaned = x.copy()
    cleaned[~finite_mask] = np.interp(idx[~finite_mask], idx[finite_mask], x[finite_mask])
    return cleaned.astype(np.float32)


def resample_series(x, target_len):
    if len(x) == target_len:
        return x.astype(np.float32, copy=False)
    if len(x) == 0:
        return np.zeros(target_len, dtype=np.float32)
    source_grid = np.linspace(0.0, 1.0, len(x))
    target_grid = np.linspace(0.0, 1.0, target_len)
    return np.interp(target_grid, source_grid, x).astype(np.float32)


def align_series_lengths(series_list, target_len):
    return np.stack([resample_series(x, target_len) for x in series_list]).astype(np.float32)


def load_dataset_data(dataset_name):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT series_length FROM datasets WHERE name = ?", (dataset_name,))
    meta_row = cursor.fetchone()
    target_len = int(meta_row[0]) if meta_row and meta_row[0] else None

    cursor.execute(
        """
        SELECT i.values_blob
        FROM instances i
        JOIN datasets d ON i.dataset_id = d.id
        WHERE d.name = ? AND i.split = 'TRAIN'
        ORDER BY i.instance_index
        """,
        (dataset_name,),
    )
    X_train = [sanitize_series(np.frombuffer(row[0], dtype=np.float32)) for row in cursor.fetchall()]

    cursor.execute(
        """
        SELECT i.values_blob, i.label
        FROM instances i
        JOIN datasets d ON i.dataset_id = d.id
        WHERE d.name = ? AND i.split = 'TEST'
        ORDER BY i.instance_index
        """,
        (dataset_name,),
    )
    test_rows = cursor.fetchall()
    X_test = [sanitize_series(np.frombuffer(row[0], dtype=np.float32)) for row in test_rows]
    y_test = np.array([int(row[1]) for row in test_rows])
    conn.close()

    if target_len is None:
        lengths = [len(x) for x in X_train + X_test]
        target_len = int(np.median(lengths)) if lengths else 0
    return align_series_lengths(X_train, target_len), align_series_lengths(X_test, target_len), y_test


def z_normalize(X):
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std


def augment_single_series(x):
    aug_x = x.copy()
    aug_type = np.random.choice(["jitter", "scale", "both"])
    if aug_type in ["jitter", "both"]:
        aug_x += np.random.normal(0, 0.03, size=len(x))
    if aug_type in ["scale", "both"]:
        aug_x *= np.random.uniform(0.9, 1.1)
    return aug_x


def time_warp(x):
    L = len(x)
    t = np.linspace(0, 1, L)
    warp_t = t + np.sin(2 * np.pi * t) * 0.08 * np.random.uniform(-1, 1)
    warp_t = np.clip(warp_t, 0, 1)
    if warp_t[-1] == warp_t[0]:
        return x
    warp_t = (warp_t - warp_t[0]) / (warp_t[-1] - warp_t[0])
    return np.interp(warp_t * (L - 1), np.arange(L), x)


def permute_segments(x, num_segments=4):
    L = len(x)
    if L < num_segments * 2:
        return x
    segment_len = L // num_segments
    segments = [x[i * segment_len : (i + 1) * segment_len] for i in range(num_segments)]
    leftover = x[num_segments * segment_len :]
    if len(leftover) > 0:
        segments.append(leftover)
    np.random.shuffle(segments)
    return np.concatenate(segments)[:L]


def augment_multi_phase(x):
    aug_x = x.copy()
    aug_type = np.random.choice(["jitter", "scale", "warp", "permute", "mix"])
    if aug_type == "jitter":
        aug_x += np.random.normal(0, 0.03, size=len(x))
    elif aug_type == "scale":
        aug_x *= np.random.uniform(0.9, 1.1)
    elif aug_type == "warp":
        aug_x = time_warp(aug_x)
    elif aug_type == "permute":
        aug_x = permute_segments(aug_x)
    elif aug_type == "mix":
        aug_x = permute_segments(aug_x)
        aug_x += np.random.normal(0, 0.02, size=len(x))
    return aug_x


def prepare_paired_dataset(X, mode, target_size=500):
    N = len(X)
    if N < target_size:
        indices = np.random.choice(N, target_size, replace=True)
        X_orig = X[indices]
    else:
        X_orig = X[:target_size]
    if mode == "multi_aug":
        X_aug = np.array([augment_multi_phase(x) for x in X_orig])
    else:
        X_aug = np.array([augment_single_series(x) for x in X_orig])
    return X_orig, X_aug


def augment_time_series(X, target_size=500):
    if len(X) >= target_size:
        return X[:target_size]
    augmented = [augment_single_series(X[np.random.randint(0, len(X))]) for _ in range(target_size - len(X))]
    return np.vstack([X, np.array(augmented)])


def train_contrastive_scores(X_train, X_test, mode, epochs=10, batch_size=128):
    original_train_size, seq_len = X_train.shape
    max_beta = 0.15 / seq_len
    gamma = 0.8 / np.log(original_train_size + 2)
    X_orig, X_aug = prepare_paired_dataset(X_train, mode=mode, target_size=500)
    train_dataset = TensorDataset(
        torch.tensor(np.expand_dims(X_orig, axis=1), dtype=torch.float32),
        torch.tensor(np.expand_dims(X_aug, axis=1), dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = AdaptiveConvVAE(latent_dim=128, seq_len=seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(epochs):
        current_beta = min(max_beta, (epoch / max(1, epochs - 3)) * max_beta)
        for x_orig, x_aug in train_loader:
            x_orig = x_orig.to(device)
            x_aug = x_aug.to(device)
            optimizer.zero_grad()
            recon_orig, mu_orig, logvar_orig = model(x_orig)
            _, mu_aug, _ = model(x_aug)
            recon_loss = nn.functional.mse_loss(recon_orig, x_orig, reduction="mean")
            kl_loss = -0.5 * torch.mean(1 + logvar_orig - mu_orig.pow(2) - logvar_orig.exp())
            contrastive_loss = torch.mean(1.0 - nn.functional.cosine_similarity(mu_orig, mu_aug, dim=1))
            loss = recon_loss + current_beta * kl_loss + gamma * contrastive_loss
            loss.backward()
            optimizer.step()
    return evaluate_mse_scores(model, X_train, X_test, max_beta, batch_size=batch_size)


def train_mse_scores(X_train, X_test, epochs=10, batch_size=128, beta=0.001):
    seq_len = X_train.shape[1]
    X_train_aug = augment_time_series(X_train, target_size=500)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(np.expand_dims(X_train_aug, axis=1), dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
    )
    model = AdaptiveConvVAE(latent_dim=128, seq_len=seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        for (x_batch,) in train_loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x_batch)
            recon_loss = nn.functional.mse_loss(recon, x_batch, reduction="mean")
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kl_loss
            loss.backward()
            optimizer.step()
    return evaluate_mse_scores(model, X_train, X_test, beta, batch_size=batch_size)


def evaluate_mse_scores(model, X_train, X_test, beta, batch_size=128):
    def score_array(X):
        out = []
        loader = DataLoader(
            TensorDataset(torch.tensor(np.expand_dims(X, axis=1), dtype=torch.float32)),
            batch_size=batch_size,
            shuffle=False,
        )
        model.eval()
        with torch.no_grad():
            for (x_batch,) in loader:
                x_batch = x_batch.to(device)
                recon, mu, logvar = model(x_batch)
                recon_errs = ((x_batch - recon) ** 2).mean(dim=(1, 2))
                kl_errs = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                out.extend((recon_errs + beta * kl_errs).cpu().numpy())
        return np.array(out)

    return score_array(X_train), score_array(X_test)


def percentile_ranks(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) <= 1:
        return np.zeros_like(scores)
    order = np.argsort(np.argsort(scores, kind="mergesort"), kind="mergesort")
    return order / (len(scores) - 1)


def weighted_rank_ensemble(score_map, weights):
    ensemble = None
    for name, weight in weights.items():
        ranks = percentile_ranks(score_map[name])
        ensemble = weight * ranks if ensemble is None else ensemble + weight * ranks
    return ensemble


def reference_percentile_scores(reference_scores, scores):
    reference_scores = np.sort(np.asarray(reference_scores, dtype=np.float64))
    scores = np.asarray(scores, dtype=np.float64)
    if len(reference_scores) == 0:
        return np.zeros_like(scores, dtype=np.float64)
    return np.searchsorted(reference_scores, scores, side="right") / len(reference_scores)


def weighted_reference_rank_ensemble(train_score_map, test_score_map, weights):
    train_ensemble = None
    test_ensemble = None
    for name, weight in weights.items():
        train_ranks = reference_percentile_scores(train_score_map[name], train_score_map[name])
        test_ranks = reference_percentile_scores(train_score_map[name], test_score_map[name])
        train_ensemble = weight * train_ranks if train_ensemble is None else train_ensemble + weight * train_ranks
        test_ensemble = weight * test_ranks if test_ensemble is None else test_ensemble + weight * test_ranks
    return train_ensemble, test_ensemble


def fit_adaptive_threshold(train_scores):
    train_scores = np.asarray(train_scores, dtype=np.float64)
    train_scores = train_scores[np.isfinite(train_scores)]
    if len(train_scores) == 0:
        return 0.0
    train_skew = scipy.stats.skew(train_scores)
    if train_skew > 1.2:
        try:
            shape, loc, scale = scipy.stats.lognorm.fit(train_scores, floc=0)
            threshold = scipy.stats.lognorm.ppf(0.98, shape, loc, scale)
        except Exception:
            threshold = np.percentile(train_scores, 98)
    elif train_skew < 0.2:
        mu_fit, std_fit = scipy.stats.norm.fit(train_scores)
        threshold = scipy.stats.norm.ppf(0.98, mu_fit, std_fit)
    else:
        try:
            a_fit, loc_fit, scale_fit = scipy.stats.gamma.fit(train_scores, floc=0)
            threshold = scipy.stats.gamma.ppf(0.98, a_fit, loc_fit, scale_fit)
        except Exception:
            threshold = np.percentile(train_scores, 98)
    if not np.isfinite(threshold):
        threshold = np.percentile(train_scores, 98)
    return float(threshold)


def fit_evt_threshold(train_scores, q=0.02):
    train_scores = np.asarray(train_scores, dtype=np.float64)
    train_scores = train_scores[np.isfinite(train_scores)]
    adaptive_threshold = fit_adaptive_threshold(train_scores)
    if len(train_scores) == 0:
        return adaptive_threshold, "empty_fallback"
    tail_base = np.percentile(train_scores, 90)
    excesses = train_scores[train_scores > tail_base] - tail_base
    if len(excesses) > 10:
        try:
            c_fit, _, scale_fit = scipy.stats.genpareto.fit(excesses, floc=0)
            prob_excess = 1.0 - (q * len(train_scores) / len(excesses))
            if prob_excess < 0:
                prob_excess = 0.98
            elif prob_excess >= 1.0:
                prob_excess = 0.999
            excess_threshold = scipy.stats.genpareto.ppf(prob_excess, c_fit, loc=0, scale=scale_fit)
            threshold = tail_base + excess_threshold
            if np.isfinite(threshold):
                return float(threshold), "evt_gpd"
        except Exception:
            pass
    if np.isfinite(adaptive_threshold):
        return adaptive_threshold, "adaptive_fallback"
    return float(np.percentile(train_scores, 98)), "percentile_fallback"


def evaluate_scores(y_test, scores, train_scores):
    threshold, threshold_method = fit_evt_threshold(train_scores)
    preds = (scores > threshold).astype(int)
    precision, recall, pr_thresholds = precision_recall_curve(y_test, scores)
    f1_scores = [f1_score(y_test, (scores >= t).astype(int), zero_division=0) for t in np.unique(scores)]
    return {
        "auc_roc": roc_auc_score(y_test, scores),
        "auc_pr": auc(recall, precision),
        "f1_evt": f1_score(y_test, preds, zero_division=0),
        "oracle_f1": max(f1_scores) if f1_scores else 0.0,
        "evt_threshold": threshold,
        "threshold_method": threshold_method,
    }


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_dataset(dataset_name, epochs=10):
    X_train, X_test, y_test = load_dataset_data(dataset_name)
    if len(X_train) == 0 or len(X_test) == 0:
        return []
    seq_len = X_train.shape[1]
    X_train = z_normalize(X_train)
    X_test = z_normalize(X_test)
    logger.info("Dataset: %s | Length: %s | Train size: %s", dataset_name, seq_len, len(X_train))

    multi_train, multi_test = train_contrastive_scores(X_train, X_test, mode="multi_aug", epochs=epochs)
    fused_train, fused_test = train_contrastive_scores(X_train, X_test, mode="fused", epochs=epochs)
    hybrid_train, hybrid_test = train_mse_scores(X_train, X_test, epochs=epochs)
    train_score_map = {"multi_aug": multi_train, "fused": fused_train, "hybrid": hybrid_train}
    test_score_map = {"multi_aug": multi_test, "fused": fused_test, "hybrid": hybrid_test}

    rows = []
    for config_name, weights in WEIGHT_CONFIGS.items():
        train_ensemble_scores, test_ensemble_scores = weighted_reference_rank_ensemble(
            train_score_map,
            test_score_map,
            weights,
        )
        metrics = evaluate_scores(y_test, test_ensemble_scores, train_ensemble_scores)
        rows.append(
            {
                "dataset_name": dataset_name,
                "config_name": config_name,
                "sequence_length": seq_len,
                **metrics,
            }
        )
    return rows


def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT name FROM datasets
        WHERE name NOT IN ('CornellWhaleChallenge', 'Wafer_normal_1')
        ORDER BY name
        """
    )
    dataset_names = [row[0] for row in cursor.fetchall()]
    conn.close()

    logger.info("Using acceleration device: %s", device)
    logger.info("Starting rank ensemble calibration on %d datasets.", len(dataset_names))
    detail_rows = []
    for idx, name in enumerate(dataset_names, 1):
        try:
            detail_rows.extend(run_dataset(name, epochs=10))
        except Exception as exc:
            logger.error("Error evaluating dataset %s: %s", name, exc, exc_info=True)
        if idx % 25 == 0 or idx == len(dataset_names):
            write_csv(DETAIL_OUT_PATH, detail_rows)
            logger.info("Progress: [%4d/%4d] rows=%d", idx, len(dataset_names), len(detail_rows))

    write_csv(DETAIL_OUT_PATH, detail_rows)
    summary_rows = []
    for config_name in WEIGHT_CONFIGS:
        subset = [r for r in detail_rows if r["config_name"] == config_name]
        if not subset:
            continue
        summary_rows.append(
            {
                "config_name": config_name,
                "num_datasets": len(subset),
                "auc_roc": np.mean([r["auc_roc"] for r in subset]),
                "auc_pr": np.mean([r["auc_pr"] for r in subset]),
                "f1_evt": np.mean([r["f1_evt"] for r in subset]),
                "oracle_f1": np.mean([r["oracle_f1"] for r in subset]),
            }
        )
    write_csv(SUMMARY_OUT_PATH, summary_rows)
    for row in summary_rows:
        logger.info(
            "%s | AUC %.4f | PR %.4f | F1 %.4f | Oracle %.4f",
            row["config_name"],
            row["auc_roc"],
            row["auc_pr"],
            row["f1_evt"],
            row["oracle_f1"],
        )
    logger.info("Rank ensemble calibration finished.")


if __name__ == "__main__":
    main()
