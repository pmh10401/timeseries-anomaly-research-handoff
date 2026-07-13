import csv
import logging
import sqlite3

import numpy as np
import scipy.stats
import torch
import torch.nn as nn
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from run_rank_ensemble_calibration import (
    DB_PATH,
    WEIGHT_CONFIGS,
    device,
    evaluate_scores,
    load_dataset_data,
    train_contrastive_scores,
    train_mse_scores,
    weighted_reference_rank_ensemble,
    write_csv,
    z_normalize,
)


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logger = logging.getLogger("RankEnsembleCalibrationV2")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.propagate = False
logger.addHandler(logging.FileHandler("/Users/minho/Documents/Dataset/rank_ensemble_calibration_v2_run.log"))
logger.addHandler(logging.StreamHandler())
for handler in logger.handlers:
    handler.setFormatter(logging.Formatter(LOG_FORMAT))

DETAIL_OUT_PATH = "/Users/minho/Documents/Dataset/vae_results_rank_ensemble_calibration_v2_train_evt.csv"
SUMMARY_OUT_PATH = "/Users/minho/Documents/Dataset/vae_results_rank_ensemble_calibration_v2_train_evt_summary.csv"
HYBRID_ROUTE_LENGTH = 150


class AdaptiveReconProbVAE(nn.Module):
    def __init__(self, latent_dim=128, seq_len=152):
        super().__init__()
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
        for i in range(self.num_layers - 1):
            dec_layers.append(
                nn.ConvTranspose1d(
                    self.trans_channels[i],
                    self.trans_channels[i + 1],
                    kernel_size=trans_kernel,
                    stride=2,
                    padding=self.padding,
                )
            )
            dec_layers.append(nn.ReLU(True))
        self.dec_shared = nn.Sequential(*dec_layers) if dec_layers else nn.Identity()

        last_in_channel = self.trans_channels[-2]
        self.dec_head_mu = nn.ConvTranspose1d(last_in_channel, 1, kernel_size=trans_kernel, stride=2, padding=self.padding)
        self.dec_head_logvar = nn.ConvTranspose1d(
            last_in_channel,
            1,
            kernel_size=trans_kernel,
            stride=2,
            padding=self.padding,
        )

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
        h = self.dec_shared(h)
        recon_mu = self.dec_head_mu(h)
        recon_logvar = self.dec_head_logvar(h)
        if recon_mu.size(2) != target_len:
            recon_mu = nn.functional.interpolate(recon_mu, size=target_len, mode="linear", align_corners=False)
            recon_logvar = nn.functional.interpolate(recon_logvar, size=target_len, mode="linear", align_corners=False)
        return recon_mu, torch.clamp(recon_logvar, min=-9.21, max=5.0)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_mu, recon_logvar = self.decode(z, x.size(2))
        return recon_mu, recon_logvar, mu, logvar


def augment_time_series(X, target_size=500):
    if len(X) >= target_size:
        return X[:target_size]
    augmented = []
    for _ in range(target_size - len(X)):
        x = X[np.random.randint(0, len(X))].copy()
        aug_type = np.random.choice(["jitter", "scale", "both"])
        if aug_type in ["jitter", "both"]:
            x += np.random.normal(0, 0.03, size=len(x))
        if aug_type in ["scale", "both"]:
            x *= np.random.uniform(0.9, 1.1)
        augmented.append(x)
    return np.vstack([X, np.array(augmented)])


def gaussian_nll(x, recon_mu, recon_logvar):
    return 0.5 * recon_logvar + 0.5 * np.log(2 * np.pi) + ((x - recon_mu) ** 2) / (2 * torch.exp(recon_logvar))


def train_recon_prob_scores(X_train, X_test, epochs=10, batch_size=128, beta=0.001, mc_samples=30):
    seq_len = X_train.shape[1]
    X_train_aug = augment_time_series(X_train, target_size=500)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(np.expand_dims(X_train_aug, axis=1), dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
    )
    model = AdaptiveReconProbVAE(latent_dim=128, seq_len=seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        for (x_batch,) in train_loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()
            recon_mu, recon_logvar, mu, logvar = model(x_batch)
            recon_loss = gaussian_nll(x_batch, recon_mu, recon_logvar).mean()
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kl_loss
            loss.backward()
            optimizer.step()

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
                mu, logvar = model.encode(x_batch)
                nll_sum = torch.zeros(x_batch.size(0), device=device)
                for _ in range(mc_samples):
                    z = model.reparameterize(mu, logvar)
                    recon_mu, recon_logvar = model.decode(z, x_batch.size(2))
                    nll_sum += gaussian_nll(x_batch, recon_mu, recon_logvar).mean(dim=(1, 2))
                nll_avg = nll_sum / mc_samples
                kl_errs = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                out.extend((nll_avg + beta * kl_errs).cpu().numpy())
        return np.array(out)

    return score_array(X_train), score_array(X_test)


def select_hybrid_scores_by_length(seq_len, mse_scores, recon_prob_scores):
    if seq_len < HYBRID_ROUTE_LENGTH:
        return recon_prob_scores[0], recon_prob_scores[1], "Recon_Probability_NLL"
    return mse_scores[0], mse_scores[1], "MSE_Reconstruction"


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
    mse_scores = train_mse_scores(X_train, X_test, epochs=epochs)
    if seq_len < HYBRID_ROUTE_LENGTH:
        recon_prob_scores = train_recon_prob_scores(X_train, X_test, epochs=epochs)
    else:
        recon_prob_scores = mse_scores
    hybrid_train, hybrid_test, route = select_hybrid_scores_by_length(seq_len, mse_scores, recon_prob_scores)
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
                "hybrid_route": route,
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
    logger.info("Starting rank ensemble calibration v2 on %d datasets.", len(dataset_names))
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
    logger.info("Rank ensemble calibration v2 finished.")


if __name__ == "__main__":
    main()
