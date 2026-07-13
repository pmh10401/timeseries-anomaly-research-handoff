import os
import sqlite3
import pandas as pd
import numpy as np
import scipy.stats
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc

# Configure logging
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler("/Users/minho/Documents/Dataset/density_weighted_run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DensityWeightedVAE")

# DB Path
DB_PATH = "/Users/minho/Documents/Dataset/univariate_ts.db"
CSV_OUT_PATH = "/Users/minho/Documents/Dataset/vae_results_adaptive_cnn_density_weighted.csv"

# Check GPU acceleration
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

logger.info(f"Using acceleration device: {device}")

class AdaptiveConvVAE(nn.Module):
    def __init__(self, latent_dim=128, seq_len=152):
        super(AdaptiveConvVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Determine dynamic network topology based on sequence length
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
        
        # Build Encoder Conv layers dynamically
        enc_layers = []
        for i in range(self.num_layers):
            enc_layers.append(
                nn.Conv1d(
                    in_channels=self.channels[i],
                    out_channels=self.channels[i+1],
                    kernel_size=self.kernel_size,
                    stride=2,
                    padding=self.padding
                )
            )
            enc_layers.append(nn.ReLU(True))
        self.enc_conv = nn.Sequential(*enc_layers)
        
        # Calculate flat dimension dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, seq_len)
            dummy_output = self.enc_conv(dummy_input)
            self.conv_out_len = dummy_output.size(2)
            self.flat_dim = self.last_channel * self.conv_out_len
            
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        
        # Decoder FC
        self.dec_fc = nn.Linear(latent_dim, self.flat_dim)
        
        # Build Decoder ConvTranspose layers dynamically
        dec_layers = []
        trans_kernel = self.kernel_size + 1 # Matching size expansion nicely with stride 2
        for i in range(self.num_layers):
            dec_layers.append(
                nn.ConvTranspose1d(
                    in_channels=self.trans_channels[i],
                    out_channels=self.trans_channels[i+1],
                    kernel_size=trans_kernel,
                    stride=2,
                    padding=self.padding
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
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, target_len):
        h = self.dec_fc(z)
        h = h.view(h.size(0), self.last_channel, self.conv_out_len)
        x_recon = self.dec_conv(h)
        if x_recon.size(2) != target_len:
            x_recon = nn.functional.interpolate(x_recon, size=target_len, mode='linear', align_corners=False)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, x.size(2))
        return x_recon, mu, logvar

def load_dataset_data(dataset_name):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # TRAIN
    cursor.execute("""
        SELECT i.values_blob
        FROM instances i
        JOIN datasets d ON i.dataset_id = d.id
        WHERE d.name = ? AND i.split = 'TRAIN'
        ORDER BY i.instance_index
    """, (dataset_name,))
    train_rows = cursor.fetchall()
    X_train = [np.frombuffer(row[0], dtype=np.float32) for row in train_rows]
    
    # TEST
    cursor.execute("""
        SELECT i.values_blob, i.label
        FROM instances i
        JOIN datasets d ON i.dataset_id = d.id
        WHERE d.name = ? AND i.split = 'TEST'
        ORDER BY i.instance_index
    """, (dataset_name,))
    test_rows = cursor.fetchall()
    X_test = [np.frombuffer(row[0], dtype=np.float32) for row in test_rows]
    y_test = [int(row[1]) for row in test_rows]
    
    conn.close()
    return np.array(X_train), np.array(X_test), np.array(y_test)

def z_normalize(X):
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std

def augment_single_series(x):
    L = len(x)
    aug_x = x.copy()
    aug_type = np.random.choice(['jitter', 'scale', 'both'])
    if aug_type in ['jitter', 'both']:
        noise = np.random.normal(0, 0.03, size=L)
        aug_x += noise
    if aug_type in ['scale', 'both']:
        scale_factor = np.random.uniform(0.9, 1.1)
        aug_x *= scale_factor
    return aug_x

def prepare_contrastive_dataset(X, target_size=500):
    N, L = X.shape
    if N < target_size:
        indices = np.random.choice(N, target_size, replace=True)
        X_orig = X[indices]
    else:
        X_orig = X[:target_size]
    X_aug = np.array([augment_single_series(x) for x in X_orig])
    return X_orig, X_aug

# 🧪 [실험 22] KNN 거리 기반 잠재 공간 밀도 가중치 산출 함수
def calculate_latent_density_weights(z, k_neighbors=5):
    # Pairwise L2 distances: shape B x B
    dist_matrix = torch.cdist(z, z, p=2)
    B = z.size(0)
    K = min(k_neighbors, B - 1)
    
    if K > 0:
        # Sort distances ascendingly: the first column is 0 (distance to self)
        topk_dists, _ = torch.topk(dist_matrix, k=K+1, largest=False, dim=1)
        knn_dists = topk_dists[:, 1:] # Drop self-distance
        avg_knn_dist = knn_dists.mean(dim=1)
        
        # Tau scaling factor (median distance)
        tau = avg_knn_dist.median()
        if tau == 0:
            tau = 1e-5
            
        # Outer boundary samples get exponentially larger weights
        weights = torch.exp(avg_knn_dist / tau)
        # Normalize weights so that their batch mean is 1.0
        weights = weights / weights.mean()
    else:
        weights = torch.ones(B, device=z.device)
        
    return weights

def run_evaluation(dataset_name, epochs=10, batch_size=128):
    try:
        X_train, X_test, y_test = load_dataset_data(dataset_name)
        if len(X_train) == 0 or len(X_test) == 0:
            return None
            
        original_train_size = len(X_train)
        seq_len = X_train.shape[1]
        
        # Scheduling
        max_beta = 0.15 / seq_len
        gamma = 0.8 / np.log(original_train_size + 2)
        
        logger.info(f"Dataset: {dataset_name} | Length: {seq_len} | Train size: {original_train_size}")
        
        X_train = z_normalize(X_train)
        X_test = z_normalize(X_test)
        
        X_train_orig, X_train_aug = prepare_contrastive_dataset(X_train, target_size=500)
        
        # Channel dimension
        X_train_orig = np.expand_dims(X_train_orig, axis=1)
        X_train_aug = np.expand_dims(X_train_aug, axis=1)
        X_test = np.expand_dims(X_test, axis=1)
        
        # Loader
        train_dataset = TensorDataset(
            torch.tensor(X_train_orig, dtype=torch.float32),
            torch.tensor(X_train_aug, dtype=torch.float32)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Model
        model = AdaptiveConvVAE(latent_dim=128, seq_len=seq_len).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Train
        model.train()
        for epoch in range(epochs):
            current_beta = min(max_beta, (epoch / max(1, epochs - 3)) * max_beta)
            for batch in train_loader:
                x_orig = batch[0].to(device)
                x_aug = batch[1].to(device)
                
                optimizer.zero_grad()
                
                recon_orig, mu_orig, logvar_orig = model(x_orig)
                z_orig = model.reparameterize(mu_orig, logvar_orig)
                
                recon_aug, mu_aug, logvar_aug = model(x_aug)
                z_aug = model.reparameterize(mu_aug, logvar_aug)
                
                # 🧪 [실험 22] KNN Density Weighting
                weights_orig = calculate_latent_density_weights(z_orig, k_neighbors=5)
                
                # Apply sample-wise weights to the Reconstruction MSE Loss
                recon_err_samples = nn.functional.mse_loss(recon_orig, x_orig, reduction='none').mean(dim=(1, 2))
                recon_loss = (recon_err_samples * weights_orig).mean()
                
                kl_loss = -0.5 * torch.mean(1 + logvar_orig - mu_orig.pow(2) - logvar_orig.exp())
                
                # Contrastive Latent Loss
                cos_sim = nn.functional.cosine_similarity(z_orig, z_aug, dim=1)
                contrastive_loss = torch.mean(1.0 - cos_sim)
                
                total_loss = recon_loss + current_beta * kl_loss + gamma * contrastive_loss
                total_loss.backward()
                optimizer.step()
                
        # Evaluate TRAIN
        model.eval()
        train_errors = []
        with torch.no_grad():
            eval_train_tensor = torch.tensor(np.expand_dims(X_train, axis=1), dtype=torch.float32)
            train_loader_eval = DataLoader(TensorDataset(eval_train_tensor), batch_size=batch_size, shuffle=False)
            for batch in train_loader_eval:
                x_batch = batch[0].to(device)
                recon, mu, logvar = model(x_batch)
                recon_errs = ((x_batch - recon) ** 2).mean(dim=(1, 2))
                kl_errs = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                scores = recon_errs + max_beta * kl_errs
                train_errors.extend(scores.cpu().numpy())
        train_errors = np.array(train_errors)
        
        # Evaluate TEST
        test_errors = []
        with torch.no_grad():
            test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
            for batch in test_loader:
                x_batch = batch[0].to(device)
                recon, mu, logvar = model(x_batch)
                recon_errs = ((x_batch - recon) ** 2).mean(dim=(1, 2))
                kl_errs = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                scores = recon_errs + max_beta * kl_errs
                test_errors.extend(scores.cpu().numpy())
        test_errors = np.array(test_errors)
        
        anomaly_scores = test_errors
        
        # Dynamic POT
        n_test_samples = len(X_test)
        q_target = max(0.001, 1.0 / n_test_samples)
        
        # 1. Percentile
        thresh_percentile = np.percentile(train_errors, 100 * (1.0 - q_target))
        
        # 2. Skewness
        train_skew = scipy.stats.skew(train_errors)
        if train_skew > 1.2:
            try:
                shape, loc, scale = scipy.stats.lognorm.fit(train_errors, floc=0)
                thresh_adaptive = scipy.stats.lognorm.ppf(1.0 - q_target, shape, loc, scale)
            except:
                thresh_adaptive = np.percentile(train_errors, 100 * (1.0 - q_target))
        elif train_skew < 0.2:
            mu_fit, std_fit = scipy.stats.norm.fit(train_errors)
            thresh_adaptive = scipy.stats.norm.ppf(1.0 - q_target, mu_fit, std_fit)
        else:
            try:
                a_fit, loc_fit, scale_fit = scipy.stats.gamma.fit(train_errors, floc=0)
                thresh_adaptive = scipy.stats.gamma.ppf(1.0 - q_target, a_fit, loc_fit, scale_fit)
            except:
                thresh_adaptive = np.percentile(train_errors, 100 * (1.0 - q_target))
        if not np.isfinite(thresh_adaptive) or thresh_adaptive <= 0:
            thresh_adaptive = np.percentile(train_errors, 100 * (1.0 - q_target))
            
        # 3. EVT-POT
        t = np.percentile(train_errors, 90)
        excesses = train_errors[train_errors > t] - t
        n = len(train_errors)
        Nt = len(excesses)
        
        if Nt > 10:
            try:
                c_fit, loc_fit, scale_fit = scipy.stats.genpareto.fit(excesses, floc=0)
                prob_excess = 1.0 - (q_target * n / Nt)
                prob_excess = np.clip(prob_excess, 0.90, 0.9999)
                excess_thresh = scipy.stats.genpareto.ppf(prob_excess, c_fit, loc=0, scale=scale_fit)
                thresh_evt = t + excess_thresh
            except:
                thresh_evt = thresh_adaptive
        else:
            thresh_evt = thresh_adaptive
        if not np.isfinite(thresh_evt) or thresh_evt <= 0:
            thresh_evt = thresh_adaptive
            
        # Metrics
        auc_roc = roc_auc_score(y_test, anomaly_scores)
        precision, recall, thresholds = precision_recall_curve(y_test, anomaly_scores)
        auc_pr = auc(recall, precision)
        
        y_pred_evt = (anomaly_scores > thresh_evt).astype(int)
        f1_evt = f1_score(y_test, y_pred_evt, zero_division=0)
        
        y_pred_percentile = (anomaly_scores > thresh_percentile).astype(int)
        f1_percentile = f1_score(y_test, y_pred_percentile, zero_division=0)
        
        y_pred_adaptive = (anomaly_scores > thresh_adaptive).astype(int)
        f1_adaptive = f1_score(y_test, y_pred_adaptive, zero_division=0)
        
        f1_scores = []
        for thresh in thresholds:
            preds = (anomaly_scores > thresh).astype(int)
            f1_scores.append(f1_score(y_test, preds, zero_division=0))
        best_f1 = max(f1_scores) if f1_scores else 0
        
        logger.info(f"[{dataset_name}] Done | AUC-ROC: {auc_roc:.4f} | EVT F1: {f1_evt:.4f} | Oracle F1: {best_f1:.4f}")
        return {
            "dataset_name": dataset_name,
            "sequence_length": seq_len,
            "target_q": q_target,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "f1_percentile": f1_percentile,
            "f1_adaptive": f1_adaptive,
            "f1_evt": f1_evt,
            "oracle_f1": best_f1
        }
    except Exception as e:
        logger.error(f"Error evaluating dataset {dataset_name}: {e}", exc_info=True)
        return None

def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name FROM datasets 
        WHERE name NOT IN ('CornellWhaleChallenge', 'Wafer_normal_1')
        ORDER BY name
    """)
    dataset_names = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    total_datasets = len(dataset_names)
    logger.info(f"Starting Latent Density-Weighted VAE on {total_datasets} datasets.")
    
    results = []
    for idx, name in enumerate(dataset_names, 1):
        res = run_evaluation(name, epochs=10)
        if res:
            results.append(res)
        if idx % 50 == 0 or idx == total_datasets:
            df = pd.DataFrame([r for r in results if r is not None])
            df.to_csv(CSV_OUT_PATH, index=False)
            valid_results = [r for r in results if r is not None]
            if valid_results:
                avg_auc = np.mean([r['auc_roc'] for r in valid_results])
                avg_evt = np.mean([r['f1_evt'] for r in valid_results])
                logger.info(f"Progress: [{idx:4d}/{total_datasets:4d}] | CumAvg AUC: {avg_auc:.4f} | CumAvg EVT F1: {avg_evt:.4f}")
                
    df = pd.DataFrame([r for r in results if r is not None])
    df.to_csv(CSV_OUT_PATH, index=False)
    logger.info("Latent Density-Weighted VAE Finished!")
    logger.info(f"- Mean F1 (EVT-POT): {df['f1_evt'].mean():.4f}")

if __name__ == "__main__":
    main()
