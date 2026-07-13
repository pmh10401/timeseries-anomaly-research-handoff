import os
import sqlite3
import pandas as pd
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc, accuracy_score

# DB Path
DB_PATH = "/Users/minho/Documents/Dataset/univariate_ts.db"
CSV_OUT_PATH = "/Users/minho/Documents/Dataset/vae_results_skewness_adaptive.csv"

# Check GPU acceleration
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128, seq_len=152):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder Conv
        self.enc_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
            nn.ReLU(True),
            nn.Conv1d(16, 32, kernel_size=15, stride=2, padding=7),
            nn.ReLU(True),
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
            nn.ReLU(True),
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
            nn.ReLU(True)
        )
        
        # Calculate flat dimension dynamically using dummy tensor (preserves temporal size)
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, seq_len)
            dummy_output = self.enc_conv(dummy_input)
            self.conv_out_len = dummy_output.size(2)
            self.flat_dim = 128 * self.conv_out_len
            
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        
        # Decoder FC
        self.dec_fc = nn.Linear(latent_dim, self.flat_dim)
        
        # Decoder Conv
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=16, stride=2, padding=7),
            nn.ReLU(True),
            nn.ConvTranspose1d(64, 32, kernel_size=16, stride=2, padding=7),
            nn.ReLU(True),
            nn.ConvTranspose1d(32, 16, kernel_size=16, stride=2, padding=7),
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 1, kernel_size=16, stride=2, padding=7),
        )

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
        h = h.view(h.size(0), 128, self.conv_out_len)
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

def augment_time_series(X, target_size=500):
    N, L = X.shape
    if N >= target_size:
        return X[:target_size]
    num_augmented = target_size - N
    augmented_samples = []
    for _ in range(num_augmented):
        idx = np.random.randint(0, N)
        x = X[idx].copy()
        aug_type = np.random.choice(['jitter', 'scale', 'both'])
        if aug_type in ['jitter', 'both']:
            noise = np.random.normal(0, 0.03, size=L)
            x += noise
        if aug_type in ['scale', 'both']:
            scale_factor = np.random.uniform(0.9, 1.1)
            x *= scale_factor
        augmented_samples.append(x)
    return np.vstack([X, np.array(augmented_samples)])

def run_evaluation(dataset_name, epochs=10, batch_size=128, beta=0.001):
    try:
        X_train, X_test, y_test = load_dataset_data(dataset_name)
        if len(X_train) == 0 or len(X_test) == 0:
            return None
            
        original_train_size = len(X_train)
        
        X_train = z_normalize(X_train)
        X_test = z_normalize(X_test)
        
        # Apply Data Augmentation
        X_train_aug = augment_time_series(X_train, target_size=500)
        
        # Add channel dimension
        X_train_aug = np.expand_dims(X_train_aug, axis=1)
        X_test = np.expand_dims(X_test, axis=1)
        
        # Loader
        train_tensor = torch.tensor(X_train_aug, dtype=torch.float32)
        train_dataset = TensorDataset(train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Model
        model = ConvVAE(latent_dim=128, seq_len=X_train.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Train
        model.train()
        for epoch in range(epochs):
            for batch in train_loader:
                x_batch = batch[0].to(device)
                optimizer.zero_grad()
                recon, mu, logvar = model(x_batch)
                
                recon_loss = nn.functional.mse_loss(recon, x_batch, reduction='mean')
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                total_loss = recon_loss + beta * kl_loss
                
                total_loss.backward()
                optimizer.step()
                
        # Evaluate TRAIN (on original z-normalized training data for calibration)
        model.eval()
        train_errors = []
        with torch.no_grad():
            eval_train_tensor = torch.tensor(np.expand_dims(X_train, axis=1), dtype=torch.float32)
            eval_train_dataset = TensorDataset(eval_train_tensor)
            train_loader_eval = DataLoader(eval_train_dataset, batch_size=batch_size, shuffle=False)
            for batch in train_loader_eval:
                x_batch = batch[0].to(device)
                recon, mu, logvar = model(x_batch)
                recon_errs = ((x_batch - recon) ** 2).mean(dim=(1, 2))
                kl_errs = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                scores = recon_errs + beta * kl_errs
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
                scores = recon_errs + beta * kl_errs
                test_errors.extend(scores.cpu().numpy())
        test_errors = np.array(test_errors)
        
        anomaly_scores = test_errors
        
        # Calculate skewness of train errors
        train_skew = scipy.stats.skew(train_errors)
        
        # Define Thresholding Strategies
        # 1. Baseline Percentile (98%)
        thresh_percentile = np.percentile(train_errors, 98)
        
        # 2. Skewness-based Adaptive Thresholding
        if train_skew > 1.2:
            # High positive skew -> fit Log-Normal
            try:
                shape, loc, scale = scipy.stats.lognorm.fit(train_errors, floc=0)
                thresh_adaptive = scipy.stats.lognorm.ppf(0.98, shape, loc, scale)
            except:
                thresh_adaptive = np.percentile(train_errors, 98)
        elif train_skew < 0.2:
            # Low skew / symmetric -> fit Gaussian Normal
            mu_fit, std_fit = scipy.stats.norm.fit(train_errors)
            thresh_adaptive = scipy.stats.norm.ppf(0.98, mu_fit, std_fit)
        else:
            # Moderately skewed -> fit Gamma
            try:
                a_fit, loc_fit, scale_fit = scipy.stats.gamma.fit(train_errors, floc=0)
                thresh_adaptive = scipy.stats.gamma.ppf(0.98, a_fit, loc_fit, scale_fit)
            except:
                thresh_adaptive = np.percentile(train_errors, 98)
                
        # Handle cases where statistics could fail or produce NaNs/Infs
        if not np.isfinite(thresh_adaptive) or thresh_adaptive <= 0:
            thresh_adaptive = np.percentile(train_errors, 98)
            
        # Metrics Calculation
        auc_roc = roc_auc_score(y_test, anomaly_scores)
        precision, recall, thresholds = precision_recall_curve(y_test, anomaly_scores)
        auc_pr = auc(recall, precision)
        
        # F1 Score comparison
        # A. Baseline Percentile
        y_pred_percentile = (anomaly_scores > thresh_percentile).astype(int)
        f1_percentile = f1_score(y_test, y_pred_percentile, zero_division=0)
        
        # B. Adaptive Threshold
        y_pred_adaptive = (anomaly_scores > thresh_adaptive).astype(int)
        f1_adaptive = f1_score(y_test, y_pred_adaptive, zero_division=0)
        
        # C. Oracle F1
        f1_scores = []
        for t in thresholds:
            preds = (anomaly_scores > t).astype(int)
            f1_scores.append(f1_score(y_test, preds, zero_division=0))
        best_f1_idx = np.argmax(f1_scores) if f1_scores else 0
        best_f1 = f1_scores[best_f1_idx] if f1_scores else 0
        
        return {
            "dataset_name": dataset_name,
            "original_train_size": original_train_size,
            "train_skewness": train_skew,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "f1_percentile": f1_percentile,
            "f1_adaptive": f1_adaptive,
            "oracle_f1": best_f1
        }
    except Exception as e:
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
    print(f"가속 디바이스: {device}")
    print(f"총 {total_datasets}개의 전체 데이터셋에 대해 '왜도 기반 적응형 임계치 VAE' 평가를 시작합니다.")
    
    results = []
    
    for idx, name in enumerate(dataset_names, 1):
        res = run_evaluation(name, epochs=10)
        if res:
            results.append(res)
            
        # Periodic output & save
        if idx % 50 == 0 or idx == total_datasets:
            df = pd.DataFrame([r for r in results if r is not None])
            df.to_csv(CSV_OUT_PATH, index=False)
            
            valid_results = [r for r in results if r is not None]
            if valid_results:
                avg_perc = np.mean([r['f1_percentile'] for r in valid_results])
                avg_adap = np.mean([r['f1_adaptive'] for r in valid_results])
                print(f"[{idx:4d}/{total_datasets:4d}] 진행 완료... (현재 완료된 {len(valid_results)}개 - Baseline F1: {avg_perc:.4f}, Adaptive F1: {avg_adap:.4f})")
                
    # Final Output
    df = pd.DataFrame([r for r in results if r is not None])
    df.to_csv(CSV_OUT_PATH, index=False)
    
    print("\n" + "="*50)
    print("왜도 기반 적응형 임계치 벤치마크 완료!")
    print(f"- 결과 저장 경로: {CSV_OUT_PATH}")
    print(f"- 성공적으로 평가된 데이터셋 수: {len(df)}")
    print(f"- 전체 평균 AUC-ROC : {df['auc_roc'].mean():.4f}")
    print(f"- 백분위 98% Baseline 평균 F1 : {df['f1_percentile'].mean():.4f}")
    print(f"- 왜도 기반 적응형 임계값 평균 F1: {df['f1_adaptive'].mean():.4f}")
    print(f"- 오라클 상한선(Oracle) 평균 F1: {df['oracle_f1'].mean():.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
