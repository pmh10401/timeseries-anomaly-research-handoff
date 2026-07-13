import os
import sqlite3
import pandas as pd
import numpy as np
import scipy.stats
import scipy.signal
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc

# DB Path
DB_PATH = "/Users/minho/Documents/Dataset/univariate_ts.db"
CSV_OUT_PATH = "/Users/minho/Documents/Dataset/vae_results_length_gated_periodicity.csv"

# Check GPU acceleration
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

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

def check_dataset_periodicity_length_gated(X_train):
    """
    Check dataset periodicity using advanced signals check,
    but GATED by sequence length constraint (L >= 128).
    """
    N, L = X_train.shape
    
    # 1. Length-Gate Filter: If sequence is short, bypass and return False immediately
    if L < 128:
        return False
        
    # Calculate ACF on mean train series after linear detrending
    mean_series = X_train.mean(axis=0)
    detrended = scipy.signal.detrend(mean_series)
    
    # Standardize to prevent scale issues in ACF
    std_val = detrended.astype(np.float64).std()
    if std_val == 0:
        std_val = 1.0
    detrended_norm = (detrended - detrended.mean()) / std_val
    
    # ACF
    acf = np.correlate(detrended_norm, detrended_norm, mode='full')
    acf = acf[acf.size // 2:]
    acf = acf / acf[0] # normalize
    
    # Peak detection in ACF (lag >= 5)
    acf_peaks, _ = scipy.signal.find_peaks(acf[5:])
    acf_peaks = acf_peaks + 5
    has_acf_peak = False
    if len(acf_peaks) > 0:
        max_peak_val = acf[acf_peaks].max()
        if max_peak_val > 0.45:
            has_acf_peak = True
            
    # FFT PSD Normalized Spectral Entropy Check
    n_fft = max(256, int(2 ** np.ceil(np.log2(L))))
    freqs, psd = scipy.signal.welch(detrended, nperseg=min(L, 256), nfft=n_fft)
    psd_sum = psd.sum()
    if psd_sum > 0:
        psd_prob = psd / psd_sum
        # normalized spectral entropy
        entropy = -np.sum(psd_prob * np.log2(psd_prob + 1e-12)) / np.log2(len(psd_prob))
    else:
        entropy = 1.0
        
    is_periodic = has_acf_peak or (entropy < 0.65)
    return bool(is_periodic)

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

def get_stft_loss(recon, target):
    # Dynamic n_fft configuration to prevent size mismatches
    seq_len = target.size(2)
    n_fft = 64
    hop_length = 16
    
    if seq_len < 64:
        n_fft = 32
        hop_length = 8
        
    stft_recon = torch.stft(recon.squeeze(1), n_fft=n_fft, hop_length=hop_length, return_complex=True)
    stft_target = torch.stft(target.squeeze(1), n_fft=n_fft, hop_length=hop_length, return_complex=True)
    
    mag_recon = torch.abs(stft_recon)
    mag_target = torch.abs(stft_target)
    
    fro_loss = torch.norm(mag_target - mag_recon, p='fro') / torch.norm(mag_target, p='fro')
    l1_loss = nn.functional.l1_loss(torch.log(mag_target + 1e-5), torch.log(mag_recon + 1e-5))
    return fro_loss + l1_loss

def run_evaluation(dataset_name, epochs=10, batch_size=128, beta=0.001):
    try:
        X_train, X_test, y_test = load_dataset_data(dataset_name)
        if len(X_train) == 0 or len(X_test) == 0:
            return None
            
        original_train_size = len(X_train)
        seq_len = X_train.shape[1]
        
        X_train = z_normalize(X_train)
        X_test = z_normalize(X_test)
        
        # Check Periodicity with Length-Gate Constraint
        is_periodic = check_dataset_periodicity_length_gated(X_train)
        lambda_spec = 0.5 if is_periodic else 0.0
        
        # Apply Data Augmentation
        X_train_aug = augment_time_series(X_train, target_size=500)
        
        # Add channel dimension
        X_train_aug = np.expand_dims(X_train_aug, axis=1)
        X_test = np.expand_dims(X_test, axis=1)
        
        # Loader
        train_tensor = torch.tensor(X_train_aug, dtype=torch.float32)
        train_dataset = TensorDataset(train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Model (Adaptive Conv-VAE)
        model = AdaptiveConvVAE(latent_dim=128, seq_len=seq_len).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Train
        model.train()
        for epoch in range(epochs):
            for batch in train_loader:
                x_batch = batch[0].to(device)
                optimizer.zero_grad()
                recon, mu, logvar = model(x_batch)
                
                # Time-domain reconstruction loss
                recon_loss_time = nn.functional.mse_loss(recon, x_batch, reduction='mean')
                
                # Optional frequency-domain reconstruction loss (if periodic and L >= 128)
                if lambda_spec > 0:
                    recon_loss_freq = get_stft_loss(recon, x_batch)
                    recon_loss = (1 - lambda_spec) * recon_loss_time + lambda_spec * recon_loss_freq
                else:
                    recon_loss = recon_loss_time
                    
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                
                total_loss = recon_loss + beta * kl_loss
                total_loss.backward()
                optimizer.step()
                
        # Evaluate TRAIN
        model.eval()
        train_errors = []
        with torch.no_grad():
            eval_train_tensor = torch.tensor(np.expand_dims(X_train, axis=1), dtype=torch.float32)
            eval_train_dataset = TensorDataset(eval_train_tensor)
            train_loader_eval = DataLoader(eval_train_dataset, batch_size=batch_size, shuffle=False)
            for batch in train_loader_eval:
                x_batch = batch[0].to(device)
                recon, mu, logvar = model(x_batch)
                
                recon_errs_time = ((x_batch - recon) ** 2).mean(dim=(1, 2))
                
                if lambda_spec > 0:
                    recon_errs_freq = torch.zeros(x_batch.size(0), device=device)
                    # compute instance-wise stft loss
                    for idx_item in range(x_batch.size(0)):
                        item_recon = recon[idx_item:idx_item+1]
                        item_target = x_batch[idx_item:idx_item+1]
                        recon_errs_freq[idx_item] = get_stft_loss(item_recon, item_target)
                    recon_errs = (1 - lambda_spec) * recon_errs_time + lambda_spec * recon_errs_freq
                else:
                    recon_errs = recon_errs_time
                    
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
                
                recon_errs_time = ((x_batch - recon) ** 2).mean(dim=(1, 2))
                
                if lambda_spec > 0:
                    recon_errs_freq = torch.zeros(x_batch.size(0), device=device)
                    for idx_item in range(x_batch.size(0)):
                        item_recon = recon[idx_item:idx_item+1]
                        item_target = x_batch[idx_item:idx_item+1]
                        recon_errs_freq[idx_item] = get_stft_loss(item_recon, item_target)
                    recon_errs = (1 - lambda_spec) * recon_errs_time + lambda_spec * recon_errs_freq
                else:
                    recon_errs = recon_errs_time
                    
                kl_errs = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                scores = recon_errs + beta * kl_errs
                test_errors.extend(scores.cpu().numpy())
        test_errors = np.array(test_errors)
        
        anomaly_scores = test_errors
        
        # 1. Baseline Percentile (98%)
        thresh_percentile = np.percentile(train_errors, 98)
        
        # 2. Skewness-based Adaptive Thresholding
        train_skew = scipy.stats.skew(train_errors)
        if train_skew > 1.2:
            try:
                shape, loc, scale = scipy.stats.lognorm.fit(train_errors, floc=0)
                thresh_adaptive = scipy.stats.lognorm.ppf(0.98, shape, loc, scale)
            except:
                thresh_adaptive = np.percentile(train_errors, 98)
        elif train_skew < 0.2:
            mu_fit, std_fit = scipy.stats.norm.fit(train_errors)
            thresh_adaptive = scipy.stats.norm.ppf(0.98, mu_fit, std_fit)
        else:
            try:
                a_fit, loc_fit, scale_fit = scipy.stats.gamma.fit(train_errors, floc=0)
                thresh_adaptive = scipy.stats.gamma.ppf(0.98, a_fit, loc_fit, scale_fit)
            except:
                thresh_adaptive = np.percentile(train_errors, 98)
                
        if not np.isfinite(thresh_adaptive) or thresh_adaptive <= 0:
            thresh_adaptive = np.percentile(train_errors, 98)
            
        # 3. Extreme Value Theory (EVT) POT GPD fit
        t = np.percentile(train_errors, 90)
        excesses = train_errors[train_errors > t] - t
        n = len(train_errors)
        Nt = len(excesses)
        q = 0.02
        
        if Nt > 10:
            try:
                c_fit, loc_fit, scale_fit = scipy.stats.genpareto.fit(excesses, floc=0)
                prob_excess = 1.0 - (q * n / Nt)
                if prob_excess < 0:
                    prob_excess = 0.98
                elif prob_excess >= 1.0:
                    prob_excess = 0.999
                excess_thresh = scipy.stats.genpareto.ppf(prob_excess, c_fit, loc=0, scale=scale_fit)
                thresh_evt = t + excess_thresh
            except Exception as e:
                thresh_evt = thresh_adaptive
        else:
            thresh_evt = thresh_adaptive
            
        if not np.isfinite(thresh_evt) or thresh_evt <= 0:
            thresh_evt = thresh_adaptive
            
        # Metrics Calculation
        auc_roc = roc_auc_score(y_test, anomaly_scores)
        precision, recall, thresholds = precision_recall_curve(y_test, anomaly_scores)
        auc_pr = auc(recall, precision)
        
        y_pred_percentile = (anomaly_scores > thresh_percentile).astype(int)
        f1_percentile = f1_score(y_test, y_pred_percentile, zero_division=0)
        
        y_pred_adaptive = (anomaly_scores > thresh_adaptive).astype(int)
        f1_adaptive = f1_score(y_test, y_pred_adaptive, zero_division=0)
        
        y_pred_evt = (anomaly_scores > thresh_evt).astype(int)
        f1_evt = f1_score(y_test, y_pred_evt, zero_division=0)
        
        # Oracle F1
        f1_scores = []
        for thresh in thresholds:
            preds = (anomaly_scores > thresh).astype(int)
            f1_scores.append(f1_score(y_test, preds, zero_division=0))
        best_f1_idx = np.argmax(f1_scores) if f1_scores else 0
        best_f1 = f1_scores[best_f1_idx] if f1_scores else 0
        
        return {
            "dataset_name": dataset_name,
            "sequence_length": seq_len,
            "is_periodic": is_periodic,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "f1_percentile": f1_percentile,
            "f1_adaptive": f1_adaptive,
            "f1_evt": f1_evt,
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
    print(f"총 {total_datasets}개의 전체 데이터셋에 대해 '길이 게이트형 주기 VAE' 평가를 시작합니다.")
    print(f"결과는 {CSV_OUT_PATH} 에 저장됩니다.\n")
    
    results = []
    
    for idx, name in enumerate(dataset_names, 1):
        res = run_evaluation(name, epochs=10)
        if res:
            results.append(res)
            
        # Save periodically
        if idx % 50 == 0 or idx == total_datasets:
            df = pd.DataFrame([r for r in results if r is not None])
            df.to_csv(CSV_OUT_PATH, index=False)
            
            valid_results = [r for r in results if r is not None]
            if valid_results:
                avg_auc = np.mean([r['auc_roc'] for r in valid_results])
                avg_evt = np.mean([r['f1_evt'] for r in valid_results])
                periodic_ratio = np.mean([1 if r['is_periodic'] else 0 for r in valid_results]) * 100
                print(f"[{idx:4d}/{total_datasets:4d}] 완료... (현재 {len(valid_results)}개 - AUC-ROC: {avg_auc:.4f}, EVT F1: {avg_evt:.4f}, 주기성 판정 비율: {periodic_ratio:.1f}%)")
                
    # Final Output
    df = pd.DataFrame([r for r in results if r is not None])
    df.to_csv(CSV_OUT_PATH, index=False)
    
    print("\n" + "="*50)
    print("길이 게이트형 주기 VAE 벤치마크 완료!")
    print(f"- 결과 저장 경로: {CSV_OUT_PATH}")
    print(f"- 성공적으로 평가된 데이터셋 수: {len(df)}")
    print(f"- 주기성 판정 데이터셋 비율 : {df['is_periodic'].mean() * 100:.2f}%")
    print(f"- 전체 평균 AUC-ROC     : {df['auc_roc'].mean():.4f}")
    print(f"- 전체 평균 AUC-PR      : {df['auc_pr'].mean():.4f}")
    print(f"- 98% Baseline 평균 F1  : {df['f1_percentile'].mean():.4f}")
    print(f"- 왜도 기반 적응형 평균 F1: {df['f1_adaptive'].mean():.4f}")
    print(f"- 극값 이론 (EVT) 평균 F1 : {df['f1_evt'].mean():.4f}")
    print(f"- 오라클 상한선(Oracle) F1 : {df['oracle_f1'].mean():.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
