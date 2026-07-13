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
CSV_OUT_PATH = "/Users/minho/Documents/Dataset/vae_results_advanced_periodicity.csv"

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

def check_dataset_periodicity_advanced(X):
    n_samples, seq_len = X.shape
    if seq_len < 15: # Too short for STFT or meaningful periodicity
        return False
        
    sample_periodic_flags = []
    # Sample up to 50 series
    sample_indices = np.random.choice(n_samples, size=min(n_samples, 50), replace=False)
    
    for idx in sample_indices:
        x = X[idx].copy()
        
        # 1. Apply linear detrending to prevent linear trends from inflating ACF
        try:
            x_detrend = scipy.signal.detrend(x)
        except:
            x_detrend = x
            
        std_val = np.std(x_detrend)
        if std_val == 0:
            sample_periodic_flags.append(False)
            continue
            
        # 2. Check ACF condition (harmonic peaks) on detrended signal
        acf = np.correlate(x_detrend, x_detrend, mode='full')
        acf = acf[acf.size // 2:] / (seq_len * (std_val ** 2))
        
        peaks = []
        for i in range(5, len(acf) - 1):
            if acf[i] > acf[i-1] and acf[i] > acf[i+1]:
                peaks.append(acf[i])
        has_acf_peak = (max(peaks) > 0.45) if peaks else False
        
        # 3. Check Spectral Entropy condition (FFT PSD concentration)
        try:
            fft_vals = np.fft.rfft(x_detrend)
            psd = np.abs(fft_vals) ** 2
            psd_sum = np.sum(psd)
            if psd_sum == 0:
                psd_norm = np.ones_like(psd) / len(psd)
            else:
                psd_norm = psd / psd_sum
                
            # Compute Spectral Entropy
            entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
            max_entropy = np.log2(len(psd_norm))
            norm_entropy = entropy / max_entropy if max_entropy > 0 else 1.0
            
            # Concentrated power spectral density indicates high periodicity
            has_spectral_concentration = norm_entropy < 0.65
        except:
            has_spectral_concentration = False
            
        # OR gate: periodic if either ACF peaks are strong or PSD is highly concentrated
        is_periodic = has_acf_peak or has_spectral_concentration
        sample_periodic_flags.append(is_periodic)
        
    # Dataset is periodic if >= 50% of representative samples pass
    ratio = np.mean(sample_periodic_flags)
    return bool(ratio >= 0.5)

def get_stft_params(seq_len):
    if seq_len >= 128:
        return 64, 16, 64
    elif seq_len >= 64:
        return 32, 8, 32
    else:
        return 16, 4, 16

def compute_stft_magnitude(x, seq_len):
    n_fft, hop_length, win_length = get_stft_params(seq_len)
    window = torch.hann_window(win_length).to(x.device)
    x_2d = x.squeeze(1)
    stft_out = torch.stft(
        x_2d,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode='reflect',
        normalized=False,
        return_complex=True
    )
    magnitude = torch.abs(stft_out)
    return magnitude

def compute_spectral_loss(x, x_recon, seq_len):
    mag_x = compute_stft_magnitude(x, seq_len)
    mag_recon = compute_stft_magnitude(x_recon, seq_len)
    
    fro_x = torch.norm(mag_x, p='fro')
    if fro_x == 0:
        fro_x = 1.0
    loss_linear = torch.norm(mag_x - mag_recon, p='fro') / fro_x
    
    eps = 1e-5
    log_x = torch.log(mag_x + eps)
    log_recon = torch.log(mag_recon + eps)
    loss_log = torch.mean(torch.abs(log_x - log_recon))
    
    return loss_linear + loss_log

def compute_sample_spectral_loss(x, x_recon, seq_len):
    mag_x = compute_stft_magnitude(x, seq_len)
    mag_recon = compute_stft_magnitude(x_recon, seq_len)
    
    diff = mag_x - mag_recon
    norm_diff = torch.norm(diff, p='fro', dim=(1, 2))
    norm_x = torch.norm(mag_x, p='fro', dim=(1, 2))
    norm_x = torch.where(norm_x == 0, torch.ones_like(norm_x), norm_x)
    loss_linear = norm_diff / norm_x
    
    eps = 1e-5
    log_diff = torch.log(mag_x + eps) - torch.log(mag_recon + eps)
    loss_log = torch.abs(log_diff).mean(dim=(1, 2))
    
    return loss_linear + loss_log

def run_evaluation(dataset_name, epochs=10, batch_size=128, beta=0.001):
    try:
        X_train, X_test, y_test = load_dataset_data(dataset_name)
        if len(X_train) == 0 or len(X_test) == 0:
            return None
            
        original_train_size = len(X_train)
        seq_len = X_train.shape[1]
        
        # 1. Apply Advanced Periodicity Classifier (Detrend + Entropy + ACF)
        is_periodic = check_dataset_periodicity_advanced(X_train)
        
        # Set spectral loss weight dynamically based on advanced prediction
        lambda_spec = 0.5 if is_periodic else 0.0
        
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
        model = ConvVAE(latent_dim=128, seq_len=seq_len).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Train
        model.train()
        for epoch in range(epochs):
            for batch in train_loader:
                x_batch = batch[0].to(device)
                optimizer.zero_grad()
                recon, mu, logvar = model(x_batch)
                
                # Time domain loss
                recon_loss = nn.functional.mse_loss(recon, x_batch, reduction='mean')
                
                # Spectral loss (only if lambda_spec > 0)
                if lambda_spec > 0:
                    spec_loss = compute_spectral_loss(x_batch, recon, seq_len)
                else:
                    spec_loss = 0.0
                    
                # KL loss
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                
                # Combined loss
                total_loss = (1.0 - lambda_spec) * recon_loss + lambda_spec * spec_loss + beta * kl_loss
                
                total_loss.backward()
                optimizer.step()
                
        # Evaluate TRAIN (on original training data for calibration)
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
                
                if lambda_spec > 0:
                    spec_errs = compute_sample_spectral_loss(x_batch, recon, seq_len)
                else:
                    spec_errs = torch.zeros_like(recon_errs)
                    
                scores = (1.0 - lambda_spec) * recon_errs + lambda_spec * spec_errs + beta * kl_errs
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
                
                if lambda_spec > 0:
                    spec_errs = compute_sample_spectral_loss(x_batch, recon, seq_len)
                else:
                    spec_errs = torch.zeros_like(recon_errs)
                    
                scores = (1.0 - lambda_spec) * recon_errs + lambda_spec * spec_errs + beta * kl_errs
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
            
        # 3. Extreme Value Theory (EVT) Peaks-Over-Threshold (POT) - GPD fit
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
            "original_train_size": original_train_size,
            "is_periodic": int(is_periodic),
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
    print(f"총 {total_datasets}개의 전체 데이터셋에 대해 '고도화된 주기성 분기 하이브리드 VAE' 평가를 시작합니다.")
    print(f"결과는 {CSV_OUT_PATH} 에 저장됩니다.\n")
    
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
                avg_auc = np.mean([r['auc_roc'] for r in valid_results])
                avg_perc = np.mean([r['f1_percentile'] for r in valid_results])
                avg_adap = np.mean([r['f1_adaptive'] for r in valid_results])
                avg_evt = np.mean([r['f1_evt'] for r in valid_results])
                periodic_ratio = np.mean([r['is_periodic'] for r in valid_results]) * 100
                print(f"[{idx:4d}/{total_datasets:4d}] 완료... (현재 {len(valid_results)}개 - AUC-ROC: {avg_auc:.4f}, EVT F1: {avg_evt:.4f}, 주기성 판정 비율: {periodic_ratio:.1f}%)")
                
    # Final Output
    df = pd.DataFrame([r for r in results if r is not None])
    df.to_csv(CSV_OUT_PATH, index=False)
    
    print("\n" + "="*50)
    print("고도화된 주기성 분기형 하이브리드 VAE 벤치마크 완료!")
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
