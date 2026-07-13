import os
import sqlite3
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc, accuracy_score
import scipy.stats as stats
import pandas as pd

# DB Path
DB_PATH = "/Users/minho/Documents/Dataset/univariate_ts.db"

# Check GPU acceleration
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
            nn.ReLU(True),
            nn.Conv1d(16, 32, kernel_size=15, stride=2, padding=7),
            nn.ReLU(True),
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
            nn.ReLU(True),
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=16, stride=2, padding=7),
            nn.ReLU(True),
            nn.ConvTranspose1d(64, 32, kernel_size=16, stride=2, padding=7),
            nn.ReLU(True),
            nn.ConvTranspose1d(32, 16, kernel_size=16, stride=2, padding=7),
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 1, kernel_size=16, stride=2, padding=7),
        )

    def forward(self, x):
        target_len = x.size(2)
        x = self.encoder(x)
        x = self.decoder(x)
        if x.size(2) != target_len:
            x = nn.functional.interpolate(x, size=target_len, mode='linear', align_corners=False)
        return x

def load_dataset_data(dataset_name):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT i.values_blob
        FROM instances i
        JOIN datasets d ON i.dataset_id = d.id
        WHERE d.name = ? AND i.split = 'TRAIN'
        ORDER BY i.instance_index
    """, (dataset_name,))
    X_train = [np.frombuffer(row[0], dtype=np.float32) for row in cursor.fetchall()]
    
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

def get_kde_threshold(train_errors, alpha=0.02):
    """
    Fits Gaussian KDE on train errors and finds the threshold T where CDF = 1 - alpha.
    """
    try:
        kde = stats.gaussian_kde(train_errors)
        # Search grid
        min_err = np.min(train_errors)
        max_err = np.max(train_errors)
        grid = np.linspace(min_err, max_err * 2, 1000)
        
        # Calculate CDF numerically
        cdf = [kde.integrate_box_1d(min_err, x) for x in grid]
        
        # Find index where CDF >= 1 - alpha
        idx = np.where(np.array(cdf) >= (1.0 - alpha))[0]
        if len(idx) > 0:
            return grid[idx[0]]
        return np.percentile(train_errors, 100 * (1 - alpha))
    except Exception:
        # Fallback to percentile if KDE fitting fails (e.g. singular covariance matrix)
        return np.percentile(train_errors, 100 * (1 - alpha))

def get_lognorm_threshold(train_errors, alpha=0.02):
    """
    Fits a Log-Normal distribution and returns the threshold at 1 - alpha.
    """
    try:
        # Avoid zero values for log-normal fitting
        errors_clean = train_errors[train_errors > 0]
        if len(errors_clean) < 5:
            errors_clean = train_errors + 1e-8
            
        shape, loc, scale = stats.lognorm.fit(errors_clean, floc=0)
        thresh = stats.lognorm.ppf(1.0 - alpha, shape, loc=loc, scale=scale)
        if np.isnan(thresh) or np.isinf(thresh):
            return np.percentile(train_errors, 100 * (1 - alpha))
        return thresh
    except Exception:
        return np.percentile(train_errors, 100 * (1 - alpha))

def get_hybrid_threshold(train_errors, alpha=0.02):
    """
    Fits Log-Normal and performs a Kolmogorov-Smirnov test.
    If p-value >= 0.05 (fits log-normal), returns log-normal threshold.
    Otherwise, falls back to KDE.
    """
    try:
        errors_clean = train_errors[train_errors > 0]
        if len(errors_clean) < 10:
            return get_kde_threshold(train_errors, alpha)
            
        shape, loc, scale = stats.lognorm.fit(errors_clean, floc=0)
        
        # KS-Test
        cdf_func = lambda x: stats.lognorm.cdf(x, shape, loc=loc, scale=scale)
        ks_res = stats.kstest(errors_clean, cdf_func)
        
        if ks_res.pvalue >= 0.05:
            thresh = stats.lognorm.ppf(1.0 - alpha, shape, loc=loc, scale=scale)
            if not np.isnan(thresh) and not np.isinf(thresh):
                return thresh
                
        return get_kde_threshold(train_errors, alpha)
    except Exception:
        return get_kde_threshold(train_errors, alpha)


def evaluate_dataset(dataset_name, epochs=10, batch_size=128):
    try:
        X_train, X_test, y_test = load_dataset_data(dataset_name)
        if len(X_train) == 0 or len(X_test) == 0:
            return None
            
        original_train_size = len(X_train)
        
        # Z-normalize
        X_train = z_normalize(X_train)
        X_test = z_normalize(X_test)
        
        # Oversample TRAIN
        if len(X_train) < 500:
            tiles = int(np.ceil(500 / len(X_train)))
            X_train = np.tile(X_train, (tiles, 1))[:500]
            
        # Tensors
        train_tensor = torch.tensor(np.expand_dims(X_train, axis=1), dtype=torch.float32)
        train_dataset = TensorDataset(train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        model = ConvAutoencoder().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # Train
        model.train()
        for epoch in range(epochs):
            for batch in train_loader:
                x_batch = batch[0].to(device)
                optimizer.zero_grad()
                recon = model(x_batch)
                loss = criterion(recon, x_batch)
                loss.backward()
                optimizer.step()
                
        # Evaluate TRAIN (original only)
        model.eval()
        train_errors = []
        with torch.no_grad():
            original_train_tensor = train_tensor[:original_train_size]
            eval_dataset = TensorDataset(original_train_tensor)
            eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
            for batch in eval_loader:
                x_batch = batch[0].to(device)
                recon = model(x_batch)
                errors = ((x_batch - recon) ** 2).mean(dim=(1, 2)).cpu().numpy()
                train_errors.extend(errors)
        train_errors = np.array(train_errors)
        
        # Evaluate TEST
        X_test_tensor = torch.tensor(np.expand_dims(X_test, axis=1), dtype=torch.float32)
        test_dataset = TensorDataset(X_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        test_errors = []
        with torch.no_grad():
            for batch in test_loader:
                x_batch = batch[0].to(device)
                recon = model(x_batch)
                errors = ((x_batch - recon) ** 2).mean(dim=(1, 2)).cpu().numpy()
                test_errors.extend(errors)
        test_errors = np.array(test_errors)
        
        # Calculate thresholds
        thresh_pct = np.percentile(train_errors, 98)
        thresh_kde = get_kde_threshold(train_errors, alpha=0.02)
        thresh_logn = get_lognorm_threshold(train_errors, alpha=0.02)
        thresh_hybrid = get_hybrid_threshold(train_errors, alpha=0.02)
        
        # Predictions
        pred_pct = (test_errors > thresh_pct).astype(int)
        pred_kde = (test_errors > thresh_kde).astype(int)
        pred_logn = (test_errors > thresh_logn).astype(int)
        pred_hybrid = (test_errors > thresh_hybrid).astype(int)
        
        # F1 Scores
        f1_pct = f1_score(y_test, pred_pct, zero_division=0)
        f1_kde = f1_score(y_test, pred_kde, zero_division=0)
        f1_logn = f1_score(y_test, pred_logn, zero_division=0)
        f1_hybrid = f1_score(y_test, pred_hybrid, zero_division=0)
        
        # Oracle F1
        precision, recall, thresholds = precision_recall_curve(y_test, test_errors)
        f1_scores = []
        for t in thresholds:
            preds = (test_errors > t).astype(int)
            f1_scores.append(f1_score(y_test, preds, zero_division=0))
        best_f1_idx = np.argmax(f1_scores) if f1_scores else 0
        f1_oracle = f1_scores[best_f1_idx] if f1_scores else 0
        
        return {
            "dataset": dataset_name,
            "f1_percentile": f1_pct,
            "f1_kde": f1_kde,
            "f1_lognorm": f1_logn,
            "f1_hybrid": f1_hybrid,
            "f1_oracle": f1_oracle
        }
    except Exception as e:
        return None

def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name FROM datasets 
        WHERE name NOT IN ('CornellWhaleChallenge', 'Wafer_normal_1')
        ORDER BY RANDOM()
        LIMIT 100
    """)
    dataset_names = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    print("scipy.stats 정교한 임계치 산정 비교 실험을 시작합니다 (무작위 100개 데이터셋).")
    print("목표: 백분위수 vs KDE 커널밀도 vs 로그정규분포 피팅 임계치 성능 비교\n")
    
    results = []
    for idx, name in enumerate(dataset_names, 1):
        res = evaluate_dataset(name, epochs=10)
        if res:
            results.append(res)
        if idx % 20 == 0 or idx == len(dataset_names):
            print(f"[{idx:3d}/100] 진행 완료...")
            
    df = pd.DataFrame([r for r in results if r is not None])
    
    print("\n" + "="*55)
    print("📊 임계값 설정 방법별 F1-Score 비교 성적표 (100개 평균)")
    print("="*55)
    print(f"1. 백분위수 기준 (Percentile 98%) F1 : 평균 {df['f1_percentile'].mean():.4f} | 중간값 {df['f1_percentile'].median():.4f}")
    print(f"2. KDE 커널밀도추정 (Gaussian KDE) F1  : 평균 {df['f1_kde'].mean():.4f} | 중간값 {df['f1_kde'].median():.4f}")
    print(f"3. 로그정규분포 피팅 (Log-Normal) F1   : 평균 {df['f1_lognorm'].mean():.4f} | 중간값 {df['f1_lognorm'].median():.4f}")
    print(f"4. 하이브리드 검정 (KS-Test Adaptive) F1: 평균 {df['f1_hybrid'].mean():.4f} | 중간값 {df['f1_hybrid'].median():.4f}")
    print(f"5. 오라클 완벽 매핑 (Oracle Upper Bound): 평균 {df['f1_oracle'].mean():.4f} | 중간값 {df['f1_oracle'].median():.4f}")
    print("="*55)

if __name__ == "__main__":
    main()
