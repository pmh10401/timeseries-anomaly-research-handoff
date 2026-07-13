import os
import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc, accuracy_score
import scipy.stats as stats

# DB Path
DB_PATH = "/Users/minho/Documents/Dataset/univariate_ts.db"
CSV_OUT_PATH = "/Users/minho/Documents/Dataset/threshold_comparison_results.csv"

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
    try:
        kde = stats.gaussian_kde(train_errors)
        min_err = np.min(train_errors)
        max_err = np.max(train_errors)
        grid = np.linspace(min_err, max_err * 2, 1000)
        cdf = [kde.integrate_box_1d(min_err, x) for x in grid]
        idx = np.where(np.array(cdf) >= (1.0 - alpha))[0]
        if len(idx) > 0:
            return grid[idx[0]]
        return np.percentile(train_errors, 100 * (1 - alpha))
    except Exception:
        return np.percentile(train_errors, 100 * (1 - alpha))

def get_lognorm_threshold(train_errors, alpha=0.02):
    try:
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

def get_size_hybrid_threshold(train_errors, original_train_size, alpha=0.02):
    """
    If sample size < 100: Use safe KDE (or percentile).
    If sample size >= 100: Use Log-Normal fitting.
    """
    if original_train_size < 100:
        return get_kde_threshold(train_errors, alpha)
    else:
        return get_lognorm_threshold(train_errors, alpha)

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
        thresh_hybrid = get_size_hybrid_threshold(train_errors, original_train_size, alpha=0.02)
        
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
            "dataset_name": dataset_name,
            "original_train_size": original_train_size,
            "f1_percentile": f1_pct,
            "f1_kde": f1_kde,
            "f1_lognorm": f1_logn,
            "f1_hybrid_size": f1_hybrid,
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
        ORDER BY name
    """)
    dataset_names = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    total_datasets = len(dataset_names)
    print(f"가속 디바이스: {device}")
    print(f"총 {total_datasets}개의 전체 데이터셋에 대해 '데이터 크기 기반 하이브리드 임계치' 배치 비교 평가를 시작합니다.")
    print(f"결과는 {CSV_OUT_PATH} 에 주기적으로 저장됩니다.\n")
    
    results = []
    for idx, name in enumerate(dataset_names, 1):
        res = evaluate_dataset(name, epochs=10)
        if res:
            results.append(res)
            
        # Periodic output & save
        if idx % 50 == 0 or idx == total_datasets:
            df = pd.DataFrame([r for r in results if r is not None])
            df.to_csv(CSV_OUT_PATH, index=False)
            
            valid_results = [r for r in results if r is not None]
            if valid_results:
                avg_pct = np.mean([r['f1_percentile'] for r in valid_results])
                avg_kde = np.mean([r['f1_kde'] for r in valid_results])
                avg_logn = np.mean([r['f1_lognorm'] for r in valid_results])
                avg_hyb = np.mean([r['f1_hybrid_size'] for r in valid_results])
                avg_oracle = np.mean([r['f1_oracle'] for r in valid_results])
                
                print(f"[{idx:4d}/{total_datasets:4d}] 진행 완료... (현재 완료 {len(valid_results)}개 평균 F1 - Baseline: {avg_pct:.4f}, KDE: {avg_kde:.4f}, LogNorm: {avg_logn:.4f}, Hybrid(Size): {avg_hyb:.4f}, Oracle: {avg_oracle:.4f})")
                
    # Final Output
    df = pd.DataFrame([r for r in results if r is not None])
    df.to_csv(CSV_OUT_PATH, index=False)
    
    print("\n" + "="*70)
    print("전체 데이터셋 임계값 고도화 평가 완료!")
    print(f"- 결과 저장 경로: {CSV_OUT_PATH}")
    print(f"- 성공적으로 평가된 데이터셋 수: {len(df)}")
    print(f"- [1] 백분위수 98% (Baseline)  F1: 평균 {df['f1_percentile'].mean():.4f} | 중간값 {df['f1_percentile'].median():.4f}")
    print(f"- [2] KDE 커널밀도추정 F1        F1: 평균 {df['f1_kde'].mean():.4f} | 중간값 {df['f1_kde'].median():.4f}")
    print(f"- [3] 로그정규분포 피팅 F1       F1: 평균 {df['f1_lognorm'].mean():.4f} | 중간값 {df['f1_lognorm'].median():.4f}")
    print(f"- [4] 데이터 크기 하이브리드 F1   F1: 평균 {df['f1_hybrid_size'].mean():.4f} | 중간값 {df['f1_hybrid_size'].median():.4f}")
    print(f"- [5] 오라클 완벽 매핑 F1          F1: 평균 {df['f1_oracle'].mean():.4f} | 중간값 {df['f1_oracle'].median():.4f}")
    print("="*70)

if __name__ == "__main__":
    main()
