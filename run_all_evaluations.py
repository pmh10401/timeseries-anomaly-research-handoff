import os
import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc, accuracy_score

# DB Path
DB_PATH = "/Users/minho/Documents/Dataset/univariate_ts.db"
CSV_OUT_PATH = "/Users/minho/Documents/Dataset/autoencoder_results.csv"

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

def run_evaluation(dataset_name, epochs=10, batch_size=128):
    try:
        X_train, X_test, y_test = load_dataset_data(dataset_name)
        if len(X_train) == 0 or len(X_test) == 0:
            return None
            
        X_train = z_normalize(X_train)
        X_test = z_normalize(X_test)
        
        # Add channel dimension
        X_train = np.expand_dims(X_train, axis=1)
        X_test = np.expand_dims(X_test, axis=1)
        
        # Loader
        train_tensor = torch.tensor(X_train, dtype=torch.float32)
        train_dataset = TensorDataset(train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Model
        model = ConvAutoencoder().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # Train
        model.train()
        for epoch in range(epochs):
            for batch in train_loader:
                x_batch = batch[0].to(device)
                optimizer.zero_grad()
                reconstructed = model(x_batch)
                loss = criterion(reconstructed, x_batch)
                loss.backward()
                optimizer.step()
                
        # Evaluate TRAIN for threshold
        model.eval()
        train_errors = []
        with torch.no_grad():
            train_loader_eval = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            for batch in train_loader_eval:
                x_batch = batch[0].to(device)
                recon = model(x_batch)
                errors = ((x_batch - recon) ** 2).mean(dim=(1, 2)).cpu().numpy()
                train_errors.extend(errors)
        train_errors = np.array(train_errors)
        
        # Evaluate TEST
        test_tensor = torch.tensor(X_test, dtype=torch.float32)
        test_dataset = TensorDataset(test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        test_errors = []
        with torch.no_grad():
            for batch in test_loader:
                x_batch = batch[0].to(device)
                recon = model(x_batch)
                errors = ((x_batch - recon) ** 2).mean(dim=(1, 2)).cpu().numpy()
                test_errors.extend(errors)
        test_errors = np.array(test_errors)
        
        anomaly_scores = test_errors
        
        # Metrics
        auc_roc = roc_auc_score(y_test, anomaly_scores)
        precision, recall, thresholds = precision_recall_curve(y_test, anomaly_scores)
        auc_pr = auc(recall, precision)
        
        # Unsupervised F1
        unsupervised_thresh = np.percentile(train_errors, 98)
        y_pred_unsupervised = (anomaly_scores > unsupervised_thresh).astype(int)
        f1_unsupervised = f1_score(y_test, y_pred_unsupervised, zero_division=0)
        acc_unsupervised = accuracy_score(y_test, y_pred_unsupervised)
        
        # Oracle F1
        f1_scores = []
        for t in thresholds:
            preds = (anomaly_scores > t).astype(int)
            f1_scores.append(f1_score(y_test, preds, zero_division=0))
        best_f1_idx = np.argmax(f1_scores) if f1_scores else 0
        best_f1 = f1_scores[best_f1_idx] if f1_scores else 0
        
        return {
            "dataset_name": dataset_name,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "sequence_length": X_train.shape[2],
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "unsupervised_f1": f1_unsupervised,
            "unsupervised_acc": acc_unsupervised,
            "oracle_f1": best_f1
        }
    except Exception as e:
        # Some tiny datasets might fail or throw value errors due to extreme shapes
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
    print(f"총 {total_datasets}개의 데이터셋에 대해 1D Conv-Autoencoder 평가를 시작합니다.")
    print(f"결과는 {CSV_OUT_PATH} 에 주기적으로 저장됩니다.\n")
    
    results = []
    
    for idx, name in enumerate(dataset_names, 1):
        res = run_evaluation(name, epochs=10)
        if res:
            results.append(res)
            
        # Periodic output & save
        if idx % 50 == 0 or idx == total_datasets:
            df = pd.DataFrame([r for r in results if r is not None])
            df.to_csv(CSV_OUT_PATH, index=False)
            
            # Print current stats
            valid_results = [r for r in results if r is not None]
            if valid_results:
                avg_auc = np.mean([r['auc_roc'] for r in valid_results])
                avg_f1 = np.mean([r['unsupervised_f1'] for r in valid_results])
                print(f"[{idx:4d}/{total_datasets:4d}] 진행 완료... (현재 완료된 {len(valid_results)}개 데이터셋 평균 AUC-ROC: {avg_auc:.4f}, 평균 F1: {avg_f1:.4f})")
                
    # Final Output
    df = pd.DataFrame([r for r in results if r is not None])
    df.to_csv(CSV_OUT_PATH, index=False)
    
    print("\n" + "="*50)
    print("모든 데이터셋에 대한 평가 완료!")
    print(f"- 결과 저장 경로: {CSV_OUT_PATH}")
    print(f"- 성공적으로 평가된 데이터셋 수: {len(df)}")
    print(f"- 전체 데이터셋 평균 AUC-ROC: {df['auc_roc'].mean():.4f}")
    print(f"- 전체 데이터셋 평균 AUC-PR : {df['auc_pr'].mean():.4f}")
    print(f"- 전체 데이터셋 평균 F1-Score: {df['unsupervised_f1'].mean():.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
