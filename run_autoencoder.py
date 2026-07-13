import sqlite3
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc, accuracy_score

# DB Path
DB_PATH = "/Users/minho/Documents/Dataset/univariate_ts.db"

# Check GPU acceleration (MPS for Apple Silicon)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("-> GPU 가속 활성화: Apple Metal Performance Shaders (MPS) 사용")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("-> GPU 가속 활성화: CUDA 사용")
else:
    device = torch.device("cpu")
    print("-> GPU 가속 비활성화: CPU 사용")

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Input shape: (batch, 1, 4000)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7), # (batch, 16, 2000)
            nn.ReLU(True),
            nn.Conv1d(16, 32, kernel_size=15, stride=2, padding=7), # (batch, 32, 1000)
            nn.ReLU(True),
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7), # (batch, 64, 500)
            nn.ReLU(True),
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7), # (batch, 128, 250)
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=16, stride=2, padding=7), # (batch, 64, 500)
            nn.ReLU(True),
            nn.ConvTranspose1d(64, 32, kernel_size=16, stride=2, padding=7), # (batch, 32, 1000)
            nn.ReLU(True),
            nn.ConvTranspose1d(32, 16, kernel_size=16, stride=2, padding=7), # (batch, 16, 2000)
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 1, kernel_size=16, stride=2, padding=7), # (batch, 1, 4000)
        )

    def forward(self, x):
        target_len = x.size(2)
        x = self.encoder(x)
        x = self.decoder(x)
        if x.size(2) != target_len:
            x = nn.functional.interpolate(x, size=target_len, mode='linear', align_corners=False)
        return x

def load_dataset_data(dataset_name):
    """
    Loads TRAIN and TEST split data for a given dataset name from SQLite.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Fetch TRAIN data
    cursor.execute("""
        SELECT i.values_blob
        FROM instances i
        JOIN datasets d ON i.dataset_id = d.id
        WHERE d.name = ? AND i.split = 'TRAIN'
        ORDER BY i.instance_index
    """, (dataset_name,))
    train_rows = cursor.fetchall()
    X_train = [np.frombuffer(row[0], dtype=np.float32) for row in train_rows]
    
    # 2. Fetch TEST data
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
    """
    Instance-level Z-normalization: (x - mean) / std for each sequence.
    """
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std) # Avoid division by zero
    return (X - mean) / std

def train_and_evaluate(dataset_name="CornellWhaleChallenge", epochs=20, batch_size=128):
    print(f"\n" + "="*60)
    print(f"데이터셋: {dataset_name} 1D Conv-Autoencoder 학습 및 평가 (Z-스코어 정규화 적용)")
    print("="*60)
    
    # 1. Load data
    X_train, X_test, y_test = load_dataset_data(dataset_name)
    
    # Apply Z-normalization
    X_train = z_normalize(X_train)
    X_test = z_normalize(X_test)
    
    print(f"- 학습용 정상 데이터: {X_train.shape[0]}개 (길이: {X_train.shape[1]})")
    print(f"- 테스트용 데이터   : {X_test.shape[0]}개 (정상: {np.sum(y_test==0)}개, 이상치: {np.sum(y_test==1)}개)")
    print(f"- 데이터 범위 검증   : Train Min={np.min(X_train):.2f}, Max={np.max(X_train):.2f}, Mean={np.mean(X_train):.4f}, Std={np.std(X_train):.4f}")
    
    # Add channel dimension: (N, 1, L)
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    
    # Create DataLoader
    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Define model, optimizer, loss
    model = ConvAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # 3. Train loop
    print("\n--- Autoencoder 학습 시작 ---")
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch in train_loader:
            x_batch = batch[0].to(device)
            
            optimizer.zero_grad()
            reconstructed = model(x_batch)
            loss = criterion(reconstructed, x_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * x_batch.size(0)
            
        epoch_loss /= len(train_loader.dataset)
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Loss: {epoch_loss:.6f}")
        
    # 4. Compute reconstruction errors on TRAIN for threshold selection
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
    
    # 5. Evaluate on TEST set
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
    
    # Unsupervised Threshold: 98th percentile of train reconstruction error
    unsupervised_thresh = np.percentile(train_errors, 98)
    y_pred_unsupervised = (anomaly_scores > unsupervised_thresh).astype(int)
    
    f1_unsupervised = f1_score(y_test, y_pred_unsupervised, zero_division=0)
    acc_unsupervised = accuracy_score(y_test, y_pred_unsupervised)
    
    # Oracle Threshold (F1-maximizing threshold on TEST)
    f1_scores = []
    for t in thresholds:
        preds = (anomaly_scores > t).astype(int)
        f1_scores.append(f1_score(y_test, preds, zero_division=0))
    
    best_f1_idx = np.argmax(f1_scores) if f1_scores else 0
    best_thresh = thresholds[best_f1_idx] if thresholds.size > 0 else 0
    best_f1 = f1_scores[best_f1_idx] if f1_scores else 0
    y_pred_oracle = (anomaly_scores > best_thresh).astype(int)
    acc_oracle = accuracy_score(y_test, y_pred_oracle)
    
    print("\n--- 최종 평가 결과 ---")
    print(f"  * AUC-ROC             : {auc_roc:.4f}")
    print(f"  * AUC-PR (PR-AUC)     : {auc_pr:.4f}")
    print(f"\n  [Unsupervised Threshold (Train 98% 기준)]")
    print(f"    - Threshold         : {unsupervised_thresh:.6f}")
    print(f"    - Accuracy          : {acc_unsupervised:.4f}")
    print(f"    - F1-Score          : {f1_unsupervised:.4f}")
    print(f"\n  [Oracle Threshold (Test F1 극대화 기준 - Upper Bound)]")
    print(f"    - Best Threshold    : {best_thresh:.6f}")
    print(f"    - Accuracy          : {acc_oracle:.4f}")
    print(f"    - F1-Score          : {best_f1:.4f}")
    print("="*60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="1D Conv-AE Evaluation")
    parser.add_argument("--dataset", type=str, default="CornellWhaleChallenge",
                        help="평가할 데이터셋 이름 (기본값: CornellWhaleChallenge)")
    args = parser.parse_args()
    
    train_and_evaluate(dataset_name=args.dataset)
