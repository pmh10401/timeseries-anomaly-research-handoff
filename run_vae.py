import sqlite3
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc, accuracy_score

# DB Path
DB_PATH = "/Users/minho/Documents/Dataset/univariate_ts.db"

# Check GPU acceleration
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("-> GPU 가속 활성화: Apple Metal Performance Shaders (MPS) 사용")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("-> GPU 가속 활성화: CUDA 사용")
else:
    device = torch.device("cpu")
    print("-> GPU 가속 비활성화: CPU 사용")

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
            
        print(f"  [Model Init] Sequence Length: {seq_len} -> Conv Output Length: {self.conv_out_len} -> Flat Dim: {self.flat_dim}")
        
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

def train_and_evaluate_vae(dataset_name="Wafer_normal_1", epochs=20, batch_size=128, beta=0.001):
    print(f"\n" + "="*60)
    print(f"데이터셋: {dataset_name} 1D Conv-VAE (Variational AE - 시간축 보존 레이어) 학습 및 평가")
    print("="*60)
    
    # 1. Load & Preprocess
    X_train, X_test, y_test = load_dataset_data(dataset_name)
    
    X_train = z_normalize(X_train)
    X_test = z_normalize(X_test)
    
    # Oversample TRAIN to 500 if smaller
    if len(X_train) < 500:
        tiles = int(np.ceil(500 / len(X_train)))
        X_train = np.tile(X_train, (tiles, 1))[:500]
        
    print(f"- 학습용 정상 데이터: {X_train.shape[0]}개 (길이: {X_train.shape[1]})")
    print(f"- 테스트용 데이터   : {X_test.shape[0]}개 (정상: {np.sum(y_test==0)}개, 이상치: {np.sum(y_test==1)}개)")
    
    # Add channel dimension
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    
    # Loader
    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Define Model (passes target sequence length to preserve spatial dimension)
    model = ConvVAE(latent_dim=128, seq_len=X_train.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 3. Train Loop
    print("\n--- VAE 학습 시작 (Reconstruction Loss + KL Divergence) ---")
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_total_loss = 0.0
        
        for batch in train_loader:
            x_batch = batch[0].to(device)
            
            optimizer.zero_grad()
            recon, mu, logvar = model(x_batch)
            
            # Reconstruction Loss
            recon_loss = nn.functional.mse_loss(recon, x_batch, reduction='mean')
            
            # KL Divergence Loss
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            # Total Loss
            total_loss = recon_loss + beta * kl_loss
            
            total_loss.backward()
            optimizer.step()
            
            epoch_recon_loss += recon_loss.item() * x_batch.size(0)
            epoch_kl_loss += kl_loss.item() * x_batch.size(0)
            epoch_total_loss += total_loss.item() * x_batch.size(0)
            
        n_samples = len(train_loader.dataset)
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Total: {epoch_total_loss/n_samples:.6f} (Recon: {epoch_recon_loss/n_samples:.6f}, KL: {epoch_kl_loss/n_samples:.6f})")
        
    # 4. Evaluate on TEST set
    model.eval()
    test_tensor = torch.tensor(X_test, dtype=torch.float32)
    test_dataset = TensorDataset(test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    anomaly_scores = []
    with torch.no_grad():
        for batch in test_loader:
            x_batch = batch[0].to(device)
            recon, mu, logvar = model(x_batch)
            
            # Pointwise reconstruction error
            recon_errs = ((x_batch - recon) ** 2).mean(dim=(1, 2))
            
            # Pointwise KL divergence per sample
            kl_errs = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            
            # Anomaly Score = Recon Loss + Beta * KL Loss
            scores = recon_errs + beta * kl_errs
            anomaly_scores.extend(scores.cpu().numpy())
            
    anomaly_scores = np.array(anomaly_scores)
    
    # 5. Calculate Metrics
    auc_roc = roc_auc_score(y_test, anomaly_scores)
    precision, recall, thresholds = precision_recall_curve(y_test, anomaly_scores)
    auc_pr = auc(recall, precision)
    
    # Oracle F1
    f1_scores = []
    for t in thresholds:
        preds = (anomaly_scores > t).astype(int)
        f1_scores.append(f1_score(y_test, preds, zero_division=0))
    best_f1_idx = np.argmax(f1_scores) if f1_scores else 0
    best_f1 = f1_scores[best_f1_idx] if f1_scores else 0
    
    print("\n--- VAE 최종 평가 결과 ---")
    print(f"  * AUC-ROC             : {auc_roc:.4f}")
    print(f"  * AUC-PR (PR-AUC)     : {auc_pr:.4f}")
    print(f"  * Oracle Best F1-Score: {best_f1:.4f}")
    print("="*60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="1D Conv-VAE Evaluation")
    parser.add_argument("--dataset", type=str, default="Wafer_normal_1",
                        help="평가할 데이터셋 이름 (기본값: Wafer_normal_1)")
    args = parser.parse_args()
    
    train_and_evaluate_vae(dataset_name=args.dataset)
