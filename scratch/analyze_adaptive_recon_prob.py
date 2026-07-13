import os
import pandas as pd
import numpy as np

# Path
DATA_DIR = "/Users/minho/Documents/Dataset"

def main():
    # Load VAE Adaptive MSE and VAE Adaptive Recon Prob
    mse_path = os.path.join(DATA_DIR, "vae_results_adaptive_cnn.csv")
    prob_path = os.path.join(DATA_DIR, "vae_results_adaptive_recon_prob.csv")
    
    if not os.path.exists(mse_path) or not os.path.exists(prob_path):
        print("Error: Required CSV files do not exist.")
        return
        
    df_mse = pd.read_csv(mse_path).set_index("dataset_name")
    df_prob = pd.read_csv(prob_path).set_index("dataset_name")
    
    # Align rows
    common_idx = df_mse.index.intersection(df_prob.index)
    df_mse = df_mse.loc[common_idx]
    df_prob = df_prob.loc[common_idx]
    
    print(f"공통 매핑 데이터셋 개수: {len(common_idx)}")
    
    short_mask = df_prob["sequence_length"] < 150
    long_mask = df_prob["sequence_length"] >= 150
    
    num_short = short_mask.sum()
    num_long = long_mask.sum()
    
    # Extract stats for Short subset
    mse_short_f1 = df_mse.loc[short_mask, "f1_evt"].mean()
    prob_short_f1 = df_prob.loc[short_mask, "f1_evt"].mean()
    
    mse_short_auc = df_mse.loc[short_mask, "auc_roc"].mean()
    prob_short_auc = df_prob.loc[short_mask, "auc_roc"].mean()
    
    # Extract stats for Long subset
    mse_long_f1 = df_mse.loc[long_mask, "f1_evt"].mean()
    prob_long_f1 = df_prob.loc[long_mask, "f1_evt"].mean()
    
    mse_long_auc = df_mse.loc[long_mask, "auc_roc"].mean()
    prob_long_auc = df_prob.loc[long_mask, "auc_roc"].mean()
    
    print(f"\n[짧은 시계열 (L < 150) 성능 대조]")
    print(f"- 적응형 MSE VAE (EVT F1)      : {mse_short_f1:.4f} (AUC: {mse_short_auc:.4f})")
    print(f"- 적응형 복원확률 VAE (EVT F1) : {prob_short_f1:.4f} (AUC: {prob_short_auc:.4f})")
    print(f"- 성능 차이                    : { (prob_short_f1 - mse_short_f1)*100:+.2f}%p (AUC: { (prob_short_auc - mse_short_auc)*100:+.2f}%p)")
    
    print(f"\n[긴 시계열 (L >= 150) 성능 대조]")
    print(f"- 적응형 MSE VAE (EVT F1)      : {mse_long_f1:.4f} (AUC: {mse_long_auc:.4f})")
    print(f"- 적응형 복원확률 VAE (EVT F1) : {prob_long_f1:.4f} (AUC: {prob_long_auc:.4f})")
    print(f"- 성능 차이                    : { (prob_long_f1 - mse_long_f1)*100:+.2f}%p (AUC: { (prob_long_auc - mse_long_auc)*100:+.2f}%p)")

if __name__ == "__main__":
    main()
