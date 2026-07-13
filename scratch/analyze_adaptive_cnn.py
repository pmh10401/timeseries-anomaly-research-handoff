import os
import pandas as pd
import numpy as np

# Path
DATA_DIR = "/Users/minho/Documents/Dataset"

def main():
    # Load VAE EVT (fixed topology) and VAE Adaptive CNN (dynamic topology)
    fixed_path = os.path.join(DATA_DIR, "vae_results_evt.csv")
    adaptive_path = os.path.join(DATA_DIR, "vae_results_adaptive_cnn.csv")
    
    if not os.path.exists(fixed_path) or not os.path.exists(adaptive_path):
        print("Error: Required CSV files do not exist.")
        return
        
    df_fixed = pd.read_csv(fixed_path).set_index("dataset_name")
    df_adaptive = pd.read_csv(adaptive_path).set_index("dataset_name")
    
    # Align rows
    common_idx = df_fixed.index.intersection(df_adaptive.index)
    df_fixed = df_fixed.loc[common_idx]
    df_adaptive = df_adaptive.loc[common_idx]
    
    print(f"공통 매핑 데이터셋 개수: {len(common_idx)}")
    
    # 1. Split into Short (L < 150) and Long (L >= 150) sequence subsets
    # sequence_length is stored in df_adaptive as 'sequence_length' or 'sequence_length' equivalent
    # Let's verify column name. In adaptive CSV, it was saved as 'sequence_length'
    seq_col = "sequence_length" if "sequence_length" in df_adaptive.columns else "original_train_size" # fallback if not matched
    # Actually, in run_all_adaptive_cnn_evaluations.py we saved it as 'sequence_length'
    
    short_mask = df_adaptive["sequence_length"] < 150
    long_mask = df_adaptive["sequence_length"] >= 150
    
    num_short = short_mask.sum()
    num_long = long_mask.sum()
    
    print(f"짧은 시계열 데이터셋 개수 (L < 150): {num_short}")
    print(f"긴 시계열 데이터셋 개수 (L >= 150): {num_long}")
    
    # 2. Extract EVT F1 statistics for Short subset
    fixed_short_f1 = df_fixed.loc[short_mask, "f1_evt"].mean()
    adaptive_short_f1 = df_adaptive.loc[short_mask, "f1_evt"].mean()
    
    fixed_short_auc = df_fixed.loc[short_mask, "auc_roc"].mean()
    adaptive_short_auc = df_adaptive.loc[short_mask, "auc_roc"].mean()
    
    # 3. Extract EVT F1 statistics for Long subset
    fixed_long_f1 = df_fixed.loc[long_mask, "f1_evt"].mean()
    adaptive_long_f1 = df_adaptive.loc[long_mask, "f1_evt"].mean()
    
    fixed_long_auc = df_fixed.loc[long_mask, "auc_roc"].mean()
    adaptive_long_auc = df_adaptive.loc[long_mask, "auc_roc"].mean()
    
    print(f"\n[짧은 시계열 (L < 150) 성능 대조]")
    print(f"- 고정형 VAE (EVT F1)   : {fixed_short_f1:.4f} (AUC: {fixed_short_auc:.4f})")
    print(f"- 적응형 VAE (EVT F1)   : {adaptive_short_f1:.4f} (AUC: {adaptive_short_auc:.4f})")
    print(f"- 성능 개선 폭          : { (adaptive_short_f1 - fixed_short_f1)*100:+.2f}%p (AUC: { (adaptive_short_auc - fixed_short_auc)*100:+.2f}%p)")
    
    print(f"\n[긴 시계열 (L >= 150) 성능 대조]")
    print(f"- 고정형 VAE (EVT F1)   : {fixed_long_f1:.4f} (AUC: {fixed_long_auc:.4f})")
    print(f"- 적응형 VAE (EVT F1)   : {adaptive_long_f1:.4f} (AUC: {adaptive_long_auc:.4f})")
    print(f"- 성능 개선 폭          : { (adaptive_long_f1 - fixed_long_f1)*100:+.2f}%p (AUC: { (adaptive_long_auc - fixed_long_auc)*100:+.2f}%p)")

if __name__ == "__main__":
    main()
