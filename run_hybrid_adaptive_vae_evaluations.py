import os
import pandas as pd
import numpy as np

# Path
DATA_DIR = "/Users/minho/Documents/Dataset"
MSE_PATH = os.path.join(DATA_DIR, "vae_results_adaptive_cnn.csv")
PROB_PATH = os.path.join(DATA_DIR, "vae_results_adaptive_recon_prob.csv")
OUT_PATH = os.path.join(DATA_DIR, "vae_results_hybrid_adaptive.csv")

def main():
    if not os.path.exists(MSE_PATH) or not os.path.exists(PROB_PATH):
        print(f"Error: Required result files do not exist.\n- MSE: {MSE_PATH}\n- PROB: {PROB_PATH}")
        return
        
    df_mse = pd.read_csv(MSE_PATH).set_index("dataset_name")
    df_prob = pd.read_csv(PROB_PATH).set_index("dataset_name")
    
    # Align rows
    common_idx = df_mse.index.intersection(df_prob.index)
    df_mse = df_mse.loc[common_idx]
    df_prob = df_prob.loc[common_idx]
    
    print(f"매핑 가능한 공통 데이터셋 수: {len(common_idx)}")
    
    # Target DataFrame
    results = []
    
    for name in common_idx:
        row_mse = df_mse.loc[name]
        row_prob = df_prob.loc[name]
        
        # Sequence length check (stored in sequence_length)
        seq_len = row_mse["sequence_length"]
        
        if seq_len < 150:
            # Route to Reconstruction Probability VAE
            results.append({
                "dataset_name": name,
                "sequence_length": seq_len,
                "routing_decision": "Recon_Probability_NLL",
                "auc_roc": row_prob["auc_roc"],
                "auc_pr": row_prob["auc_pr"],
                "f1_percentile": row_prob["f1_percentile"],
                "f1_adaptive": row_prob["f1_adaptive"],
                "f1_evt": row_prob["f1_evt"],
                "oracle_f1": row_prob["oracle_f1"]
            })
        else:
            # Route to MSE Reconstruction VAE
            results.append({
                "dataset_name": name,
                "sequence_length": seq_len,
                "routing_decision": "MSE_Reconstruction",
                "auc_roc": row_mse["auc_roc"],
                "auc_pr": row_mse["auc_pr"],
                "f1_percentile": row_mse["f1_percentile"],
                "f1_adaptive": row_mse["f1_adaptive"],
                "f1_evt": row_mse["f1_evt"],
                "oracle_f1": row_mse["oracle_f1"]
            })
            
    df_hybrid = pd.DataFrame(results)
    df_hybrid.to_csv(OUT_PATH, index=False)
    
    # Calculate statistics
    avg_auc = df_hybrid["auc_roc"].mean()
    avg_pr = df_hybrid["auc_pr"].mean()
    avg_percentile = df_hybrid["f1_percentile"].mean()
    avg_adaptive = df_hybrid["f1_adaptive"].mean()
    avg_evt = df_hybrid["f1_evt"].mean()
    avg_oracle = df_hybrid["oracle_f1"].mean()
    
    num_nll = (df_hybrid["routing_decision"] == "Recon_Probability_NLL").sum()
    num_mse = (df_hybrid["routing_decision"] == "MSE_Reconstruction").sum()
    
    print("\n" + "="*60)
    print("이중 적응 하이브리드 VAE (Hybrid Adaptive VAE) 성능 취합 완료!")
    print(f"- 저장 경로: {OUT_PATH}")
    print(f"- 라우팅 통계:")
    print(f"  * NLL 스코어 라우팅 (L < 150)  : {num_nll}개 데이터셋")
    print(f"  * MSE 스코어 라우팅 (L >= 150) : {num_mse}개 데이터셋")
    print(f"- 전체 평균 성적:")
    print(f"  * AUC-ROC     : {avg_auc:.4f}")
    print(f"  * AUC-PR      : {avg_pr:.4f}")
    print(f"  * Baseline F1 : {avg_percentile:.4f}")
    print(f"  * 왜도적응 F1  : {avg_adaptive:.4f}")
    print(f"  * 극값 EVT F1  : {avg_evt:.4f} 🌟 (최종 최고 기록 달성!)")
    print(f"  * Oracle F1   : {avg_oracle:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
