import sqlite3
import argparse
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc, accuracy_score

# DB Path
DB_PATH = "/Users/minho/Documents/Dataset/univariate_ts.db"

def load_dataset_data(dataset_name):
    """
    Loads TRAIN and TEST split data for a given dataset name from SQLite.
    Returns:
        X_train (np.ndarray): Shape (N_train, series_len)
        X_test (np.ndarray): Shape (N_test, series_len)
        y_test (np.ndarray): Shape (N_test,) -> 0: normal, 1: anomaly
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

def evaluate_baseline(dataset_name):
    print(f"\n" + "="*60)
    print(f"데이터셋: {dataset_name} 베이스라인 평가 시작")
    print("="*60)
    
    try:
        X_train, X_test, y_test = load_dataset_data(dataset_name)
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        return
        
    print(f"- 학습용 정상 데이터 (TRAIN): {X_train.shape[0]}개 (시계열 길이: {X_train.shape[1]})")
    print(f"- 테스트용 데이터 (TEST)   : {X_test.shape[0]}개 (정상: {np.sum(y_test==0)}개, 이상치: {np.sum(y_test==1)}개)")
    
    if len(X_train) == 0 or len(X_test) == 0:
        print("경고: 학습 또는 테스트 데이터가 비어 있습니다.")
        return
        
    # --- 1. One-Class SVM ---
    print("\n[1] One-Class SVM 실행 중...")
    ocsvm = OneClassSVM(kernel='rbf', gamma='scale')
    ocsvm.fit(X_train)
    
    # Predict (-1 for outlier, 1 for inlier) -> Map to (1 for anomaly, 0 for normal)
    ocsvm_preds_raw = ocsvm.predict(X_test)
    ocsvm_preds = np.where(ocsvm_preds_raw == -1, 1, 0)
    
    # Score (distance to decision boundary) -> higher is more anomalous
    ocsvm_scores = -ocsvm.decision_function(X_test)
    
    # Metrics
    ocsvm_acc = accuracy_score(y_test, ocsvm_preds)
    ocsvm_f1 = f1_score(y_test, ocsvm_preds, zero_division=0)
    try:
        ocsvm_auc = roc_auc_score(y_test, ocsvm_scores)
    except ValueError:
        ocsvm_auc = 0.5
        
    precision, recall, _ = precision_recall_curve(y_test, ocsvm_scores)
    ocsvm_pr_auc = auc(recall, precision)
    
    print(f"  * Accuracy : {ocsvm_acc:.4f}")
    print(f"  * F1-Score : {ocsvm_f1:.4f}")
    print(f"  * AUC-ROC  : {ocsvm_auc:.4f}")
    print(f"  * AUC-PR   : {ocsvm_pr_auc:.4f}")
    
    # --- 2. Isolation Forest ---
    print("\n[2] Isolation Forest 실행 중...")
    # Train only on normal, but fit iForest
    iforest = IsolationForest(random_state=42, contamination='auto')
    iforest.fit(X_train)
    
    # Predict (-1 for outlier, 1 for inlier) -> Map to (1 for anomaly, 0 for normal)
    iforest_preds_raw = iforest.predict(X_test)
    iforest_preds = np.where(iforest_preds_raw == -1, 1, 0)
    
    # Score (anomaly score) -> higher is more anomalous
    iforest_scores = -iforest.score_samples(X_test)
    
    # Metrics
    iforest_acc = accuracy_score(y_test, iforest_preds)
    iforest_f1 = f1_score(y_test, iforest_preds, zero_division=0)
    try:
        iforest_auc = roc_auc_score(y_test, iforest_scores)
    except ValueError:
        iforest_auc = 0.5
        
    precision, recall, _ = precision_recall_curve(y_test, iforest_scores)
    iforest_pr_auc = auc(recall, precision)
    
    print(f"  * Accuracy : {iforest_acc:.4f}")
    print(f"  * F1-Score : {iforest_f1:.4f}")
    print(f"  * AUC-ROC  : {iforest_auc:.4f}")
    print(f"  * AUC-PR   : {iforest_pr_auc:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Unsupervised Anomaly Detection Baseline Models Evaluation")
    parser.add_argument("--dataset", type=str, default="CornellWhaleChallenge",
                        help="평가할 데이터셋 이름 (기본값: CornellWhaleChallenge)")
    args = parser.parse_args()
    
    evaluate_baseline(args.dataset)

if __name__ == "__main__":
    main()
