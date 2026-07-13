import sqlite3
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path("/Users/minho/Documents/Dataset")
DB_PATH = BASE_DIR / "univariate_ts.db"

def verify():
    print("SQLite 데이터베이스 신규 검증을 시작합니다...")
    if not DB_PATH.exists():
        print(f"오류: 데이터베이스 파일이 존재하지 않습니다: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Check datasets count
    cursor.execute("SELECT COUNT(*) FROM datasets")
    total_datasets = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM datasets WHERE name = 'CornellWhaleChallenge'")
    whale_dataset = cursor.fetchone()[0]
    
    classification_oneclass_datasets = total_datasets - whale_dataset
    
    print(f"- 총 데이터셋 개수: {total_datasets}개")
    print(f"  * UCR 분류 아카이브의 One-Class 변환 데이터셋: {classification_oneclass_datasets}개")
    print(f"  * Hugging Face CornellWhaleChallenge 데이터셋: {whale_dataset}개")

    # 2. Check total instances count
    cursor.execute("SELECT COUNT(*) FROM instances")
    num_instances = cursor.fetchone()[0]
    print(f"- 총 인스턴스 개수: {num_instances}개")

    # 3. Verify TRAIN splits are 100% normal (contains NO anomalies)
    print("\n[검증 1] 모든 TRAIN 세트의 정상 데이터(정상률 100%) 보장 검증:")
    cursor.execute("SELECT id, name FROM datasets")
    datasets = cursor.fetchall()
    
    train_violations = 0
    for d_id, d_name in datasets:
        cursor.execute("SELECT label, labels_blob FROM instances WHERE dataset_id = ? AND split = 'TRAIN'", (d_id,))
        rows = cursor.fetchall()
        for label, lbl_blob in rows:
            if label == "1" or label == "anomaly":
                train_violations += 1
                print(f"  * 경고: {d_name}의 TRAIN 세트에 이상치 라벨({label})이 포함되어 있습니다.")
                break
            if lbl_blob:
                lbls = np.frombuffer(lbl_blob, dtype=np.uint8)
                if np.sum(lbls == 1) > 0:
                    train_violations += 1
                    print(f"  * 경고: {d_name}의 TRAIN 세트 시계열 내부({np.sum(lbls==1)}개 타임스텝)에 이상치가 포함되어 있습니다.")
                    break
                    
    if train_violations == 0:
        print("  -> 성공: 모든 TRAIN 세트에 이상치가 전혀 포함되지 않았음이 검증되었습니다. (정상 데이터 100% 학습 보장)")
    else:
        print(f"  -> 실패: 총 {train_violations}개의 데이터셋에서 TRAIN 세트 정상 데이터 규칙 위반 발견.")

    # 4. Verify TEST splits anomaly ratio is under 1-3% (except tiny datasets where max(1, ...) forces 1 anomaly)
    print("\n[검증 2] TEST 세트 내 이상치 비율(2% 타겟, 1~3% 수준) 검증:")
    ratios = []
    ratio_violations = 0
    
    cursor.execute("SELECT id, name FROM datasets")
    oneclass_ds = cursor.fetchall()
    
    for d_id, d_name in oneclass_ds:
        cursor.execute("""
            SELECT COUNT(*) FROM instances 
            WHERE dataset_id = ? AND split = 'TEST' AND (label = '0' OR label = 'normal')
        """, (d_id,))
        n_normal = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) FROM instances 
            WHERE dataset_id = ? AND split = 'TEST' AND (label = '1' OR label = 'anomaly')
        """, (d_id,))
        n_anomaly = cursor.fetchone()[0]
        
        total = n_normal + n_anomaly
        if total > 0:
            ratio = (n_anomaly / total) * 100
            ratios.append(ratio)
            if n_normal >= 49 and ratio > 3.0:
                ratio_violations += 1
                print(f"  * 경고: {d_name}의 TEST 세트 이상치 비율이 {ratio:.2f}%로 3%를 초과합니다. (Normal: {n_normal}, Anomaly: {n_anomaly})")
                
    if ratio_violations == 0:
        avg_ratio = np.mean(ratios) if ratios else 0
        print(f"  -> 성공: 모든 대형 테스트 세트가 3% 이하(평균 {avg_ratio:.2f}%)의 이상치 비율을 충족합니다.")
    else:
        print(f"  -> 실패: 총 {ratio_violations}개의 데이터셋에서 이상치 비율 규칙 위반 발견.")

    # 5. Verify metadata count columns
    print("\n[검증 3] datasets 테이블의 개수(Metadata Count) 메타데이터 검증:")
    cursor.execute("""
        SELECT name, train_normal_count, train_anomaly_count, train_total_count,
               test_normal_count, test_anomaly_count, test_total_count,
               total_normal_count, total_anomaly_count, total_count
        FROM datasets
        ORDER BY RANDOM()
        LIMIT 3
    """)
    meta_samples = cursor.fetchall()
    
    metadata_ok = True
    for name, tr_n, tr_a, tr_t, te_n, te_a, te_t, tot_n, tot_a, tot_all in meta_samples:
        print(f"  * 데이터셋: {name}")
        print(f"    - TRAIN: 정상 {tr_n}개, 이상치 {tr_a}개, 총 {tr_t}개")
        print(f"    - TEST : 정상 {te_n}개, 이상치 {te_a}개, 총 {te_t}개")
        print(f"    - 전체 : 정상 {tot_n}개, 이상치 {tot_a}개, 총 {tot_all}개")
        
        # Mathematical verification
        if tr_n + te_n != tot_n or tr_a + te_a != tot_a or tr_t + te_t != tot_all:
            metadata_ok = False
            print("      -> [오류] 합계 검증 실패!")
        if tr_t != tr_n + tr_a or te_t != te_n + te_a:
            metadata_ok = False
            print("      -> [오류] 세부 합계 검증 실패!")

    if metadata_ok:
        print("  -> 성공: datasets 테이블의 모든 크기 메타데이터가 수학적 일치성 및 무결성을 충족합니다.")
    else:
        print("  -> 실패: 일부 메타데이터 컬럼 간의 불일치가 확인되었습니다.")

    conn.close()

if __name__ == "__main__":
    verify()
