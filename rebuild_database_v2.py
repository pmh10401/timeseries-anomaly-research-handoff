import os
import re
import sqlite3
import urllib.request
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download

# Paths
BASE_DIR = Path("/Users/minho/Documents/Dataset")
UNIVARIATE_DIR = BASE_DIR / "Univariate_ts"
DB_PATH = BASE_DIR / "univariate_ts.db"
TEMP_DIR = BASE_DIR / "temp_downloads"

def parse_ts_file(filepath):
    """
    Parses a single UCR/aeon .ts file.
    Returns (metadata_dict, instances_list) where instances_list is a list of tuples: (numpy_float32_array, label_str)
    """
    metadata = {}
    instances = []
    
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        in_data = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            
            if not in_data:
                if line.lower().startswith("@data"):
                    in_data = True
                    continue
                if line.startswith("@"):
                    match = re.match(r"@(\w+)\s+(.+)", line)
                    if match:
                        key = match.group(1)
                        val = match.group(2).strip()
                        if val.lower() == "true":
                            val = True
                        elif val.lower() == "false":
                            val = False
                        elif val.isdigit():
                            val = int(val)
                        metadata[key] = val
            else:
                if ":" in line:
                    parts = line.rsplit(":", 1)
                    values_str = parts[0]
                    label = parts[1]
                else:
                    parts = line.rsplit(",", 1)
                    values_str = parts[0]
                    label = parts[1]
                
                values = []
                for v in values_str.split(","):
                    v = v.strip()
                    if v == "?":
                        values.append(float("nan"))
                    else:
                        values.append(float(v))
                
                arr = np.array(values, dtype=np.float32)
                instances.append((arr, label))
                
    return metadata, instances

def setup_database():
    """
    Initializes a fresh database, removing the old one first.
    """
    if DB_PATH.exists():
        print(f"기존 데이터베이스 제거 중: {DB_PATH}")
        DB_PATH.unlink()
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create datasets table with count columns
    cursor.execute("""
    CREATE TABLE datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        univariate BOOLEAN NOT NULL,
        equal_length BOOLEAN NOT NULL,
        series_length INTEGER,
        has_missing BOOLEAN NOT NULL,
        train_normal_count INTEGER NOT NULL,
        train_anomaly_count INTEGER NOT NULL,
        train_total_count INTEGER NOT NULL,
        test_normal_count INTEGER NOT NULL,
        test_anomaly_count INTEGER NOT NULL,
        test_total_count INTEGER NOT NULL,
        total_normal_count INTEGER NOT NULL,
        total_anomaly_count INTEGER NOT NULL,
        total_count INTEGER NOT NULL
    );
    """)
    
    # Create instances table
    cursor.execute("""
    CREATE TABLE instances (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id INTEGER NOT NULL,
        split TEXT NOT NULL,
        instance_index INTEGER NOT NULL,
        label TEXT NOT NULL,
        values_blob BLOB NOT NULL,
        labels_blob BLOB,
        FOREIGN KEY (dataset_id) REFERENCES datasets(id)
    );
    """)
    
    # Create unique index
    cursor.execute("""
    CREATE UNIQUE INDEX idx_instances_dataset_split_index 
    ON instances(dataset_id, split, instance_index);
    """)
    
    conn.commit()
    return conn

def convert_classification_to_oneclass(conn):
    """
    Converts standard UCR classification datasets into One-Class Anomaly Detection datasets.
    """
    print("\n--- UCR 128 분류 데이터셋의 One-Class 변환 시작 ---")
    cursor = conn.cursor()
    
    subdirs = sorted([d for d in UNIVARIATE_DIR.iterdir() if d.is_dir()])
    total_folders = len(subdirs)
    
    np.random.seed(42) # For reproducibility
    converted_count = 0
    
    for folder_idx, d in enumerate(subdirs, 1):
        dataset_name = d.name
        ts_files = list(d.glob("*.ts"))
        if not ts_files:
            continue
            
        # Parse train & test files and combine them
        train_file = d / f"{dataset_name}_TRAIN.ts"
        test_file = d / f"{dataset_name}_TEST.ts"
        
        # fallback finding
        if not train_file.exists():
            train_candidates = [f for f in ts_files if "train" in f.name.lower()]
            if train_candidates: train_file = train_candidates[0]
        if not test_file.exists():
            test_candidates = [f for f in ts_files if "test" in f.name.lower()]
            if test_candidates: test_file = test_candidates[0]
            
        all_instances = []
        metadata = {}
        
        for fpath in [train_file, test_file]:
            if fpath.exists():
                meta, insts = parse_ts_file(fpath)
                all_instances.extend(insts)
                metadata.update(meta)
                
        if not all_instances:
            continue
            
        # Extract unique class labels
        labels = [inst[1] for inst in all_instances]
        unique_labels = sorted(list(set(labels)))
        
        # Skip datasets with less than 2 classes
        if len(unique_labels) < 2:
            continue
            
        for normal_label in unique_labels:
            sub_dataset_name = f"{dataset_name}_normal_{normal_label}"
            
            # Separate normal and abnormal instances
            normal_insts = [inst[0] for inst in all_instances if inst[1] == normal_label]
            anomaly_insts = [inst[0] for inst in all_instances if inst[1] != normal_label]
            
            n_normal = len(normal_insts)
            n_anomaly = len(anomaly_insts)
            
            if n_normal == 0 or n_anomaly == 0:
                continue
                
            # Shuffle normal instances
            normal_indices = np.arange(n_normal)
            np.random.shuffle(normal_indices)
            
            # Split normal into 80% train, 20% test-unseen
            split_idx = int(n_normal * 0.8)
            train_indices = normal_indices[:split_idx]
            test_unseen_indices = normal_indices[split_idx:]
            
            n_train_normal = len(train_indices)
            n_train_anomaly = 0
            n_train_total = n_train_normal
            
            # 2% Anomaly Ratio Target: N_normal >= 49 * N_anomaly
            n_anomaly_to_use = max(1, n_normal // 49)
            if n_anomaly_to_use > n_anomaly:
                n_anomaly_to_use = n_anomaly
                
            # Randomly pick anomalies
            selected_anomaly_indices = np.random.choice(n_anomaly, n_anomaly_to_use, replace=False)
            
            # Normal test samples required: N_n = 49 * N_anomaly_to_use
            n_normal_required = 49 * n_anomaly_to_use
            
            selected_test_normal_insts = []
            unseen_indices_to_use = list(test_unseen_indices)
            if len(unseen_indices_to_use) >= n_normal_required:
                chosen_unseen = np.random.choice(unseen_indices_to_use, n_normal_required, replace=False)
                for c_idx in chosen_unseen:
                    selected_test_normal_insts.append(normal_insts[c_idx])
            else:
                for c_idx in unseen_indices_to_use:
                    selected_test_normal_insts.append(normal_insts[c_idx])
                n_borrow = n_normal_required - len(unseen_indices_to_use)
                borrowed_indices = np.random.choice(train_indices, n_borrow, replace=True)
                for c_idx in borrowed_indices:
                    selected_test_normal_insts.append(normal_insts[c_idx])
                    
            n_test_normal = len(selected_test_normal_insts)
            n_test_anomaly = len(selected_anomaly_indices)
            n_test_total = n_test_normal + n_test_anomaly
            
            tot_normal = n_train_normal + n_test_normal
            tot_anomaly = n_test_anomaly
            tot_all = n_train_total + n_test_total
            
            # Write metadata to DB
            series_len = metadata.get("seriesLength", len(normal_insts[0]))
            cursor.execute("""
            INSERT INTO datasets (
                name, univariate, equal_length, series_length, has_missing,
                train_normal_count, train_anomaly_count, train_total_count,
                test_normal_count, test_anomaly_count, test_total_count,
                total_normal_count, total_anomaly_count, total_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (sub_dataset_name, True, True, series_len, False,
                  n_train_normal, n_train_anomaly, n_train_total,
                  n_test_normal, n_test_anomaly, n_test_total,
                  tot_normal, tot_anomaly, tot_all))
            dataset_id = cursor.lastrowid
            
            # Insert TRAIN instances (normal only, labels are all 0)
            train_inserts = []
            for t_idx, c_idx in enumerate(train_indices):
                arr = normal_insts[c_idx]
                blob_val = arr.tobytes()
                blob_lbl = np.zeros(len(arr), dtype=np.uint8).tobytes()
                train_inserts.append((dataset_id, "TRAIN", t_idx, "0", blob_val, blob_lbl))
                
            cursor.executemany("""
            INSERT INTO instances (dataset_id, split, instance_index, label, values_blob, labels_blob)
            VALUES (?, ?, ?, ?, ?, ?)
            """, train_inserts)
            
            # Insert TEST instances
            test_inserts = []
            t_idx = 0
            for arr in selected_test_normal_insts:
                blob_val = arr.tobytes()
                blob_lbl = np.zeros(len(arr), dtype=np.uint8).tobytes()
                test_inserts.append((dataset_id, "TEST", t_idx, "0", blob_val, blob_lbl))
                t_idx += 1
                
            for c_idx in selected_anomaly_indices:
                arr = anomaly_insts[c_idx]
                blob_val = arr.tobytes()
                blob_lbl = np.ones(len(arr), dtype=np.uint8).tobytes()
                test_inserts.append((dataset_id, "TEST", t_idx, "1", blob_val, blob_lbl))
                t_idx += 1
                
            cursor.executemany("""
            INSERT INTO instances (dataset_id, split, instance_index, label, values_blob, labels_blob)
            VALUES (?, ?, ?, ?, ?, ?)
            """, test_inserts)
            
            converted_count += 1
            
        if folder_idx % 20 == 0 or folder_idx == total_folders:
            print(f"UCR Classification 변환 진행률: {folder_idx}/{total_folders}")
            
    conn.commit()
    print(f"UCR Classification One-Class 변환 및 적재 완료 (총 {converted_count}개 서브셋 생성).")

def convert_cornell_whale(conn):
    print("\n--- Hugging Face CornellWhaleChallenge 변환 시작 ---")
    cursor = conn.cursor()
    
    # Download files
    print("고래 데이터셋 파일 다운로드 중...")
    x_path = hf_hub_download(repo_id="monster-monash/CornellWhaleChallenge", filename="CornellWhaleChallenge_X.npy", repo_type="dataset")
    y_path = hf_hub_download(repo_id="monster-monash/CornellWhaleChallenge", filename="CornellWhaleChallenge_y.npy", repo_type="dataset")
    
    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path)
    
    # Separate normal (0) and abnormal (1) indices
    normal_indices = np.where(y == 0)[0]
    anomaly_indices = np.where(y == 1)[0]
    
    n_normal = len(normal_indices)
    n_anomaly = len(anomaly_indices)
    
    print(f"정상 샘플 수: {n_normal}, 이상치 샘플 수: {n_anomaly}")
    
    np.random.seed(42)
    np.random.shuffle(normal_indices)
    
    # 80:20 split of normal
    split_idx = int(n_normal * 0.8)
    train_indices = normal_indices[:split_idx]
    test_unseen_indices = normal_indices[split_idx:]
    
    n_train_normal = len(train_indices)
    n_train_anomaly = 0
    n_train_total = n_train_normal
    
    # 2% Anomaly Ratio Target: N_normal >= 49 * N_anomaly
    n_anomaly_to_use = n_normal // 49
    if n_anomaly_to_use > n_anomaly:
        n_anomaly_to_use = n_anomaly
        
    selected_anomaly_indices = np.random.choice(anomaly_indices, n_anomaly_to_use, replace=False)
    
    n_normal_required = 49 * n_anomaly_to_use
    selected_test_normal_indices = []
    
    # Fill test normal
    unseen_list = list(test_unseen_indices)
    if len(unseen_list) >= n_normal_required:
        chosen = np.random.choice(unseen_list, n_normal_required, replace=False)
        selected_test_normal_indices.extend(chosen)
    else:
        selected_test_normal_indices.extend(unseen_list)
        n_borrow = n_normal_required - len(unseen_list)
        borrowed = np.random.choice(train_indices, n_borrow, replace=True)
        selected_test_normal_indices.extend(borrowed)
        
    n_test_normal = len(selected_test_normal_indices)
    n_test_anomaly = len(selected_anomaly_indices)
    n_test_total = n_test_normal + n_test_anomaly
    
    tot_normal = n_train_normal + n_test_normal
    tot_anomaly = n_test_anomaly
    tot_all = n_train_total + n_test_total
    
    # Write dataset metadata
    dataset_name = "CornellWhaleChallenge"
    series_len = X.shape[2] # length is 4000
    cursor.execute("""
    INSERT INTO datasets (
        name, univariate, equal_length, series_length, has_missing,
        train_normal_count, train_anomaly_count, train_total_count,
        test_normal_count, test_anomaly_count, test_total_count,
        total_normal_count, total_anomaly_count, total_count
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (dataset_name, True, True, series_len, False,
          n_train_normal, n_train_anomaly, n_train_total,
          n_test_normal, n_test_anomaly, n_test_total,
          tot_normal, tot_anomaly, tot_all))
    dataset_id = cursor.lastrowid
    
    # Insert TRAIN
    train_inserts = []
    for t_idx, idx in enumerate(train_indices):
        arr = X[idx][0].astype(np.float32)
        blob_val = arr.tobytes()
        blob_lbl = np.zeros(series_len, dtype=np.uint8).tobytes()
        train_inserts.append((dataset_id, "TRAIN", t_idx, "0", blob_val, blob_lbl))
        
    cursor.executemany("""
    INSERT INTO instances (dataset_id, split, instance_index, label, values_blob, labels_blob)
    VALUES (?, ?, ?, ?, ?, ?)
    """, train_inserts)
    
    # Insert TEST
    test_inserts = []
    t_idx = 0
    for idx in selected_test_normal_indices:
        arr = X[idx][0].astype(np.float32)
        blob_val = arr.tobytes()
        blob_lbl = np.zeros(series_len, dtype=np.uint8).tobytes()
        test_inserts.append((dataset_id, "TEST", t_idx, "0", blob_val, blob_lbl))
        t_idx += 1
        
    for idx in selected_anomaly_indices:
        arr = X[idx][0].astype(np.float32)
        blob_val = arr.tobytes()
        blob_lbl = np.ones(series_len, dtype=np.uint8).tobytes()
        test_inserts.append((dataset_id, "TEST", t_idx, "1", blob_val, blob_lbl))
        t_idx += 1
        
    cursor.executemany("""
    INSERT INTO instances (dataset_id, split, instance_index, label, values_blob, labels_blob)
    VALUES (?, ?, ?, ?, ?, ?)
    """, test_inserts)
    
    conn.commit()
    print(f"CornellWhaleChallenge 적재 완료 (TRAIN: {len(train_indices)}개, TEST: {t_idx}개).")

def clean_temp():
    import shutil
    if TEMP_DIR.exists():
        print("\n임시 다운로드 파일 정리 중...")
        shutil.rmtree(TEMP_DIR)
        print("정리 완료.")

def main():
    conn = setup_database()
    
    try:
        # 1. Convert standard UCR classification datasets
        convert_classification_to_oneclass(conn)
        
        # 2. CornellWhaleChallenge
        convert_cornell_whale(conn)
        
        # 3. Clean temp
        clean_temp()
        
        db_size_mb = DB_PATH.stat().st_size / (1024 * 1024)
        print("\n" + "="*50)
        print("비지도 이상치 탐지 전용 데이터베이스 구축 완료!")
        print(f"- DB 경로: {DB_PATH}")
        print(f"- 최종 파일 크기: {db_size_mb:.2f} MB")
        print("="*50)
        
    except Exception as e:
        print(f"데이터베이스 재구축 중 오류 발생: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
