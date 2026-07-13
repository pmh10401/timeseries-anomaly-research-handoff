import os
import re
import json
import zipfile
import sqlite3
import urllib.request
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path("/Users/minho/Documents/Dataset")
DB_PATH = BASE_DIR / "univariate_ts.db"
TEMP_DIR = BASE_DIR / "temp_downloads"

# URLs
UCR_ANOMALY_URL = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip"
NAB_URL = "https://github.com/numenta/NAB/archive/refs/heads/master.zip"

def download_and_extract(url, name):
    os.makedirs(TEMP_DIR, exist_ok=True)
    zip_path = TEMP_DIR / f"{name}.zip"
    extract_path = TEMP_DIR / name
    
    if not zip_path.exists():
        print(f"[{name}] 다운로드 중: {url}")
        # Show progress roughly
        def reporthook(blocknum, blocksize, totalsize):
            readdata = blocknum * blocksize
            if totalsize > 0:
                percent = readdata * 100 / totalsize
                print(f"\r다운로드 중... {percent:.1f}%", end="")
            else:
                print(f"\r다운로드 중... {readdata} bytes", end="")
        
        urllib.request.urlretrieve(url, zip_path, reporthook)
        print("\n다운로드 완료.")
    
    if not extract_path.exists():
        print(f"[{name}] 압축 해제 중...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("압축 해제 완료.")
        
    return extract_path

def setup_labels_column(conn):
    """
    Option A 스키마에 개별 타임스텝별 이상치 라벨(0/1)을 저장할 labels_blob 컬럼을 추가합니다.
    """
    cursor = conn.cursor()
    try:
        cursor.execute("ALTER TABLE instances ADD COLUMN labels_blob BLOB;")
        conn.commit()
        print("데이터베이스 스키마 확장 완료 (labels_blob 컬럼 추가).")
    except sqlite3.OperationalError:
        # Column already exists
        pass

def parse_and_insert_ucr_anomaly(conn, extracted_dir):
    print("\n--- UCR Anomaly Archive 파싱 및 저장 시작 ---")
    cursor = conn.cursor()
    
    # Locate the files
    candidate_dirs = list(extracted_dir.glob("**/UCR_TimeSeriesAnomalyDatasets2021"))
    if not candidate_dirs:
        candidate_dirs = [extracted_dir]
    
    files_dir = None
    for d in candidate_dirs:
        files_candidates = list(d.glob("**/Files")) + list(d.glob("**/UCR_Anomaly_FullData"))
        if files_candidates:
            files_dir = files_candidates[0]
            break
    if not files_dir:
        files_dir = extracted_dir
        
    txt_files = list(files_dir.glob("**/*_UCR_Anomaly_*.txt"))
    if not txt_files:
        txt_files = [f for f in files_dir.glob("**/*.txt") if "UCR_Anomaly" in f.name]
        
    print(f"발견된 UCR Anomaly 파일 수: {len(txt_files)}")
    
    # Regex to parse file name metadata
    pattern = re.compile(r"(\d+)_UCR_Anomaly_([\w\d\-]+)_(\d+)_(\d+)_(\d+)\.txt")
    
    inserted_count = 0
    for idx, fpath in enumerate(sorted(txt_files), 1):
        match = pattern.match(fpath.name)
        if not match:
            continue
            
        file_id = match.group(1)
        dataset_name = f"UCR_Anomaly_{file_id}_{match.group(2)}"
        train_len = int(match.group(3))
        anomaly_start = int(match.group(4))
        anomaly_end = int(match.group(5))
        
        # Read the values (handle space/tab/newline separation robustly)
        values = []
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                for token in line.split():
                    token = token.strip()
                    if token:
                        values.append(float(token))
            
        series_len = len(values)
        if series_len == 0:
            continue
            
        # Create dataset entry
        cursor.execute("SELECT id FROM datasets WHERE name = ?", (dataset_name,))
        row = cursor.fetchone()
        if row:
            dataset_id = row[0]
        else:
            cursor.execute("""
            INSERT INTO datasets (name, univariate, equal_length, series_length, has_missing)
            VALUES (?, ?, ?, ?, ?)
            """, (dataset_name, True, True, series_len, False))
            dataset_id = cursor.lastrowid
            
        # Prepare arrays
        arr_values = np.array(values, dtype=np.float32)
        
        # Point-wise labels (0: normal, 1: anomaly)
        arr_labels = np.zeros(series_len, dtype=np.uint8)
        arr_labels[anomaly_start : anomaly_end + 1] = 1
        
        # TRAIN Insert
        train_val_blob = arr_values[:train_len].tobytes()
        train_lbl_blob = arr_labels[:train_len].tobytes()
        cursor.execute("""
        INSERT OR REPLACE INTO instances (dataset_id, split, instance_index, label, values_blob, labels_blob)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (dataset_id, "TRAIN", 0, "anomaly_detection", train_val_blob, train_lbl_blob))
        
        # TEST Insert
        test_val_blob = arr_values[train_len:].tobytes()
        test_lbl_blob = arr_labels[train_len:].tobytes()
        cursor.execute("""
        INSERT OR REPLACE INTO instances (dataset_id, split, instance_index, label, values_blob, labels_blob)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (dataset_id, "TEST", 0, "anomaly_detection", test_val_blob, test_lbl_blob))
        
        inserted_count += 1
        if idx % 50 == 0 or idx == len(txt_files):
            print(f"UCR Anomaly 처리 진행률: {idx}/{len(txt_files)}")
            
    conn.commit()
    print(f"UCR Anomaly 통합 완료 (총 {inserted_count}개 파일 등록).")

def parse_and_insert_nab(conn, extracted_dir):
    print("\n--- NAB (Numenta Anomaly Benchmark) 파싱 및 저장 시작 ---")
    cursor = conn.cursor()
    
    # Locate NAB root directory inside zip
    nab_root = list(extracted_dir.glob("**/NAB-master"))
    if not nab_root:
        nab_root = [extracted_dir]
    root_path = nab_root[0]
    
    # Load labels
    labels_file = root_path / "labels" / "combined_windows.json"
    if not labels_file.exists():
        print("오류: NAB 라벨 파일(combined_windows.json)을 찾을 수 없습니다.")
        return
        
    with open(labels_file, "r") as f:
        labels_dict = json.load(f)
        
    inserted_count = 0
    # Process each file labeled in json
    for rel_path, windows in labels_dict.items():
        csv_path = root_path / "data" / rel_path
        if not csv_path.exists():
            continue
            
        dataset_name = f"NAB_{Path(rel_path).stem}"
        
        # Parse CSV
        timestamps = []
        values = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            header = f.readline() # skip header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    timestamps.append(parts[0])
                    values.append(float(parts[1]))
                    
        series_len = len(values)
        if series_len == 0:
            continue
            
        arr_labels = np.zeros(series_len, dtype=np.uint8)
        
        for window in windows:
            start_str, end_str = window[0], window[1]
            for t_idx, t_str in enumerate(timestamps):
                if start_str <= t_str <= end_str:
                    arr_labels[t_idx] = 1
                    
        arr_values = np.array(values, dtype=np.float32)
        
        # Insert dataset
        cursor.execute("SELECT id FROM datasets WHERE name = ?", (dataset_name,))
        row = cursor.fetchone()
        if row:
            dataset_id = row[0]
        else:
            cursor.execute("""
            INSERT INTO datasets (name, univariate, equal_length, series_length, has_missing)
            VALUES (?, ?, ?, ?, ?)
            """, (dataset_name, True, True, series_len, False))
            dataset_id = cursor.lastrowid
            
        val_blob = arr_values.tobytes()
        lbl_blob = arr_labels.tobytes()
        cursor.execute("""
        INSERT OR REPLACE INTO instances (dataset_id, split, instance_index, label, values_blob, labels_blob)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (dataset_id, "TEST", 0, "anomaly_detection", val_blob, lbl_blob))
        
        inserted_count += 1
        
    conn.commit()
    print(f"NAB 데이터셋 통합 완료 (총 {inserted_count}개 파일 등록).")

def clean_temp():
    import shutil
    if TEMP_DIR.exists():
        print("\n임시 다운로드 파일 정리 중...")
        shutil.rmtree(TEMP_DIR)
        print("정리 완료.")

def main():
    conn = sqlite3.connect(DB_PATH)
    setup_labels_column(conn)
    
    try:
        # 1. UCR Anomaly Archive
        ucr_dir = download_and_extract(UCR_ANOMALY_URL, "UCR_Anomaly")
        parse_and_insert_ucr_anomaly(conn, ucr_dir)
        
        # 2. NAB
        nab_dir = download_and_extract(NAB_URL, "NAB")
        parse_and_insert_nab(conn, nab_dir)
        
        # 3. Clean up
        clean_temp()
        
        # Output final sizes
        db_size_mb = DB_PATH.stat().st_size / (1024 * 1024)
        print("\n" + "="*50)
        print("이상치 탐지 데이터셋 추가 완료!")
        print(f"- DB 경로: {DB_PATH}")
        print(f"- 최종 파일 크기: {db_size_mb:.2f} MB")
        print("="*50)
        
    except Exception as e:
        print(f"변환 작업 중 오류 발생: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
