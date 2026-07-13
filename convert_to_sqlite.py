import os
import re
import sqlite3
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path("/Users/minho/Documents/Dataset")
UNIVARIATE_DIR = BASE_DIR / "Univariate_ts"
DB_PATH = BASE_DIR / "univariate_ts.db"

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
            
            # Skip comments
            if line.startswith("#"):
                continue
            
            if not in_data:
                # Check for data tag
                if line.lower().startswith("@data"):
                    in_data = True
                    continue
                
                # Parse tags
                if line.startswith("@"):
                    match = re.match(r"@(\w+)\s+(.+)", line)
                    if match:
                        key = match.group(1)
                        val = match.group(2).strip()
                        
                        # Convert types where appropriate
                        if val.lower() == "true":
                            val = True
                        elif val.lower() == "false":
                            val = False
                        elif val.isdigit():
                            val = int(val)
                        
                        metadata[key] = val
            else:
                # Parsing data lines: "val1,val2,...,valN:label" or "val1,val2,...,valN,label"
                if ":" in line:
                    parts = line.rsplit(":", 1)
                    values_str = parts[0]
                    label = parts[1]
                else:
                    parts = line.rsplit(",", 1)
                    values_str = parts[0]
                    label = parts[1]
                
                # Parse values (handle missing values '?' as nan)
                values = []
                for v in values_str.split(","):
                    v = v.strip()
                    if v == "?":
                        values.append(float("nan"))
                    else:
                        values.append(float(v))
                
                # Convert to float32 numpy array
                arr = np.array(values, dtype=np.float32)
                instances.append((arr, label))
                
    return metadata, instances

def setup_database(conn):
    """
    Creates the tables and indexes according to Option A.
    """
    cursor = conn.cursor()
    
    # Create datasets table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        univariate BOOLEAN NOT NULL,
        equal_length BOOLEAN NOT NULL,
        series_length INTEGER,
        has_missing BOOLEAN NOT NULL
    );
    """)
    
    # Create instances table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS instances (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id INTEGER NOT NULL,
        split TEXT NOT NULL,
        instance_index INTEGER NOT NULL,
        label TEXT NOT NULL,
        values_blob BLOB NOT NULL,
        FOREIGN KEY (dataset_id) REFERENCES datasets(id)
    );
    """)
    
    # Create unique index to prevent duplicate inserts and speed up lookups
    cursor.execute("""
    CREATE UNIQUE INDEX IF NOT EXISTS idx_instances_dataset_split_index 
    ON instances(dataset_id, split, instance_index);
    """)
    
    conn.commit()

def main():
    print("SQLite 통합 데이터베이스 생성 시작...")
    
    # Connect to SQLite
    conn = sqlite3.connect(DB_PATH)
    setup_database(conn)
    
    # Find all subdirectories
    subdirs = sorted([d for d in UNIVARIATE_DIR.iterdir() if d.is_dir()])
    total_datasets = len(subdirs)
    
    print(f"발견된 데이터셋 폴더 수: {total_datasets}")
    
    cursor = conn.cursor()
    
    converted_datasets_count = 0
    total_instances_count = 0
    
    for i, d in enumerate(subdirs, 1):
        dataset_name = d.name
        # Find .ts files inside the directory
        ts_files = list(d.glob("*.ts"))
        if not ts_files:
            continue
            
        print(f"[{i}/{total_datasets}] {dataset_name} 데이터셋 처리 중...")
        
        # We will parse metadata from any train file, or test file if train is missing
        train_file = d / f"{dataset_name}_TRAIN.ts"
        test_file = d / f"{dataset_name}_TEST.ts"
        
        # If the filenames are slightly different (e.g. lowercase), find them
        if not train_file.exists():
            train_candidates = [f for f in ts_files if "train" in f.name.lower()]
            if train_candidates:
                train_file = train_candidates[0]
        if not test_file.exists():
            test_candidates = [f for f in ts_files if "test" in f.name.lower()]
            if test_candidates:
                test_file = test_candidates[0]
        
        # Parse train/test
        for file_path, split_name in [(train_file, "TRAIN"), (test_file, "TEST")]:
            if not file_path.exists():
                continue
                
            try:
                metadata, instances = parse_ts_file(file_path)
                
                # Default values from directory if not present in metadata
                problem_name = metadata.get("problemName", dataset_name)
                univariate = metadata.get("univariate", True)
                equal_length = metadata.get("equalLength", True)
                series_length = metadata.get("seriesLength", None)
                missing = metadata.get("missing", False)
                
                # Check if dataset already in DB
                cursor.execute("SELECT id FROM datasets WHERE name = ?", (problem_name,))
                row = cursor.fetchone()
                if row:
                    dataset_id = row[0]
                else:
                    cursor.execute("""
                    INSERT INTO datasets (name, univariate, equal_length, series_length, has_missing)
                    VALUES (?, ?, ?, ?, ?)
                    """, (problem_name, univariate, equal_length, series_length, missing))
                    dataset_id = cursor.lastrowid
                
                # Insert instances
                insert_data = []
                for idx, (arr, label) in enumerate(instances):
                    blob_data = arr.tobytes()
                    insert_data.append((dataset_id, split_name, idx, label, blob_data))
                
                # Execute bulk insert
                cursor.executemany("""
                INSERT OR REPLACE INTO instances (dataset_id, split, instance_index, label, values_blob)
                VALUES (?, ?, ?, ?, ?)
                """, insert_data)
                
                total_instances_count += len(instances)
                
            except Exception as e:
                print(f"오류 발생 ({file_path.name}): {e}")
        
        conn.commit()
        converted_datasets_count += 1
        
    conn.close()
    
    db_size_mb = DB_PATH.stat().st_size / (1024 * 1024)
    print("\n" + "="*50)
    print("통합 및 변환 작업이 완료되었습니다!")
    print(f"- 생성된 DB 경로: {DB_PATH}")
    print(f"- 성공적으로 변환된 데이터셋: {converted_datasets_count}개")
    print(f"- 총 저장된 인스턴스(TRAIN + TEST): {total_instances_count}개")
    print(f"- 데이터베이스 파일 크기: {db_size_mb:.2f} MB")
    print("="*50)

if __name__ == "__main__":
    main()
