import os
import sys
import time
import subprocess
import psutil

# Configurations
DATA_DIR = "/Users/minho/Documents/Dataset"
PYTHON_BIN = os.path.join(DATA_DIR, ".venv/bin/python")

# Current target process to wait for
MONITOR_SCRIPT_NAME = "run_all_length_gated_periodicity_evaluations.py"

# Future sequential experiments to run in order
SEQUENTIAL_SCRIPTS = [
    "run_all_adaptive_cnn_dynamic_quantile.py", # [실험 4] 분위수 비례 POT 스케일링
    "run_all_adaptive_cnn_kl_annealing.py",     # [실험 3] KL Weight Annealing
    "run_all_adaptive_cnn_contrastive.py",      # [실험 1] 정상 잠재 공간 대조
    "run_all_adaptive_transformer_vae.py"       # [실험 2] Transformer 어텐션 VAE
]

LOG_FILE_PATH = os.path.join(DATA_DIR, "sequential_experiments_pipeline.log")

def log_message(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg)
    with open(LOG_FILE_PATH, "a") as f:
        f.write(formatted_msg + "\n")

def is_process_running(script_name):
    """
    Check if a python script with script_name is running.
    """
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = proc.info['cmdline']
            if cmd and any(script_name in arg for arg in cmd):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False

def main():
    log_message("="*60)
    log_message("연속 실험 오케스트레이션 파이프라인 가동 개시!")
    log_message(f"로그 파일 경로: {LOG_FILE_PATH}")
    log_message(f"1단계: 선행 실험 '{MONITOR_SCRIPT_NAME}'의 종료를 대기합니다...")
    
    # Wait for the first script to finish
    wait_count = 0
    while is_process_running(MONITOR_SCRIPT_NAME):
        time.sleep(15)
        wait_count += 1
        if wait_count % 20 == 0: # Every 5 minutes
            log_message(f"  * 대기 중... 선행 실험 '{MONITOR_SCRIPT_NAME}'이 아직 실행 중입니다.")
            
    log_message(f"🌟 선행 실험 '{MONITOR_SCRIPT_NAME}' 종료가 감지되었습니다!")
    log_message("="*60)
    
    # Run subsequent experiments sequentially
    for idx, script in enumerate(SEQUENTIAL_SCRIPTS, 1):
        script_path = os.path.join(DATA_DIR, script)
        if not os.path.exists(script_path):
            log_message(f"❌ 에러: {script} 파일을 찾을 수 없습니다. 경로: {script_path}")
            continue
            
        log_message(f"▶️ [실험 {idx}/{len(SEQUENTIAL_SCRIPTS)}] '{script}' 실행 시작...")
        start_time = time.time()
        
        try:
            # Execute python script and stream logs to stdout/file
            process = subprocess.Popen(
                [PYTHON_BIN, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Print output in real-time
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                # Also log to file
                with open(LOG_FILE_PATH, "a") as f:
                    f.write(line)
                    
            process.wait()
            elapsed_time = time.time() - start_time
            
            if process.returncode == 0:
                log_message(f"✅ [실험 {idx}/{len(SEQUENTIAL_SCRIPTS)}] '{script}' 성공적으로 종료! (소요 시간: {elapsed_time/60:.2f}분)")
            else:
                log_message(f"⚠️ [실험 {idx}/{len(SEQUENTIAL_SCRIPTS)}] '{script}' 오류 종료! (Return Code: {process.returncode}, 소요 시간: {elapsed_time/60:.2f}분)")
                
        except Exception as e:
            log_message(f"❌ [실험 {idx}/{len(SEQUENTIAL_SCRIPTS)}] '{script}' 실행 중 중차대한 예외 발생: {e}")
            
        log_message("-"*60)
        
    log_message("🎉 모든 연속 차세대 혁신 실험 파이프라인이 정상 완수되었습니다!")
    log_message("="*60)

if __name__ == "__main__":
    main()
