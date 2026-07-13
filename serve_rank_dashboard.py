#!/usr/bin/env python3
import base64
import csv
import hmac
import json
import os
import re
import sqlite3
import subprocess
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np


ROOT = Path("/Users/minho/Documents/timesries project")
DATA_DIR = Path("/Users/minho/Documents/Dataset")
DB_PATH = DATA_DIR / "univariate_ts.db"
DETAIL_PATH = DATA_DIR / "rank_ensemble_threshold_calibration.csv"
SUMMARY_PATH = DATA_DIR / "rank_ensemble_threshold_calibration_summary.csv"
LOG_PATH = DATA_DIR / "rank_ensemble_threshold_calibration_stdout.log"
STATE_PATH = DATA_DIR / "rank_experiments_sequential_state.json"
PID_PATH = DATA_DIR / "rank_dashboard_server.pid"
HEARTBEAT_PATH = DATA_DIR / "rank_experiments_dashboard_heartbeat.json"
EXTERNAL_LIVE_RUN_PATH = DATA_DIR / "rank_dashboard_external_live_run.json"
EXPECTED_DATASETS = 1117
AUTH_REALM = "Rank Experiment Monitor"
RESOURCE_CACHE = {"gpu": {"expires_at": 0, "value": None}}
CSV_CACHE = {}
SERVER_STARTED_UNIX = time.time()
EVALUATION_EXCLUSIONS = {"CornellWhaleChallenge", "Wafer_normal_1"}
VALIDATION_PLAN = [
    {"id": "experiment_143_test_length_budget_binding_audit", "label": "Exp143 · TEST-length budget binding audit", "status": "planned", "runner_id": "experiment_144_train_threshold_only_no_budget"},
    {"id": "experiment_144_train_threshold_only_no_budget", "label": "Exp144 · TRAIN-threshold-only no budget", "status": "planned", "runner_id": "experiment_144_train_threshold_only_no_budget"},
    {"id": "experiment_145_train_normal_count_budget", "label": "Exp145 · TRAIN-normal conformal budget", "status": "blocked", "blocked_reason": "TRAIN instance와 실제 wafer run의 대응이 DB에서 검증되지 않음"},
    {"id": "experiment_146_fixed_k_workload_sensitivity", "label": "Exp146 · Fixed-K sensitivity", "status": "planned", "runner_id": "experiment_144_train_threshold_only_no_budget"},
    {"id": "experiment_147_policy_level_train_only_no_budget", "label": "Exp147 · B2 + threshold-only", "status": "planned", "runner_id": "experiment_144_train_threshold_only_no_budget"},
    {"id": "experiment_148_policy_level_train_only_train_budget", "label": "Exp148 · B2 + TRAIN-normal budget", "status": "blocked", "blocked_reason": "Exp145 run-grain blocker에 의존"},
]

METRIC_GLOSSARY = [
    {
        "id": "progress",
        "label": "Progress",
        "group": "진행 지표",
        "description": "전체 데이터셋 중 현재 실험이 처리한 비율입니다.",
        "direction": "높을수록 실험이 많이 진행된 상태입니다.",
        "example": "예: 25%이면 1,117개 중 약 279개 데이터셋 처리가 끝난 상태입니다.",
    },
    {
        "id": "current_dataset",
        "label": "Current Dataset",
        "group": "진행 지표",
        "description": "최근 처리했거나 처리 중인 데이터셋 이름입니다.",
        "direction": "어느 데이터셋에서 시간이 걸리는지 확인할 때 봅니다.",
        "example": "예: CricketZ_normal_9가 보이면 CricketZ 계열 9번 정상 반복 평가를 지나고 있다는 뜻입니다.",
    },
    {
        "id": "runtime",
        "label": "Runtime / ETA",
        "group": "진행 지표",
        "description": "실험이 실행된 시간과 예상 남은 시간입니다.",
        "direction": "ETA는 최근 처리 속도에 따라 계속 바뀔 수 있습니다.",
        "example": "예: Runtime 20m, ETA 40m이면 지금 속도 기준 약 40분 뒤 완료 예상입니다.",
    },
    {
        "id": "recent_speed",
        "label": "Recent/min",
        "group": "진행 지표",
        "description": "최근 progress 로그 사이에서 1분에 처리한 데이터셋 수입니다.",
        "direction": "높을수록 최근 처리 속도가 빠릅니다.",
        "example": "예: 12.5이면 최근 구간에서 분당 약 12.5개 데이터셋을 처리했다는 뜻입니다.",
    },
    {
        "id": "operational_pick",
        "label": "Operational Pick",
        "group": "운영 지표",
        "description": "F1만 보지 않고 오탐, 알람 신뢰도, 정상 초과율을 함께 본 운영 후보입니다.",
        "direction": "운영 기본 후보가 뜨면 실제 서비스 기본값으로 검토할 만합니다.",
        "example": "예: F1이 조금 낮아도 FP와 Train Exceed가 낮으면 운영 후보가 될 수 있습니다.",
    },
    {
        "id": "mean_fp",
        "label": "Mean FP",
        "group": "운영 지표",
        "description": "정상인데 이상치로 잘못 알린 평균 개수입니다.",
        "direction": "낮을수록 사용자가 불필요한 알람을 덜 받습니다.",
        "example": "예: Mean FP 1.25이면 데이터셋 하나당 평균 1.25개의 정상 샘플을 이상치로 잘못 잡은 것입니다.",
    },
    {
        "id": "alert_precision",
        "label": "Alert Precision",
        "group": "운영 지표",
        "description": "모델이 이상치라고 알린 것 중 실제 이상치였던 비율입니다.",
        "direction": "높을수록 알람을 믿기 쉽습니다.",
        "example": "예: 60%이면 알람 10개 중 약 6개는 실제 이상치, 4개는 오탐입니다.",
    },
    {
        "id": "train_exceed",
        "label": "Train Exceed",
        "group": "운영 지표",
        "description": "정상 학습 데이터가 threshold를 넘은 비율입니다.",
        "direction": "낮을수록 정상 상태에서 오탐이 적을 가능성이 큽니다.",
        "example": "예: 0.80%이면 정상 학습 샘플 1,000개 중 약 8개가 threshold를 넘었다는 뜻입니다.",
    },
    {
        "id": "pred",
        "label": "Pred",
        "group": "판정 지표",
        "description": "모델이 이상치라고 판정한 전체 개수입니다. TP와 FP를 모두 포함합니다.",
        "direction": "너무 높으면 알람이 많고, 너무 낮으면 놓치는 이상치가 늘 수 있습니다.",
        "example": "예: pred 5, FP 3이면 총 5개를 알렸고 그중 3개는 오탐입니다.",
    },
    {
        "id": "tp",
        "label": "TP",
        "group": "판정 지표",
        "description": "실제 이상치를 이상치로 맞게 잡은 개수입니다.",
        "direction": "높을수록 실제 이상을 잘 잡습니다.",
        "example": "예: TP 2이면 실제 이상치 2개를 제대로 찾은 것입니다.",
    },
    {
        "id": "fp",
        "label": "FP",
        "group": "판정 지표",
        "description": "정상인데 이상치로 잘못 잡은 개수입니다.",
        "direction": "낮을수록 불필요한 알람이 줄어듭니다.",
        "example": "예: FP 3이면 정상 샘플 3개를 이상치로 잘못 알린 것입니다.",
    },
    {
        "id": "fn",
        "label": "FN",
        "group": "판정 지표",
        "description": "실제 이상치인데 모델이 놓친 개수입니다.",
        "direction": "낮을수록 놓친 이상치가 적습니다.",
        "example": "예: FN 1이면 실제 이상치 1개를 정상으로 지나친 것입니다.",
    },
    {
        "id": "f1",
        "label": "Mean / Median F1",
        "group": "성능 지표",
        "description": "이상치를 잘 잡는 능력과 오탐을 줄이는 능력을 함께 보는 점수입니다.",
        "direction": "높을수록 좋지만, 운영에서는 FP와 Train Exceed도 함께 봐야 합니다.",
        "example": "예: Mean F1 0.55는 전체 평균 성능, Median F1 0.67은 중간 데이터셋의 성능입니다.",
    },
    {
        "id": "zero_f1",
        "label": "Zero-F1",
        "group": "성능 지표",
        "description": "F1 점수가 0이 된 실패 케이스 수입니다.",
        "direction": "낮을수록 완전히 실패한 데이터셋이 적습니다.",
        "example": "예: zero 30이면 30개 데이터셋에서 이상치를 못 잡았거나 오탐만 냈다는 뜻입니다.",
    },
    {
        "id": "oracle_f1",
        "label": "Oracle F1",
        "group": "성능 지표",
        "description": "같은 score에서 threshold를 가장 잘 골랐을 때 가능한 상한에 가까운 점수입니다.",
        "direction": "현재 threshold 문제가 큰지, score 자체가 약한지 구분할 때 봅니다.",
        "example": "예: Oracle F1은 높은데 실제 F1이 낮으면 threshold 정책 개선 여지가 큽니다.",
    },
    {
        "id": "auc_pr",
        "label": "PR / AUC-PR",
        "group": "성능 지표",
        "description": "이상치가 적은 상황에서 score가 이상치를 얼마나 앞쪽에 잘 올리는지 보는 지표입니다.",
        "direction": "높을수록 threshold를 잘 잡았을 때 좋은 성능을 낼 가능성이 큽니다.",
        "example": "예: AUC-PR 0.80이면 이상치 후보 순위 품질이 꽤 좋다는 신호입니다.",
    },
    {
        "id": "coverage",
        "label": "Coverage",
        "group": "품질 지표",
        "description": "예상 데이터셋 중 결과가 생성된 비율입니다.",
        "direction": "높을수록 평가가 더 많이 완료된 상태입니다.",
        "example": "예: Coverage 90%이면 1,117개 중 약 1,005개 데이터셋 결과가 쌓인 상태입니다.",
    },
    {
        "id": "family_trouble",
        "label": "Family Trouble Spots",
        "group": "품질 지표",
        "description": "특정 데이터셋 계열에서 반복적으로 오탐이나 실패가 많은지 보여줍니다.",
        "direction": "상위에 뜨는 family는 모델 개선이나 별도 guard가 필요한 후보입니다.",
        "example": "예: Crop FP 180이면 Crop 계열에서 정상 샘플을 이상치로 많이 잘못 잡고 있다는 뜻입니다.",
    },
]

EXPERIMENTS = {
    "rank_v1_train_evt": {
        "label": "Experiment 24 · Train EVT rank ensemble",
        "detail_csv": DATA_DIR / "vae_results_rank_ensemble_calibration_train_evt.csv",
        "summary_csv": DATA_DIR / "vae_results_rank_ensemble_calibration_train_evt_summary.csv",
        "stdout_log": DATA_DIR / "rank_ensemble_train_evt_stdout.log",
    },
    "rank_threshold_calibration": {
        "label": "Experiment 25 · Rank threshold calibration",
        "detail_csv": DATA_DIR / "rank_ensemble_threshold_calibration.csv",
        "summary_csv": DATA_DIR / "rank_ensemble_threshold_calibration_summary.csv",
        "stdout_log": DATA_DIR / "rank_ensemble_threshold_calibration_stdout.log",
    },
    "experiment_26_rocket": {
        "label": "Experiment 26 · ROCKET random convolution anomaly score",
        "detail_csv": DATA_DIR / "experiment_26_rocket_results.csv",
        "summary_csv": DATA_DIR / "experiment_26_rocket_summary.csv",
        "stdout_log": DATA_DIR / "experiment_26_rocket_stdout.log",
    },
    "experiment_27_rocket_score_variants": {
        "label": "Experiment 27 · ROCKET score variants bundled",
        "detail_csv": DATA_DIR / "experiment_27_rocket_score_variants_results.csv",
        "summary_csv": DATA_DIR / "experiment_27_rocket_score_variants_summary.csv",
        "stdout_log": DATA_DIR / "experiment_27_rocket_score_variants_stdout.log",
    },
    "experiment_27e_rocket_1024_top32": {
        "label": "Experiment 27-E · ROCKET 1024 kernels top-32 score",
        "detail_csv": DATA_DIR / "experiment_27e_rocket_1024_top32_results.csv",
        "summary_csv": DATA_DIR / "experiment_27e_rocket_1024_top32_summary.csv",
        "stdout_log": DATA_DIR / "experiment_27e_rocket_1024_top32_stdout.log",
    },
    "experiment_27f_rstsf_interval": {
        "label": "Experiment 27-F · r-STSF random interval anomaly score",
        "detail_csv": DATA_DIR / "experiment_27f_rstsf_interval_results.csv",
        "summary_csv": DATA_DIR / "experiment_27f_rstsf_interval_summary.csv",
        "stdout_log": DATA_DIR / "experiment_27f_rstsf_interval_stdout.log",
    },
    "experiment_28_minirocket_multirocket_features": {
        "label": "Experiment 28 · MiniROCKET/MultiROCKET-style feature expansion",
        "detail_csv": DATA_DIR / "experiment_28_minirocket_multirocket_features_results.csv",
        "summary_csv": DATA_DIR / "experiment_28_minirocket_multirocket_features_summary.csv",
        "stdout_log": DATA_DIR / "experiment_28_minirocket_multirocket_features_stdout.log",
    },
    "experiment_29_train_normal_threshold_calibration": {
        "label": "Experiment 29 · Train-normal threshold calibration",
        "detail_csv": DATA_DIR / "experiment_29_train_normal_threshold_calibration_results.csv",
        "summary_csv": DATA_DIR / "experiment_29_train_normal_threshold_calibration_summary.csv",
        "stdout_log": DATA_DIR / "experiment_29_train_normal_threshold_calibration_stdout.log",
    },
    "experiment_30_knn_threshold_sweep": {
        "label": "Experiment 30 · KNN operational threshold sweep",
        "detail_csv": DATA_DIR / "experiment_30_knn_threshold_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_30_knn_threshold_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_30_knn_threshold_sweep_stdout.log",
    },
    "experiment_31_knn_operational_budget_sweep": {
        "label": "Experiment 31 · KNN operational budget sweep",
        "detail_csv": DATA_DIR / "experiment_31_knn_operational_budget_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_31_knn_operational_budget_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_31_knn_operational_budget_sweep_stdout.log",
    },
    "experiment_32_knn_score_capacity_sweep": {
        "label": "Experiment 32 · KNN score capacity sweep",
        "detail_csv": DATA_DIR / "experiment_32_knn_score_capacity_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_32_knn_score_capacity_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_32_knn_score_capacity_sweep_stdout.log",
    },
    "experiment_34_balanced_feature_capacity_sweep": {
        "label": "Experiment 34 · Clean-balanced ROCKET capacity sweep",
        "detail_csv": DATA_DIR / "experiment_34_balanced_feature_capacity_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_34_balanced_feature_capacity_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_34_balanced_feature_capacity_sweep_stdout.log",
        "expected_datasets": 173,
    },
    "experiment_35_balanced_threshold_policy_sweep": {
        "label": "Experiment 35 · Clean-balanced threshold policy diagnosis",
        "detail_csv": DATA_DIR / "experiment_35_balanced_threshold_policy_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_35_balanced_threshold_policy_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_35_balanced_threshold_policy_sweep_stdout.log",
        "expected_datasets": 173,
    },
    "experiment_36_balanced_score_normalization_sweep": {
        "label": "Experiment 36 · Clean-balanced density-normalized KNN scores",
        "detail_csv": DATA_DIR / "experiment_36_balanced_score_normalization_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_36_balanced_score_normalization_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_36_balanced_score_normalization_sweep_stdout.log",
        "expected_datasets": 173,
    },
    "experiment_37_balanced_bagged_rocket_ensemble": {
        "label": "Experiment 37 · Clean-balanced bagged ROCKET ensemble",
        "detail_csv": DATA_DIR / "experiment_37_balanced_bagged_rocket_ensemble_results.csv",
        "summary_csv": DATA_DIR / "experiment_37_balanced_bagged_rocket_ensemble_summary.csv",
        "stdout_log": DATA_DIR / "experiment_37_balanced_bagged_rocket_ensemble_stdout.log",
        "expected_datasets": 173,
    },
    "experiment_38_balanced_actual_length_handling": {
        "label": "Experiment 38 · Clean-balanced actual length handling",
        "detail_csv": DATA_DIR / "experiment_38_balanced_actual_length_handling_results.csv",
        "summary_csv": DATA_DIR / "experiment_38_balanced_actual_length_handling_summary.csv",
        "stdout_log": DATA_DIR / "experiment_38_balanced_actual_length_handling_stdout.log",
        "expected_datasets": 173,
    },
    "experiment_39_balanced_candidate_retest": {
        "label": "Experiment 39 · Clean-balanced candidate retest and family guard",
        "detail_csv": DATA_DIR / "experiment_39_balanced_candidate_retest_results.csv",
        "summary_csv": DATA_DIR / "experiment_39_balanced_candidate_retest_summary.csv",
        "stdout_log": DATA_DIR / "experiment_39_balanced_candidate_retest_stdout.log",
        "expected_datasets": 173,
    },
    "experiment_35_original_threshold_policy_sweep": {
        "label": "Experiment 35 · Original repeated-normal threshold policy",
        "detail_csv": DATA_DIR / "experiment_35_original_threshold_policy_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_35_original_threshold_policy_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_35_original_threshold_policy_sweep_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_36_original_score_normalization_sweep": {
        "label": "Experiment 36 · Original repeated-normal score normalization",
        "detail_csv": DATA_DIR / "experiment_36_original_score_normalization_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_36_original_score_normalization_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_36_original_score_normalization_sweep_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_37_original_bagged_rocket_ensemble": {
        "label": "Experiment 37 · Original repeated-normal bagged ROCKET",
        "detail_csv": DATA_DIR / "experiment_37_original_bagged_rocket_ensemble_results.csv",
        "summary_csv": DATA_DIR / "experiment_37_original_bagged_rocket_ensemble_summary.csv",
        "stdout_log": DATA_DIR / "experiment_37_original_bagged_rocket_ensemble_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_38_original_actual_length_handling": {
        "label": "Experiment 38 · Original repeated-normal length handling",
        "detail_csv": DATA_DIR / "experiment_38_original_actual_length_handling_results.csv",
        "summary_csv": DATA_DIR / "experiment_38_original_actual_length_handling_summary.csv",
        "stdout_log": DATA_DIR / "experiment_38_original_actual_length_handling_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_39_original_candidate_retest": {
        "label": "Experiment 39 · Original repeated-normal candidate retest",
        "detail_csv": DATA_DIR / "experiment_39_original_candidate_retest_results.csv",
        "summary_csv": DATA_DIR / "experiment_39_original_candidate_retest_summary.csv",
        "stdout_log": DATA_DIR / "experiment_39_original_candidate_retest_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_40_original_score_normalization_sweep": {
        "label": "Experiment 40 · Original repeated-normal score normalization",
        "detail_csv": DATA_DIR / "experiment_40_original_score_normalization_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_40_original_score_normalization_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_40_original_score_normalization_sweep_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_41_multi_aug_robust_baseline": {
        "label": "Experiment 41 · Multi-Aug robust baseline",
        "detail_csv": DATA_DIR / "experiment_41_multi_aug_robust_baseline_results.csv",
        "summary_csv": DATA_DIR / "experiment_41_multi_aug_robust_baseline_summary.csv",
        "stdout_log": DATA_DIR / "experiment_41_multi_aug_robust_baseline_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_42_multi_aug_robust_operational": {
        "label": "Experiment 42 · Multi-Aug robust operational thresholds",
        "detail_csv": DATA_DIR / "experiment_42_multi_aug_robust_operational_results.csv",
        "summary_csv": DATA_DIR / "experiment_42_multi_aug_robust_operational_summary.csv",
        "stdout_log": DATA_DIR / "experiment_42_multi_aug_robust_operational_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_43_explanation_space_transforms": {
        "label": "Experiment 43 · Explanation-space transforms",
        "detail_csv": DATA_DIR / "experiment_43_explanation_space_transforms_results.csv",
        "summary_csv": DATA_DIR / "experiment_43_explanation_space_transforms_summary.csv",
        "stdout_log": DATA_DIR / "experiment_43_explanation_space_transforms_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_44_classical_embedding_baselines": {
        "label": "Experiment 44 · Classical embedding baselines",
        "detail_csv": DATA_DIR / "experiment_44_classical_embedding_baselines_results.csv",
        "summary_csv": DATA_DIR / "experiment_44_classical_embedding_baselines_summary.csv",
        "stdout_log": DATA_DIR / "experiment_44_classical_embedding_baselines_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_45_model_hard_diagnostic_harness": {
        "label": "Experiment 45 · Model-hard diagnostic harness",
        "detail_csv": DATA_DIR / "experiment_45_model_hard_diagnostic_harness_results.csv",
        "summary_csv": DATA_DIR / "experiment_45_model_hard_diagnostic_harness_summary.csv",
        "stdout_log": DATA_DIR / "experiment_45_model_hard_diagnostic_harness_stdout.log",
        "expected_datasets": 20,
    },
    "experiment_46_model_hard_interval_drcif_lite": {
        "label": "Experiment 46 · Model-hard interval quantile DrCIF-lite",
        "detail_csv": DATA_DIR / "experiment_46_model_hard_interval_drcif_lite_results.csv",
        "summary_csv": DATA_DIR / "experiment_46_model_hard_interval_drcif_lite_summary.csv",
        "stdout_log": DATA_DIR / "experiment_46_model_hard_interval_drcif_lite_stdout.log",
        "expected_datasets": 13,
    },
    "experiment_47_model_hard_frequency_rocket": {
        "label": "Experiment 47 · Model-hard frequency decomposition ROCKET",
        "detail_csv": DATA_DIR / "experiment_47_model_hard_frequency_rocket_results.csv",
        "summary_csv": DATA_DIR / "experiment_47_model_hard_frequency_rocket_summary.csv",
        "stdout_log": DATA_DIR / "experiment_47_model_hard_frequency_rocket_stdout.log",
        "expected_datasets": 32,
    },
    "experiment_48_model_hard_shapelet_prototype": {
        "label": "Experiment 48 · Model-hard shapelet normal prototypes",
        "detail_csv": DATA_DIR / "experiment_48_model_hard_shapelet_prototype_results.csv",
        "summary_csv": DATA_DIR / "experiment_48_model_hard_shapelet_prototype_summary.csv",
        "stdout_log": DATA_DIR / "experiment_48_model_hard_shapelet_prototype_stdout.log",
        "expected_datasets": 17,
    },
    "experiment_49_model_hard_anomaly_injection": {
        "label": "Experiment 49 · Model-hard anomaly injection score",
        "detail_csv": DATA_DIR / "experiment_49_model_hard_anomaly_injection_results.csv",
        "summary_csv": DATA_DIR / "experiment_49_model_hard_anomaly_injection_summary.csv",
        "stdout_log": DATA_DIR / "experiment_49_model_hard_anomaly_injection_stdout.log",
        "expected_datasets": 46,
    },
    "experiment_50_model_hard_timeseries_imaging": {
        "label": "Experiment 50 · Model-hard time-series imaging smoke test",
        "detail_csv": DATA_DIR / "experiment_50_model_hard_timeseries_imaging_results.csv",
        "summary_csv": DATA_DIR / "experiment_50_model_hard_timeseries_imaging_summary.csv",
        "stdout_log": DATA_DIR / "experiment_50_model_hard_timeseries_imaging_stdout.log",
        "expected_datasets": 86,
    },
    "experiment_51_full_timeseries_imaging_selector_probe": {
        "label": "Experiment 51 · Full original time-series imaging selector probe",
        "detail_csv": DATA_DIR / "experiment_51_full_timeseries_imaging_selector_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_51_full_timeseries_imaging_selector_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_51_full_timeseries_imaging_selector_probe_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_52_imaging_multiscale_fusion_probe": {
        "label": "Experiment 52 · Imaging multiscale fusion probe",
        "detail_csv": DATA_DIR / "experiment_52_imaging_multiscale_fusion_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_52_imaging_multiscale_fusion_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_52_imaging_multiscale_fusion_probe_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_53a_imaging_texture_features_probe": {
        "label": "Experiment 53a · Imaging HOG/LBP texture feature probe",
        "detail_csv": DATA_DIR / "experiment_53a_imaging_texture_features_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_53a_imaging_texture_features_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_53a_imaging_texture_features_probe_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_53b_imaging_train_robust_scaling_probe": {
        "label": "Experiment 53b · Imaging train-normal robust scaling probe",
        "detail_csv": DATA_DIR / "experiment_53b_imaging_train_robust_scaling_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_53b_imaging_train_robust_scaling_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_53b_imaging_train_robust_scaling_probe_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_54_imaging_resolution_pca_sweep": {
        "label": "Experiment 54 · Imaging resolution and PCA compression sweep",
        "detail_csv": DATA_DIR / "experiment_54_imaging_resolution_pca_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_54_imaging_resolution_pca_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_54_imaging_resolution_pca_sweep_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_55_imaging_scaling_sweep": {
        "label": "Experiment 55 · Imaging train-normal and per-series scaling sweep",
        "detail_csv": DATA_DIR / "experiment_55_imaging_scaling_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_55_imaging_scaling_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_55_imaging_scaling_sweep_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_56_imaging_glcm_texture_probe": {
        "label": "Experiment 56 · Imaging GLCM texture feature probe",
        "detail_csv": DATA_DIR / "experiment_56_imaging_glcm_texture_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_56_imaging_glcm_texture_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_56_imaging_glcm_texture_probe_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_57_imaging_small_cnn_mps_probe": {
        "label": "Experiment 57 · Imaging small CNN MPS normal autoencoder probe",
        "detail_csv": DATA_DIR / "experiment_57_imaging_small_cnn_mps_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_57_imaging_small_cnn_mps_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_57_imaging_small_cnn_mps_probe_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_58_imaging_pretrained_cnn_feature_probe": {
        "label": "Experiment 58 · Imaging pretrained CNN feature probe",
        "detail_csv": DATA_DIR / "experiment_58_imaging_pretrained_cnn_feature_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_58_imaging_pretrained_cnn_feature_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_58_imaging_pretrained_cnn_feature_probe_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_59_rocket_imaging_selector": {
        "label": "Experiment 59 · ROCKET + Exp55/Exp56 selector",
        "detail_csv": DATA_DIR / "experiment_59_rocket_imaging_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_59_rocket_imaging_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_59_rocket_imaging_selector_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_60_selector_fp_guard_variants": {
        "label": "Experiment 60 · Selector FP guard variants",
        "detail_csv": DATA_DIR / "experiment_60_selector_fp_guard_variants_results.csv",
        "summary_csv": DATA_DIR / "experiment_60_selector_fp_guard_variants_summary.csv",
        "stdout_log": DATA_DIR / "experiment_60_selector_fp_guard_variants_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_61_selector_index_agreement": {
        "label": "Experiment 61 · Selector sample-index agreement",
        "detail_csv": DATA_DIR / "experiment_61_selector_index_agreement_results.csv",
        "summary_csv": DATA_DIR / "experiment_61_selector_index_agreement_summary.csv",
        "stdout_log": DATA_DIR / "experiment_61_selector_index_agreement_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_62_selector_guarded_index_agreement": {
        "label": "Experiment 62 · Selector guarded index agreement",
        "detail_csv": DATA_DIR / "experiment_62_selector_guarded_index_agreement_results.csv",
        "summary_csv": DATA_DIR / "experiment_62_selector_guarded_index_agreement_summary.csv",
        "stdout_log": DATA_DIR / "experiment_62_selector_guarded_index_agreement_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_63_guarded_cap_sweep": {
        "label": "Experiment 63 · Guarded cap sweep",
        "detail_csv": DATA_DIR / "experiment_63_guarded_cap_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_63_guarded_cap_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_63_guarded_cap_sweep_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_64_guarded_with_fallback": {
        "label": "Experiment 64 · Guarded selector fallback",
        "detail_csv": DATA_DIR / "experiment_64_guarded_with_fallback_results.csv",
        "summary_csv": DATA_DIR / "experiment_64_guarded_with_fallback_summary.csv",
        "stdout_log": DATA_DIR / "experiment_64_guarded_with_fallback_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_65_confidence_tier_selector": {
        "label": "Experiment 65 · Confidence tier selector",
        "detail_csv": DATA_DIR / "experiment_65_confidence_tier_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_65_confidence_tier_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_65_confidence_tier_selector_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_66_train_normal_alert_budget": {
        "label": "Experiment 66 · Train-normal alert budget",
        "detail_csv": DATA_DIR / "experiment_66_train_normal_alert_budget_results.csv",
        "summary_csv": DATA_DIR / "experiment_66_train_normal_alert_budget_summary.csv",
        "stdout_log": DATA_DIR / "experiment_66_train_normal_alert_budget_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_67_hard_family_fallback_selector": {
        "label": "Experiment 67 · Hardness signal fallback",
        "detail_csv": DATA_DIR / "experiment_67_hard_family_fallback_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_67_hard_family_fallback_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_67_hard_family_fallback_selector_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_68_final_operational_selector": {
        "label": "Experiment 68 · Final operational selector",
        "detail_csv": DATA_DIR / "experiment_68_final_operational_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_68_final_operational_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_68_final_operational_selector_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_68b_final_operational_fallback_sweep": {
        "label": "Experiment 68b · Final operational fallback sweep",
        "detail_csv": DATA_DIR / "experiment_68b_final_operational_fallback_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_68b_final_operational_fallback_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_68b_final_operational_fallback_sweep_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_69_operational_train_exceed_calibration": {
        "label": "Experiment 69 · Operational train exceed calibration",
        "detail_csv": DATA_DIR / "experiment_69_operational_train_exceed_calibration_results.csv",
        "summary_csv": DATA_DIR / "experiment_69_operational_train_exceed_calibration_summary.csv",
        "stdout_log": DATA_DIR / "experiment_69_operational_train_exceed_calibration_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_69b_no_prediction_fallback_calibration": {
        "label": "Experiment 69b · No-prediction fallback calibration",
        "detail_csv": DATA_DIR / "experiment_69b_no_prediction_fallback_calibration_results.csv",
        "summary_csv": DATA_DIR / "experiment_69b_no_prediction_fallback_calibration_summary.csv",
        "stdout_log": DATA_DIR / "experiment_69b_no_prediction_fallback_calibration_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_70_zero_mode_family_repair_selector": {
        "label": "Experiment 70 · Zero-mode family repair selector",
        "detail_csv": DATA_DIR / "experiment_70_zero_mode_family_repair_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_70_zero_mode_family_repair_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_70_zero_mode_family_repair_selector_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_71a_large_data_rocket_fallback": {
        "label": "Experiment 71a · Large-data ROCKET fallback repair",
        "detail_csv": DATA_DIR / "experiment_71a_large_data_rocket_fallback_results.csv",
        "summary_csv": DATA_DIR / "experiment_71a_large_data_rocket_fallback_summary.csv",
        "stdout_log": DATA_DIR / "experiment_71a_large_data_rocket_fallback_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_71b_large_data_rocket_review_tier": {
        "label": "Experiment 71b · Large-data ROCKET review tier",
        "detail_csv": DATA_DIR / "experiment_71b_large_data_rocket_review_tier_results.csv",
        "summary_csv": DATA_DIR / "experiment_71b_large_data_rocket_review_tier_summary.csv",
        "stdout_log": DATA_DIR / "experiment_71b_large_data_rocket_review_tier_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_72a_large_data_rank_ensemble": {
        "label": "Experiment 72a · Large-data rank ensemble repair",
        "detail_csv": DATA_DIR / "experiment_72a_large_data_rank_ensemble_results.csv",
        "summary_csv": DATA_DIR / "experiment_72a_large_data_rank_ensemble_summary.csv",
        "stdout_log": DATA_DIR / "experiment_72a_large_data_rank_ensemble_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_72b_large_data_source_disagreement": {
        "label": "Experiment 72b · Large-data source disagreement repair",
        "detail_csv": DATA_DIR / "experiment_72b_large_data_source_disagreement_results.csv",
        "summary_csv": DATA_DIR / "experiment_72b_large_data_source_disagreement_summary.csv",
        "stdout_log": DATA_DIR / "experiment_72b_large_data_source_disagreement_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_73a_large_rank_rocket_guard": {
        "label": "Experiment 73a · Large-rank ROCKET guard",
        "detail_csv": DATA_DIR / "experiment_73a_large_rank_rocket_guard_results.csv",
        "summary_csv": DATA_DIR / "experiment_73a_large_rank_rocket_guard_summary.csv",
        "stdout_log": DATA_DIR / "experiment_73a_large_rank_rocket_guard_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_73b_large_rank_two_model_guard": {
        "label": "Experiment 73b · Large-rank two-model guard",
        "detail_csv": DATA_DIR / "experiment_73b_large_rank_two_model_guard_results.csv",
        "summary_csv": DATA_DIR / "experiment_73b_large_rank_two_model_guard_summary.csv",
        "stdout_log": DATA_DIR / "experiment_73b_large_rank_two_model_guard_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_73c_large_rank_budget_guard": {
        "label": "Experiment 73c · Large-rank budget guard",
        "detail_csv": DATA_DIR / "experiment_73c_large_rank_budget_guard_results.csv",
        "summary_csv": DATA_DIR / "experiment_73c_large_rank_budget_guard_summary.csv",
        "stdout_log": DATA_DIR / "experiment_73c_large_rank_budget_guard_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_73d_large_rank_combined_guard": {
        "label": "Experiment 73d · Large-rank combined guard",
        "detail_csv": DATA_DIR / "experiment_73d_large_rank_combined_guard_results.csv",
        "summary_csv": DATA_DIR / "experiment_73d_large_rank_combined_guard_summary.csv",
        "stdout_log": DATA_DIR / "experiment_73d_large_rank_combined_guard_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_74a_large_rank_margin_guard": {
        "label": "Experiment 74a · Large-rank confidence margin guard",
        "detail_csv": DATA_DIR / "experiment_74a_large_rank_margin_guard_results.csv",
        "summary_csv": DATA_DIR / "experiment_74a_large_rank_margin_guard_summary.csv",
        "stdout_log": DATA_DIR / "experiment_74a_large_rank_margin_guard_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_74b_large_rank_family_budget": {
        "label": "Experiment 74b · Large-rank family budget guard",
        "detail_csv": DATA_DIR / "experiment_74b_large_rank_family_budget_results.csv",
        "summary_csv": DATA_DIR / "experiment_74b_large_rank_family_budget_summary.csv",
        "stdout_log": DATA_DIR / "experiment_74b_large_rank_family_budget_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_74c_large_rank_margin_family_guard": {
        "label": "Experiment 74c · Large-rank margin and family guard",
        "detail_csv": DATA_DIR / "experiment_74c_large_rank_margin_family_guard_results.csv",
        "summary_csv": DATA_DIR / "experiment_74c_large_rank_margin_family_guard_summary.csv",
        "stdout_log": DATA_DIR / "experiment_74c_large_rank_margin_family_guard_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_74d_large_rank_review_tier_split": {
        "label": "Experiment 74d · Large-rank alert/review tier split",
        "detail_csv": DATA_DIR / "experiment_74d_large_rank_review_tier_split_results.csv",
        "summary_csv": DATA_DIR / "experiment_74d_large_rank_review_tier_split_summary.csv",
        "stdout_log": DATA_DIR / "experiment_74d_large_rank_review_tier_split_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_75_score_source_diagnostic_harness": {
        "label": "Experiment 75 · Score-source diagnostic harness",
        "detail_csv": DATA_DIR / "experiment_75_score_source_diagnostic_harness_results.csv",
        "summary_csv": DATA_DIR / "experiment_75_score_source_diagnostic_harness_summary.csv",
        "stdout_log": DATA_DIR / "experiment_75_score_source_diagnostic_harness_stdout.log",
        "expected_datasets": 339,
    },
    "experiment_76_spectral_derivative_rocket_hard_family": {
        "label": "Experiment 76 · Spectral/derivative ROCKET hard-family score",
        "detail_csv": DATA_DIR / "experiment_76_spectral_derivative_rocket_hard_family_results.csv",
        "summary_csv": DATA_DIR / "experiment_76_spectral_derivative_rocket_hard_family_summary.csv",
        "stdout_log": DATA_DIR / "experiment_76_spectral_derivative_rocket_hard_family_stdout.log",
        "expected_datasets": 339,
    },
    "experiment_77_shapelet_prototype_low_train_guard": {
        "label": "Experiment 77 · Shapelet/prototype low-train guard",
        "detail_csv": DATA_DIR / "experiment_77_shapelet_prototype_low_train_guard_results.csv",
        "summary_csv": DATA_DIR / "experiment_77_shapelet_prototype_low_train_guard_summary.csv",
        "stdout_log": DATA_DIR / "experiment_77_shapelet_prototype_low_train_guard_stdout.log",
        "expected_datasets": 339,
    },
    "experiment_78_robust_density_corrected_score": {
        "label": "Experiment 78 · Robust density-corrected score",
        "detail_csv": DATA_DIR / "experiment_78_robust_density_corrected_score_results.csv",
        "summary_csv": DATA_DIR / "experiment_78_robust_density_corrected_score_summary.csv",
        "stdout_log": DATA_DIR / "experiment_78_robust_density_corrected_score_stdout.log",
        "expected_datasets": 339,
    },
    "experiment_79_vae_epoch_sweep_guarded": {
        "label": "Experiment 79 · VAE/AE epoch sweep guarded",
        "detail_csv": DATA_DIR / "experiment_79_vae_epoch_sweep_guarded_results.csv",
        "summary_csv": DATA_DIR / "experiment_79_vae_epoch_sweep_guarded_summary.csv",
        "stdout_log": DATA_DIR / "experiment_79_vae_epoch_sweep_guarded_stdout.log",
        "expected_datasets": 222,
    },
    "experiment_80_train_only_score_selector": {
        "label": "Experiment 80 · Train-only score selector",
        "detail_csv": DATA_DIR / "experiment_80_train_only_score_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_80_train_only_score_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_80_train_only_score_selector_stdout.log",
        "expected_datasets": 339,
    },
    "experiment_81_aeon_multirocket_official_full": {
        "label": "Experiment 81 · Official aeon MultiROCKET full",
        "detail_csv": DATA_DIR / "experiment_81_aeon_multirocket_official_full_results.csv",
        "summary_csv": DATA_DIR / "experiment_81_aeon_multirocket_official_full_summary.csv",
        "stdout_log": DATA_DIR / "experiment_81_aeon_multirocket_official_full_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_82_hydra_hard_family_subset": {
        "label": "Experiment 82 · HYDRA hard-family subset",
        "detail_csv": DATA_DIR / "experiment_82_hydra_hard_family_subset_results.csv",
        "summary_csv": DATA_DIR / "experiment_82_hydra_hard_family_subset_summary.csv",
        "stdout_log": DATA_DIR / "experiment_82_hydra_hard_family_subset_stdout.log",
        "expected_datasets": 84,
    },
    "experiment_83_multirocket_hydra_hard339": {
        "label": "Experiment 83 · MultiROCKET+HYDRA hard339",
        "detail_csv": DATA_DIR / "experiment_83_multirocket_hydra_hard339_results.csv",
        "summary_csv": DATA_DIR / "experiment_83_multirocket_hydra_hard339_summary.csv",
        "stdout_log": DATA_DIR / "experiment_83_multirocket_hydra_hard339_stdout.log",
        "expected_datasets": 339,
    },
    "experiment_84_feature_pruning_operational_stability": {
        "label": "Experiment 84 · Feature pruning operational stability",
        "detail_csv": DATA_DIR / "experiment_84_feature_pruning_operational_stability_results.csv",
        "summary_csv": DATA_DIR / "experiment_84_feature_pruning_operational_stability_summary.csv",
        "stdout_log": DATA_DIR / "experiment_84_feature_pruning_operational_stability_stdout.log",
        "expected_datasets": 339,
    },
    "experiment_85_exp84_hard_specialist_selector": {
        "label": "Experiment 85 · Exp84 hard specialist selector",
        "detail_csv": DATA_DIR / "experiment_85_exp84_hard_specialist_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_85_exp84_hard_specialist_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_85_exp84_hard_specialist_selector_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_86_train_only_agreement_exp84_selector": {
        "label": "Experiment 86 · Train-only agreement Exp84 selector",
        "detail_csv": DATA_DIR / "experiment_86_train_only_agreement_exp84_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_86_train_only_agreement_exp84_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_86_train_only_agreement_exp84_selector_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_87_exp84_index_diagnostics": {
        "label": "Experiment 87 · Exp84 index diagnostics",
        "detail_csv": DATA_DIR / "experiment_87_exp84_index_diagnostics_results.csv",
        "summary_csv": DATA_DIR / "experiment_87_exp84_index_diagnostics_summary.csv",
        "stdout_log": DATA_DIR / "experiment_87_exp84_index_diagnostics_stdout.log",
        "expected_datasets": 339,
    },
    "experiment_88_true_agreement_exp84_selector": {
        "label": "Experiment 88 · True agreement Exp84 selector",
        "detail_csv": DATA_DIR / "experiment_88_true_agreement_exp84_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_88_true_agreement_exp84_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_88_true_agreement_exp84_selector_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_89_74d_with_exp84_candidate": {
        "label": "Experiment 89 · 74d with Exp84 candidate",
        "detail_csv": DATA_DIR / "experiment_89_74d_with_exp84_candidate_results.csv",
        "summary_csv": DATA_DIR / "experiment_89_74d_with_exp84_candidate_summary.csv",
        "stdout_log": DATA_DIR / "experiment_89_74d_with_exp84_candidate_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_90_zero_f1_repair_selector": {
        "label": "Experiment 90 · Zero-F1 repair selector",
        "detail_csv": DATA_DIR / "experiment_90_zero_f1_repair_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_90_zero_f1_repair_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_90_zero_f1_repair_selector_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_91_guarded_candidate_union_repair": {
        "label": "Experiment 91 · Guarded candidate-union repair",
        "detail_csv": DATA_DIR / "experiment_91_guarded_candidate_union_repair_results.csv",
        "summary_csv": DATA_DIR / "experiment_91_guarded_candidate_union_repair_summary.csv",
        "stdout_log": DATA_DIR / "experiment_91_guarded_candidate_union_repair_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_92_operational_hybrid_selector": {
        "label": "Experiment 92 · Operational hybrid selector",
        "detail_csv": DATA_DIR / "experiment_92_operational_hybrid_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_92_operational_hybrid_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_92_operational_hybrid_selector_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_93_nonpos_candidate_reranker": {
        "label": "Experiment 93 · Non-position candidate reranker",
        "detail_csv": DATA_DIR / "experiment_93_nonpos_candidate_reranker_results.csv",
        "summary_csv": DATA_DIR / "experiment_93_nonpos_candidate_reranker_summary.csv",
        "stdout_log": DATA_DIR / "experiment_93_nonpos_candidate_reranker_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_113_train_normal_conformal_fusion": {
        "label": "Experiment 113 · Train-normal conformal fusion",
        "detail_csv": DATA_DIR / "experiment_113_train_normal_conformal_fusion_results.csv",
        "summary_csv": DATA_DIR / "experiment_113_train_normal_conformal_fusion_summary.csv",
        "stdout_log": DATA_DIR / "experiment_113_train_normal_conformal_fusion_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_114_pseudo_anomaly_score_probe": {
        "label": "Experiment 114 · Pseudo-anomaly score probe",
        "detail_csv": DATA_DIR / "experiment_114_pseudo_anomaly_score_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_114_pseudo_anomaly_score_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_114_pseudo_anomaly_score_probe_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_115_local_normal_state_score": {
        "label": "Experiment 115 · Local normal-state score",
        "detail_csv": DATA_DIR / "experiment_115_local_normal_state_score_results.csv",
        "summary_csv": DATA_DIR / "experiment_115_local_normal_state_score_summary.csv",
        "stdout_log": DATA_DIR / "experiment_115_local_normal_state_score_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_116_train_only_reliability_reranker": {
        "label": "Experiment 116 · Train-only reliability reranker",
        "detail_csv": DATA_DIR / "experiment_116_train_only_reliability_reranker_results.csv",
        "summary_csv": DATA_DIR / "experiment_116_train_only_reliability_reranker_summary.csv",
        "stdout_log": DATA_DIR / "experiment_116_train_only_reliability_reranker_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_117_train_only_candidate_budget": {
        "label": "Experiment 117 · Train-only candidate budget",
        "detail_csv": DATA_DIR / "experiment_117_train_only_candidate_budget_results.csv",
        "summary_csv": DATA_DIR / "experiment_117_train_only_candidate_budget_summary.csv",
        "stdout_log": DATA_DIR / "experiment_117_train_only_candidate_budget_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_118_rocket512_knn3_exp93_source_probe": {
        "label": "Experiment 118 · ROCKET-512 KNN-3 Exp93 source probe",
        "detail_csv": DATA_DIR / "experiment_118_rocket512_knn3_exp93_source_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_118_rocket512_knn3_exp93_source_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_118_rocket512_knn3_exp93_source_probe_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_119a_exp93_rank_order_validation": {
        "label": "Experiment 119a · Exp93 rank-order validation",
        "detail_csv": DATA_DIR / "experiment_119a_exp93_rank_order_validation_results.csv",
        "summary_csv": DATA_DIR / "experiment_119a_exp93_rank_order_validation_summary.csv",
        "stdout_log": DATA_DIR / "experiment_119a_exp93_rank_order_validation_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_119b_rocket256_512_validated_rank_compare": {
        "label": "Experiment 119b · ROCKET 256/512 validated rank comparison",
        "detail_csv": DATA_DIR / "experiment_119b_rocket256_512_validated_rank_compare_results.csv",
        "summary_csv": DATA_DIR / "experiment_119b_rocket256_512_validated_rank_compare_summary.csv",
        "stdout_log": DATA_DIR / "experiment_119b_rocket256_512_validated_rank_compare_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_120_exp93_reference_rebaseline": {
        "label": "Experiment 120 · Exp93 reference rebaseline",
        "detail_csv": DATA_DIR / "experiment_120_exp93_reference_rebaseline_results.csv",
        "summary_csv": DATA_DIR / "experiment_120_exp93_reference_rebaseline_summary.csv",
        "stdout_log": DATA_DIR / "experiment_120_exp93_reference_rebaseline_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_121_exp95_validated_rank_review": {
        "label": "Experiment 121 · Exp95 validated-rank review",
        "detail_csv": DATA_DIR / "experiment_121_exp95_validated_rank_review_results.csv",
        "summary_csv": DATA_DIR / "experiment_121_exp95_validated_rank_review_summary.csv",
        "stdout_log": DATA_DIR / "experiment_121_exp95_validated_rank_review_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_122_exp96_validated_rank_workflow": {
        "label": "Experiment 122 · Exp96 validated-rank workflow",
        "detail_csv": DATA_DIR / "experiment_122_exp96_validated_rank_workflow_results.csv",
        "summary_csv": DATA_DIR / "experiment_122_exp96_validated_rank_workflow_summary.csv",
        "stdout_log": DATA_DIR / "experiment_122_exp96_validated_rank_workflow_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_123_exp103_validated_rank_review_sources": {
        "label": "Experiment 123 · Exp103 validated-rank review sources",
        "detail_csv": DATA_DIR / "experiment_123_exp103_validated_rank_review_sources_results.csv",
        "summary_csv": DATA_DIR / "experiment_123_exp103_validated_rank_review_sources_summary.csv",
        "stdout_log": DATA_DIR / "experiment_123_exp103_validated_rank_review_sources_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_124_exp104_validated_rank_dimensions": {
        "label": "Experiment 124 · Exp104 validated-rank dimensions",
        "detail_csv": DATA_DIR / "experiment_124_exp104_validated_rank_dimensions_results.csv",
        "summary_csv": DATA_DIR / "experiment_124_exp104_validated_rank_dimensions_summary.csv",
        "stdout_log": DATA_DIR / "experiment_124_exp104_validated_rank_dimensions_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_125_exp105_validated_rank_combinations": {
        "label": "Experiment 125 · Exp105 validated-rank combinations",
        "detail_csv": DATA_DIR / "experiment_125_exp105_validated_rank_combinations_results.csv",
        "summary_csv": DATA_DIR / "experiment_125_exp105_validated_rank_combinations_summary.csv",
        "stdout_log": DATA_DIR / "experiment_125_exp105_validated_rank_combinations_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_126_exp106_validated_rank_gated_combo": {
        "label": "Experiment 126 · Exp106 validated-rank gated combo",
        "detail_csv": DATA_DIR / "experiment_126_exp106_validated_rank_gated_combo_results.csv",
        "summary_csv": DATA_DIR / "experiment_126_exp106_validated_rank_gated_combo_summary.csv",
        "stdout_log": DATA_DIR / "experiment_126_exp106_validated_rank_gated_combo_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_127_exp107_validated_rank_disagreement": {
        "label": "Experiment 127 · Exp107 validated-rank disagreement",
        "detail_csv": DATA_DIR / "experiment_127_exp107_validated_rank_disagreement_results.csv",
        "summary_csv": DATA_DIR / "experiment_127_exp107_validated_rank_disagreement_summary.csv",
        "stdout_log": DATA_DIR / "experiment_127_exp107_validated_rank_disagreement_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_128_rocket512_only_review_selector": {
        "label": "Experiment 128 · ROCKET-512 only review selector",
        "detail_csv": DATA_DIR / "experiment_128_rocket512_only_review_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_128_rocket512_only_review_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_128_rocket512_only_review_selector_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_129_rocket512_base_selector_rebuild": {
        "label": "Experiment 129 · ROCKET-512 base selector rebuild",
        "detail_csv": DATA_DIR / "experiment_129_rocket512_base_selector_rebuild_results.csv",
        "summary_csv": DATA_DIR / "experiment_129_rocket512_base_selector_rebuild_summary.csv",
        "stdout_log": DATA_DIR / "experiment_129_rocket512_base_selector_rebuild_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_130_rocket512_two_block_knn": {
        "label": "Experiment 130 · ROCKET-512 two-block KNN",
        "detail_csv": DATA_DIR / "experiment_130_rocket512_two_block_knn_results.csv",
        "summary_csv": DATA_DIR / "experiment_130_rocket512_two_block_knn_summary.csv",
        "stdout_log": DATA_DIR / "experiment_130_rocket512_two_block_knn_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_131_rocket_block_b_calibration": {
        "label": "Experiment 131 · ROCKET Block-B calibration",
        "detail_csv": DATA_DIR / "experiment_131_rocket_block_b_calibration_results.csv",
        "summary_csv": DATA_DIR / "experiment_131_rocket_block_b_calibration_summary.csv",
        "stdout_log": DATA_DIR / "experiment_131_rocket_block_b_calibration_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_132_block_b_review_integration": {
        "label": "Experiment 132 · Block-B review integration",
        "detail_csv": DATA_DIR / "experiment_132_block_b_review_integration_results.csv",
        "summary_csv": DATA_DIR / "experiment_132_block_b_review_integration_summary.csv",
        "stdout_log": DATA_DIR / "experiment_132_block_b_review_integration_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_133_block_b_confidence_tiers": {
        "label": "Experiment 133 · Block-B confidence tiers",
        "detail_csv": DATA_DIR / "experiment_133_block_b_confidence_tiers_results.csv",
        "summary_csv": DATA_DIR / "experiment_133_block_b_confidence_tiers_summary.csv",
        "stdout_log": DATA_DIR / "experiment_133_block_b_confidence_tiers_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_134_block_b_review_tail_guard": {
        "label": "Experiment 134 · Block-B review tail guard",
        "detail_csv": DATA_DIR / "experiment_134_block_b_review_tail_guard_results.csv",
        "summary_csv": DATA_DIR / "experiment_134_block_b_review_tail_guard_summary.csv",
        "stdout_log": DATA_DIR / "experiment_134_block_b_review_tail_guard_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_135_block_c_review_confirmation": {
        "label": "Experiment 135 · Block-C review confirmation",
        "detail_csv": DATA_DIR / "experiment_135_block_c_review_confirmation_results.csv",
        "summary_csv": DATA_DIR / "experiment_135_block_c_review_confirmation_summary.csv",
        "stdout_log": DATA_DIR / "experiment_135_block_c_review_confirmation_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_136_family_holdout_review_audit": {
        "label": "Experiment 136 · Family holdout review audit",
        "detail_csv": DATA_DIR / "experiment_136_family_holdout_review_audit_results.csv",
        "summary_csv": DATA_DIR / "experiment_136_family_holdout_review_audit_summary.csv",
        "stdout_log": DATA_DIR / "experiment_136_family_holdout_review_audit_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_137_operational_triage": {
        "label": "Experiment 137 · Operational alert triage",
        "detail_csv": DATA_DIR / "experiment_137_operational_triage_results.csv",
        "summary_csv": DATA_DIR / "experiment_137_operational_triage_summary.csv",
        "stdout_log": DATA_DIR / "experiment_137_operational_triage_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_94_nonpos_rank_consensus_v2": {
        "label": "Experiment 94 · Non-position rank consensus v2",
        "detail_csv": DATA_DIR / "experiment_94_nonpos_rank_consensus_v2_results.csv",
        "summary_csv": DATA_DIR / "experiment_94_nonpos_rank_consensus_v2_summary.csv",
        "stdout_log": DATA_DIR / "experiment_94_nonpos_rank_consensus_v2_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_95_topk_review_tier": {
        "label": "Experiment 95 · Top-k review tier",
        "detail_csv": DATA_DIR / "experiment_95_topk_review_tier_results.csv",
        "summary_csv": DATA_DIR / "experiment_95_topk_review_tier_summary.csv",
        "stdout_log": DATA_DIR / "experiment_95_topk_review_tier_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_96_review_tier_operational_workflow": {
        "label": "Experiment 96 · Review tier operational workflow",
        "detail_csv": DATA_DIR / "experiment_96_review_tier_operational_workflow_results.csv",
        "summary_csv": DATA_DIR / "experiment_96_review_tier_operational_workflow_summary.csv",
        "stdout_log": DATA_DIR / "experiment_96_review_tier_operational_workflow_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_94b_corrected_nonpos_rank_consensus": {
        "label": "Experiment 94b · Corrected non-position rank consensus",
        "detail_csv": DATA_DIR / "experiment_94b_corrected_nonpos_rank_consensus_results.csv",
        "summary_csv": DATA_DIR / "experiment_94b_corrected_nonpos_rank_consensus_summary.csv",
        "stdout_log": DATA_DIR / "experiment_94b_corrected_nonpos_rank_consensus_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_97_zero_f1_feature_need_diagnosis": {
        "label": "Experiment 97 · Zero-F1 feature need diagnosis",
        "detail_csv": DATA_DIR / "experiment_97_zero_f1_feature_need_diagnosis_results.csv",
        "summary_csv": DATA_DIR / "experiment_97_zero_f1_feature_need_diagnosis_summary.csv",
        "stdout_log": DATA_DIR / "experiment_97_zero_f1_feature_need_diagnosis_stdout.log",
        "expected_datasets": 239,
    },
    "experiment_98_tiny_train_normal_pooling": {
        "label": "Experiment 98 · Tiny-train normal pooling",
        "detail_csv": DATA_DIR / "experiment_98_tiny_train_normal_pooling_results.csv",
        "summary_csv": DATA_DIR / "experiment_98_tiny_train_normal_pooling_summary.csv",
        "stdout_log": DATA_DIR / "experiment_98_tiny_train_normal_pooling_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_98b_train_only_tiny_pooling": {
        "label": "Experiment 98b · Train-only tiny pooling",
        "detail_csv": DATA_DIR / "experiment_98b_train_only_tiny_pooling_results.csv",
        "summary_csv": DATA_DIR / "experiment_98b_train_only_tiny_pooling_summary.csv",
        "stdout_log": DATA_DIR / "experiment_98b_train_only_tiny_pooling_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_99_spectral_derivative_feature_score": {
        "label": "Experiment 99 · Spectral/derivative feature score",
        "detail_csv": DATA_DIR / "experiment_99_spectral_derivative_feature_score_results.csv",
        "summary_csv": DATA_DIR / "experiment_99_spectral_derivative_feature_score_summary.csv",
        "stdout_log": DATA_DIR / "experiment_99_spectral_derivative_feature_score_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_100_spectral_review_and_guard": {
        "label": "Experiment 100 · Spectral review and guard",
        "detail_csv": DATA_DIR / "experiment_100_spectral_review_and_guard_results.csv",
        "summary_csv": DATA_DIR / "experiment_100_spectral_review_and_guard_summary.csv",
        "stdout_log": DATA_DIR / "experiment_100_spectral_review_and_guard_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_101_shapelet_normal_prototype": {
        "label": "Experiment 101 · Shapelet normal prototype",
        "detail_csv": DATA_DIR / "experiment_101_shapelet_normal_prototype_results.csv",
        "summary_csv": DATA_DIR / "experiment_101_shapelet_normal_prototype_summary.csv",
        "stdout_log": DATA_DIR / "experiment_101_shapelet_normal_prototype_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_102_feature_source_selector": {
        "label": "Experiment 102 · Feature source selector",
        "detail_csv": DATA_DIR / "experiment_102_feature_source_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_102_feature_source_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_102_feature_source_selector_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_103_higher_dim_review_sources": {
        "label": "Experiment 103 · Higher-dim review sources",
        "detail_csv": DATA_DIR / "experiment_103_higher_dim_review_sources_results.csv",
        "summary_csv": DATA_DIR / "experiment_103_higher_dim_review_sources_summary.csv",
        "stdout_log": DATA_DIR / "experiment_103_higher_dim_review_sources_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_104_score_dimensionality_sweep": {
        "label": "Experiment 104 · Score dimensionality sweep",
        "detail_csv": DATA_DIR / "experiment_104_score_dimensionality_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_104_score_dimensionality_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_104_score_dimensionality_sweep_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_105_score_combination_methods": {
        "label": "Experiment 105 · Score combination methods",
        "detail_csv": DATA_DIR / "experiment_105_score_combination_methods_results.csv",
        "summary_csv": DATA_DIR / "experiment_105_score_combination_methods_summary.csv",
        "stdout_log": DATA_DIR / "experiment_105_score_combination_methods_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_106_gated_score_combo_selector": {
        "label": "Experiment 106 · Gated score combo selector",
        "detail_csv": DATA_DIR / "experiment_106_gated_score_combo_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_106_gated_score_combo_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_106_gated_score_combo_selector_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_107_exp103_combo_disagreement": {
        "label": "Experiment 107 · Exp103/combo disagreement",
        "detail_csv": DATA_DIR / "experiment_107_exp103_combo_disagreement_results.csv",
        "summary_csv": DATA_DIR / "experiment_107_exp103_combo_disagreement_summary.csv",
        "stdout_log": DATA_DIR / "experiment_107_exp103_combo_disagreement_stdout.log",
        "expected_datasets": 1117,
    },
    "experiment_108_imaging_pretrained_vit_feature_probe": {
        "label": "Experiment 108 · Imaging pretrained ViT hard-subset probe",
        "detail_csv": DATA_DIR / "experiment_108_imaging_pretrained_vit_feature_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_108_imaging_pretrained_vit_feature_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_108_imaging_pretrained_vit_feature_probe_stdout.log",
        "expected_datasets": 339,
    },
    "experiment_109_vit_compression_alternatives": {
        "label": "Experiment 109 · ViT compression alternatives",
        "detail_csv": DATA_DIR / "experiment_109_vit_compression_alternatives_results.csv",
        "summary_csv": DATA_DIR / "experiment_109_vit_compression_alternatives_summary.csv",
        "stdout_log": DATA_DIR / "experiment_109_vit_compression_alternatives_stdout.log",
        "expected_datasets": 339,
    },
    "experiment_110_exp84_score_backend_probe": {
        "label": "Experiment 110 · Exp84 score backend probe",
        "detail_csv": DATA_DIR / "experiment_110_exp84_score_backend_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_110_exp84_score_backend_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_110_exp84_score_backend_probe_stdout.log",
        "expected_datasets": 90,
    },
    "experiment_111_vit_fast_compression_probe": {
        "label": "Experiment 111 · Cached ViT compression probe",
        "detail_csv": DATA_DIR / "experiment_111_vit_fast_compression_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_111_vit_fast_compression_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_111_vit_fast_compression_probe_stdout.log",
        "expected_datasets": 90,
    },
    "experiment_111b_vit_manifold_compression_probe": {
        "label": "Experiment 111b · ViT manifold compression probe",
        "detail_csv": DATA_DIR / "experiment_111b_vit_manifold_compression_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_111b_vit_manifold_compression_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_111b_vit_manifold_compression_probe_stdout.log",
        "expected_datasets": 45,
    },
    "experiment_112_parametric_umap_oof_probe": {
        "label": "Experiment 112 · Parametric UMAP OOF probe",
        "detail_csv": DATA_DIR / "experiment_112_parametric_umap_oof_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_112_parametric_umap_oof_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_112_parametric_umap_oof_probe_stdout.log",
        "expected_datasets": 15,
    },
}

EXCLUDED_OPERATIONAL_EXPERIMENTS = {
    "experiment_70_zero_mode_family_repair_selector": "배제: family zero-mode policy is derived from prior labeled failure diagnostics",
    "experiment_85_exp84_hard_specialist_selector": "배제: best gain-family selector uses prior benchmark family gains",
    "experiment_91_guarded_candidate_union_repair": "배제: tail-position replacement uses evaluation-layout leakage",
    "experiment_117_train_only_candidate_budget": "배제: corrected Exp93-base budget adds 7 FP for one small F1 gain; conflicts with the false-alarm-first operating objective",
    "experiment_92_operational_hybrid_selector": "배제: tail-position replacement uses evaluation-layout leakage",
}

OPERATIONAL_DEFAULT_SELECTORS = {
    "experiment_90_zero_f1_repair_selector": "noalert_top1_train_safe_repair",
    "experiment_93_nonpos_candidate_reranker": "nonpos_weak_alert_replace",
}
PRIMARY_OPERATIONAL_EXPERIMENT = "experiment_93_nonpos_candidate_reranker"


INDEX_HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Rank Experiment Monitor</title>
  <style>
    :root {
      --bg: #f6f8fb;
      --panel: #ffffff;
      --panel-2: #f1f5f9;
      --text: #152033;
      --muted: #64748b;
      --line: #dbe3ee;
      --accent: #0f766e;
      --accent-2: #2563eb;
      --warn: #b45309;
      --bad: #b91c1c;
      --good: #15803d;
      --shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
      padding: 22px 28px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,.9);
      position: sticky;
      top: 0;
      z-index: 10;
      backdrop-filter: blur(12px);
    }
    h1 { margin: 0; font-size: 22px; line-height: 1.2; font-weight: 760; }
    .sub { margin-top: 4px; color: var(--muted); font-size: 13px; }
    .status {
      display: flex;
      align-items: center;
      gap: 10px;
      min-width: 220px;
      justify-content: flex-end;
      color: var(--muted);
      font-size: 13px;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--warn);
      box-shadow: 0 0 0 4px rgba(180,83,9,.12);
    }
    .dot.running {
      background: var(--good);
      box-shadow: 0 0 0 4px rgba(21,128,61,.13);
    }
    main { padding: 24px 28px 34px; max-width: 1500px; width: 100%; margin: 0 auto; }
    .grid { display: grid; gap: 16px; min-width: 0; }
    .grid > * { min-width: 0; }
    .metrics { grid-template-columns: repeat(5, minmax(150px, 1fr)); }
    .ops { grid-template-columns: repeat(5, minmax(150px, 1fr)); margin-top: 16px; }
    .two { grid-template-columns: minmax(0, 1.05fr) minmax(420px, .95fr); margin-top: 16px; }
    .three { grid-template-columns: minmax(0, .85fr) minmax(0, 1.15fr); margin-top: 16px; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      min-width: 0;
    }
    .card { padding: 16px; min-height: 110px; }
    .label { color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; }
    .label-row { display: inline-flex; align-items: center; gap: 6px; }
    .help {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 18px;
      height: 18px;
      border: 1px solid var(--line);
      border-radius: 50%;
      background: #fff;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      cursor: help;
    }
    .help:focus { outline: 2px solid rgba(37,99,235,.25); outline-offset: 2px; }
    .value { margin-top: 8px; font-size: 30px; font-weight: 780; line-height: 1; }
    .resource-value { font-size: 20px; line-height: 1.18; white-space: nowrap; }
    .long-value {
      font-size: 16px;
      line-height: 1.22;
      overflow-wrap: anywhere;
      word-break: break-word;
      white-space: normal;
    }
    .hint { margin-top: 8px; color: var(--muted); font-size: 12px; line-height: 1.35; }
    .bar {
      height: 10px;
      border-radius: 999px;
      overflow: hidden;
      background: #e2e8f0;
      margin-top: 14px;
    }
    .bar span { display: block; height: 100%; width: 0; background: linear-gradient(90deg, var(--accent), var(--accent-2)); }
    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 15px 16px;
      border-bottom: 1px solid var(--line);
    }
    .panel-head h2 { margin: 0; font-size: 15px; }
    .pill { color: var(--muted); font-size: 12px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 10px 12px; text-align: right; border-bottom: 1px solid #edf2f7; white-space: nowrap; }
    th:first-child, td:first-child { text-align: left; }
    th { color: var(--muted); font-size: 11px; text-transform: uppercase; background: var(--panel-2); }
    tr.best td:first-child { color: var(--accent); font-weight: 760; }
    tr.selectable { cursor: pointer; }
    tr.selected td { background: #ecfeff; }
    .table-wrap { overflow: auto; max-height: 430px; min-width: 0; }
    .log {
      margin: 0;
      padding: 14px 16px;
      height: 310px;
      overflow: auto;
      background: #0f172a;
      color: #dbeafe;
      font-size: 12px;
      line-height: 1.55;
      border-radius: 0 0 8px 8px;
      white-space: pre-wrap;
    }
    .timeline { padding: 12px 16px 16px; display: grid; gap: 10px; }
    .event { display: grid; grid-template-columns: 84px 1fr auto; gap: 12px; align-items: center; font-size: 13px; }
    .event .time { color: var(--muted); font-variant-numeric: tabular-nums; }
    .event .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .event .meta { color: var(--muted); font-size: 12px; }
    .detail-box { padding: 12px 16px 16px; display: grid; gap: 12px; }
    .detail-title { font-size: 13px; font-weight: 760; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .mini-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; }
    .mini-stat { background: var(--panel-2); border: 1px solid var(--line); border-radius: 8px; padding: 10px; min-height: 62px; }
    .mini-stat .k { color: var(--muted); font-size: 11px; font-weight: 700; text-transform: uppercase; }
    .mini-stat .v { margin-top: 6px; font-size: 18px; font-weight: 760; }
    .metric-help-grid { padding: 12px 16px 16px; display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
    .metric-help-card { border: 1px solid var(--line); border-radius: 8px; background: #fff; padding: 12px; min-width: 0; }
    .metric-help-card .group { color: var(--muted); font-size: 11px; font-weight: 760; text-transform: uppercase; }
    .metric-help-card h3 { margin: 5px 0 6px; font-size: 14px; line-height: 1.25; }
    .metric-help-card p { margin: 0 0 7px; color: var(--text); font-size: 13px; line-height: 1.45; }
    .metric-help-card .example { color: var(--muted); }
    .metric-help-panel { margin-top: 16px; }
    .metric-help-summary {
      cursor: pointer;
      list-style: none;
      border-bottom: 0;
      user-select: none;
    }
    .metric-help-summary::-webkit-details-marker { display: none; }
    .metric-help-panel[open] .metric-help-summary { border-bottom: 1px solid var(--line); }
    .summary-main { display: grid; gap: 3px; }
    .summary-main h2 { margin: 0; font-size: 15px; }
    .summary-main .hint { margin: 0; text-transform: none; font-weight: 500; }
    .summary-action {
      color: var(--accent);
      font-size: 12px;
      font-weight: 760;
    }
    .summary-action::before { content: "펼치기"; }
    .metric-help-panel[open] .summary-action::before { content: "접기"; }
    .metric-popover {
      position: fixed;
      z-index: 30;
      display: none;
      width: min(360px, calc(100vw - 32px));
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 18px 45px rgba(15, 23, 42, .18);
      padding: 12px;
    }
    .metric-popover.visible { display: block; }
    .metric-popover-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
    .metric-popover .group { color: var(--muted); font-size: 11px; font-weight: 760; text-transform: uppercase; }
    .metric-popover h3 { margin: 4px 0 7px; font-size: 15px; line-height: 1.25; }
    .metric-popover p { margin: 0 0 8px; font-size: 13px; line-height: 1.45; }
    .metric-popover .example { color: var(--muted); }
    .metric-popover-close {
      width: 24px;
      height: 24px;
      border: 1px solid var(--line);
      border-radius: 50%;
      background: #fff;
      color: var(--muted);
      font-weight: 800;
      cursor: pointer;
    }
    .breakdown { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .kv-list { margin: 0; padding: 0; list-style: none; display: grid; gap: 6px; font-size: 13px; }
    .kv-list li { display: flex; justify-content: space-between; gap: 10px; border-bottom: 1px solid #edf2f7; padding-bottom: 6px; }
    .kv-list .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .tier { font-weight: 760; color: var(--accent); }
    .delta-pos { color: var(--good); font-weight: 760; }
    .delta-neg { color: var(--bad); font-weight: 760; }
    .status-chip {
      display: inline-flex;
      align-items: center;
      height: 22px;
      padding: 0 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      color: var(--good);
      background: rgba(21, 128, 61, .10);
    }
    .muted-cell { color: var(--muted); }
    .footer { margin-top: 16px; color: var(--muted); font-size: 12px; display: flex; justify-content: space-between; gap: 16px; }
    .toolbar { display: flex; align-items: center; gap: 8px; }
    .toolbar select { border: 1px solid var(--line); background: #fff; color: var(--text); border-radius: 6px; padding: 5px 8px; font-size: 12px; }
    .action-button { border: 1px solid var(--line); background: #fff; color: var(--text); border-radius: 6px; padding: 7px 10px; font-size: 12px; font-weight: 700; cursor: pointer; }
    .action-button:hover { border-color: #94a3b8; background: #f8fafc; }
    .context-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }
    .context-item { border: 1px solid var(--line); padding: 10px; border-radius: 6px; }
    .context-item .k { color: var(--muted); font-size: 11px; text-transform: uppercase; }
    .context-item .v { margin-top: 4px; font-size: 15px; font-weight: 700; overflow-wrap: anywhere; }
    @media (max-width: 980px) {
      header { align-items: flex-start; flex-direction: column; }
      .status { justify-content: flex-start; }
      .metrics, .ops, .two, .three, .breakdown, .mini-grid, .metric-help-grid, .context-grid { grid-template-columns: 1fr; }
      main { padding: 18px; }
      .value { font-size: 26px; }
      .footer, .event { grid-template-columns: 1fr; flex-direction: column; align-items: flex-start; }
      .event .meta { white-space: normal; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Rank Threshold Calibration Monitor</h1>
      <div class="sub">실험 진행률, 현재 처리 데이터셋, threshold 전략 성능을 실시간으로 확인합니다.</div>
    </div>
    <div class="status"><span id="statusDot" class="dot"></span><span id="statusText">연결 중</span></div>
  </header>
  <main>
    <section class="grid metrics">
      <div class="panel card">
        <div class="label label-row">Progress <button class="help" type="button" data-metric="progress" aria-label="Progress 설명">?</button></div>
        <div class="value" id="progressValue">0%</div>
        <div class="bar"><span id="progressBar"></span></div>
        <div class="hint" id="datasetCount">0 / 1117 datasets</div>
      </div>
      <div class="panel card">
        <div class="label label-row">Current Dataset <button class="help" type="button" data-metric="current_dataset" aria-label="Current Dataset 설명">?</button></div>
        <div class="value long-value" id="currentDataset">-</div>
        <div class="hint" id="currentMeta">waiting for log</div>
      </div>
      <div class="panel card">
        <div class="label label-row">Runtime <button class="help" type="button" data-metric="runtime" aria-label="Runtime 설명">?</button></div>
        <div class="value" id="runtime">-</div>
        <div class="hint" id="eta">ETA 계산 중</div>
      </div>
      <div class="panel card">
        <div class="label label-row">Best Strategy <button class="help" type="button" data-metric="f1" aria-label="Best Strategy 설명">?</button></div>
        <div class="value long-value" id="bestStrategy">-</div>
        <div class="hint" id="bestMetric">mean F1 -</div>
      </div>
      <div class="panel card">
        <div class="label">Resources</div>
        <div class="value resource-value" id="cpu">-</div>
        <div class="hint" id="pid">PID -</div>
        <div class="hint" id="gpu">GPU -</div>
      </div>
    </section>

    <section class="grid ops">
      <div class="panel card">
        <div class="label label-row">Operational Pick <button class="help" type="button" data-metric="operational_pick" aria-label="Operational Pick 설명">?</button></div>
        <div class="value long-value" id="operationalPick">-</div>
        <div class="hint" id="operationalTier">운영 후보 판정 대기</div>
      </div>
      <div class="panel card">
        <div class="label label-row">Mean FP <button class="help" type="button" data-metric="mean_fp" aria-label="Mean FP 설명">?</button></div>
        <div class="value" id="meanFp">-</div>
        <div class="hint">낮을수록 사용자 피로도 감소</div>
      </div>
      <div class="panel card">
        <div class="label label-row">Alert Precision <button class="help" type="button" data-metric="alert_precision" aria-label="Alert Precision 설명">?</button></div>
        <div class="value" id="alertPrecision">-</div>
        <div class="hint">알람 중 실제 이상 비율</div>
      </div>
      <div class="panel card">
        <div class="label label-row">Train Exceed <button class="help" type="button" data-metric="train_exceed" aria-label="Train Exceed 설명">?</button></div>
        <div class="value" id="trainExceed">-</div>
        <div class="hint">정상 학습 구간 threshold 초과율</div>
      </div>
      <div class="panel card">
        <div class="label label-row">Coverage <button class="help" type="button" data-metric="coverage" aria-label="Coverage 설명">?</button></div>
        <div class="value" id="coverageValue">-</div>
        <div class="hint" id="coverageHint">dataset coverage</div>
      </div>
    </section>

    <details class="panel metric-help-panel">
      <summary class="panel-head metric-help-summary">
        <span class="summary-main">
          <h2>지표 설명</h2>
          <span class="hint">필요할 때만 펼쳐 전체 설명을 확인합니다. 상단 지표의 ? 버튼으로도 바로 볼 수 있습니다.</span>
        </span>
        <span class="summary-action" aria-hidden="true"></span>
      </summary>
      <div class="metric-help-grid" id="metricHelpGrid"></div>
    </details>
    <div class="metric-popover" id="metricHelpPopover" role="dialog" aria-live="polite" aria-hidden="true">
      <div class="metric-popover-head">
        <div>
          <div class="group" id="metricPopoverGroup">-</div>
          <h3 id="metricPopoverTitle">-</h3>
        </div>
        <button class="metric-popover-close" type="button" id="metricPopoverClose" aria-label="지표 설명 닫기">x</button>
      </div>
      <p id="metricPopoverDescription"></p>
      <p id="metricPopoverDirection"></p>
      <p class="example" id="metricPopoverExample"></p>
    </div>

    <section class="grid three">
      <div class="panel">
        <div class="panel-head">
          <h2>Run Health</h2>
          <span class="pill" id="throughputInfo">throughput</span>
        </div>
        <div class="detail-box">
          <div class="mini-grid">
            <div class="mini-stat"><div class="k">Recent/min</div><div class="v" id="recentSpeed">-</div></div>
            <div class="mini-stat"><div class="k">Average/min</div><div class="v" id="avgSpeed">-</div></div>
            <div class="mini-stat"><div class="k">Recent ETA</div><div class="v" id="recentEta">-</div></div>
            <div class="mini-stat"><div class="k">Last Mark</div><div class="v" id="lastProgressMark">-</div></div>
          </div>
        </div>
      </div>
      <div class="panel">
        <div class="panel-head">
          <h2>Experiment Compare</h2>
          <span class="pill" id="compareInfo">current vs completed</span>
        </div>
        <div class="table-wrap" style="max-height:220px">
          <table>
            <thead><tr><th>status</th><th>experiment</th><th>best</th><th>mean F1</th><th>Δ best</th><th>FP</th><th>precision</th><th>train exceed</th></tr></thead>
            <tbody id="compareRows"></tbody>
          </table>
        </div>
      </div>
    </section>

    <section class="grid two">
      <div class="panel">
        <div class="panel-head"><h2>Process Detail</h2><span class="pill" id="processInfo">no active process</span></div>
        <div class="detail-box"><ul class="kv-list" id="processList"></ul></div>
      </div>
      <div class="panel">
        <div class="panel-head"><h2>Warnings</h2><span class="pill" id="warningInfo">0 warnings</span></div>
        <div class="detail-box"><ul class="kv-list" id="warningList"></ul></div>
      </div>
    </section>

    <section class="grid two">
      <div class="panel">
        <div class="panel-head">
          <h2>Strategy Ranking</h2>
          <span class="pill" id="rowsInfo">rows -</span>
        </div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>strategy</th><th>tier</th><th>mean F1</th><th>median</th><th>FP</th><th>precision</th><th>train exceed</th><th>pred</th><th>zero</th></tr></thead>
            <tbody id="strategyRows"></tbody>
          </table>
        </div>
      </div>
      <div class="panel">
        <div class="panel-head">
          <h2>Strategy Detail</h2>
          <span class="pill" id="strategyDetailTier">select a row</span>
        </div>
        <div class="detail-box">
          <div class="detail-title" id="strategyDetailTitle">-</div>
          <div class="mini-grid">
            <div class="mini-stat"><div class="k">TP</div><div class="v" id="detailTp">-</div></div>
            <div class="mini-stat"><div class="k">FP</div><div class="v" id="detailFp">-</div></div>
            <div class="mini-stat"><div class="k">FN</div><div class="v" id="detailFn">-</div></div>
            <div class="mini-stat"><div class="k">Precision</div><div class="v" id="detailPrecision">-</div></div>
          </div>
          <div class="breakdown">
            <div>
              <div class="label" style="margin-bottom:8px">Top FP Datasets</div>
              <ul class="kv-list" id="topFpList"></ul>
            </div>
            <div>
              <div class="label" style="margin-bottom:8px">Zero-F1 Samples</div>
              <ul class="kv-list" id="zeroF1List"></ul>
            </div>
          </div>
          <div>
            <div class="label" style="margin-bottom:8px">Family Trouble Spots</div>
            <ul class="kv-list" id="familyTroubleList"></ul>
          </div>
        </div>
      </div>
    </section>

    <section class="grid two">
      <div class="panel">
        <div class="panel-head">
          <h2>Dataset Coverage & Errors</h2>
          <span class="pill" id="errorInfo">0 errors</span>
        </div>
        <div class="detail-box">
          <div class="mini-grid">
            <div class="mini-stat"><div class="k">Done</div><div class="v" id="coverageDone">-</div></div>
            <div class="mini-stat"><div class="k">Errors</div><div class="v" id="coverageErrors">-</div></div>
            <div class="mini-stat"><div class="k">Remaining</div><div class="v" id="coverageRemaining">-</div></div>
            <div class="mini-stat"><div class="k">Coverage</div><div class="v" id="coveragePercent">-</div></div>
          </div>
          <div class="breakdown">
            <div>
              <div class="label" style="margin-bottom:8px">Error Reasons</div>
              <ul class="kv-list" id="errorReasonList"></ul>
            </div>
            <div>
              <div class="label" style="margin-bottom:8px">Error Families</div>
              <ul class="kv-list" id="errorFamilyList"></ul>
            </div>
          </div>
        </div>
      </div>
      <div class="panel">
        <div class="panel-head">
          <h2>Recent Datasets</h2>
          <span class="pill">latest log entries</span>
        </div>
        <div class="timeline" id="timeline"></div>
      </div>
    </section>

    <section class="grid two" style="margin-top:16px">
      <div class="panel">
        <div class="panel-head">
          <h2>GPT Handoff</h2>
          <span class="pill" id="gptExportStatus">ready</span>
        </div>
        <div class="detail-box">
          <div class="toolbar">
            <button class="action-button" type="button" id="copyGptHandoff">Copy Markdown</button>
            <button class="action-button" type="button" id="downloadGptJson">Download JSON</button>
          </div>
        </div>
      </div>
      <details class="panel" id="datasetContext">
        <summary class="panel-head metric-help-summary">
          <span class="summary-main"><h2>Dataset Context</h2><span class="hint" id="datasetContextSummary">database grain and purpose</span></span>
          <span class="summary-action" aria-hidden="true"></span>
        </summary>
        <div class="detail-box context-grid">
          <div class="context-item"><div class="k">DB datasets</div><div class="v" id="contextDatasetCount">-</div></div>
          <div class="context-item"><div class="k">Evaluation</div><div class="v" id="contextEvaluationCount">-</div></div>
          <div class="context-item"><div class="k">TRAIN normal</div><div class="v" id="contextTrainCount">-</div></div>
          <div class="context-item"><div class="k">TEST total / anomaly</div><div class="v" id="contextTestCount">-</div></div>
          <div class="context-item"><div class="k">Candidate index</div><div class="v" id="contextCandidateMeaning">-</div></div>
          <div class="context-item"><div class="k">Production mapping</div><div class="v" id="contextProductionMapping">-</div></div>
        </div>
      </details>
    </section>

    <section class="panel" style="margin-top:16px">
      <div class="panel-head">
        <h2>Experiment Queue</h2>
        <span class="pill" id="queueInfo">waiting</span>
      </div>
      <div class="table-wrap" style="max-height:280px">
        <table>
          <thead><tr><th>status</th><th>experiment</th><th>progress</th><th>rows</th><th>datasets</th><th>note</th></tr></thead>
          <tbody id="queueRows"></tbody>
        </table>
      </div>
    </section>

    <section class="panel" style="margin-top:16px">
      <div class="panel-head">
        <h2>Completed Results</h2>
        <div class="toolbar"><select id="completedFilter" aria-label="완료 실험 필터"><option value="all">전체</option><option value="operational">운영 후보</option><option value="research">연구 후보</option><option value="excluded">배제</option></select><span class="pill" id="completedInfo">0 completed</span></div>
      </div>
      <div class="table-wrap" style="max-height:320px">
        <table>
          <thead><tr><th>experiment</th><th>best config / strategy</th><th>tier</th><th>mean F1</th><th>median</th><th>FP</th><th>precision</th><th>train exceed</th><th>pred</th><th>PR</th><th>oracle</th><th>zero</th><th>datasets</th><th>rows</th><th>elapsed</th><th>finished</th></tr></thead>
          <tbody id="completedRows"></tbody>
        </table>
      </div>
    </section>

    <section class="panel" style="margin-top:16px">
      <div class="panel-head">
        <h2>Live Log</h2>
        <span class="pill" id="updatedAt">-</span>
      </div>
      <pre class="log" id="logBox"></pre>
    </section>
    <div class="footer">
      <span id="runtimeHealth">runtime health loading</span>
      <span>읽기 전용 대시보드입니다. 실험 프로세스와 결과 파일은 수정하지 않습니다.</span>
      <span>자동 새로고침: 5초</span>
    </div>
  </main>
  <script>
    const fmt = (v, d=3) => Number.isFinite(Number(v)) ? Number(v).toFixed(d) : '-';
    const fmtPct = v => `${fmt(v, 2)}%`;
    const fmtRate = v => Number.isFinite(Number(v)) ? `${(Number(v) * 100).toFixed(2)}%` : '-';
    const fmtPrecision = v => Number.isFinite(Number(v)) ? `${(Number(v) * 100).toFixed(1)}%` : '-';
    const fmtSigned = v => Number.isFinite(Number(v)) ? `${Number(v) >= 0 ? '+' : ''}${Number(v).toFixed(3)}` : '-';
    let selectedStrategy = null;
    let metricGlossaryById = {};
    function setText(id, value) { document.getElementById(id).textContent = value; }
    function appendCells(row, values) {
      values.forEach(value => {
        const td = document.createElement('td');
        if (value && typeof value === 'object') {
          td.textContent = value.text ?? '';
          if (value.className) td.className = value.className;
        } else {
          td.textContent = value ?? '';
        }
        row.appendChild(td);
      });
    }
    function renderKvList(id, rows, emptyText, formatter) {
      const list = document.getElementById(id);
      list.innerHTML = '';
      if (!rows || !rows.length) {
        const li = document.createElement('li');
        li.innerHTML = `<span>${emptyText}</span><span>-</span>`;
        list.appendChild(li);
        return;
      }
      rows.forEach(row => {
        const li = document.createElement('li');
        const [left, right] = formatter(row);
        li.innerHTML = `<span class="name">${left}</span><span>${right}</span>`;
        list.appendChild(li);
      });
    }
    function objectRows(obj) {
      return Object.entries(obj || {}).map(([key, value]) => ({key, value}));
    }
    function renderMetricGlossary(glossary) {
      const grid = document.getElementById('metricHelpGrid');
      const byId = Object.fromEntries((glossary || []).map(item => [item.id, item]));
      metricGlossaryById = byId;
      grid.innerHTML = '';
      (glossary || []).forEach(item => {
        const div = document.createElement('div');
        div.className = 'metric-help-card';
        div.innerHTML = `<div class="group">${item.group}</div><h3>${item.label}</h3><p>${item.description}</p><p>${item.direction}</p><p class="example">${item.example}</p>`;
        grid.appendChild(div);
      });
      document.querySelectorAll('[data-metric]').forEach(button => {
        const item = byId[button.dataset.metric];
        if (!item) return;
        button.title = `${item.description} ${item.example}`;
        button.onclick = event => {
          event.preventDefault();
          event.stopPropagation();
          showMetricHelp(button.dataset.metric, button);
        };
      });
    }
    function showMetricHelp(metricId, anchor) {
      const item = metricGlossaryById[metricId];
      const popover = document.getElementById('metricHelpPopover');
      if (!item || !popover) return;
      setText('metricPopoverGroup', item.group || '-');
      setText('metricPopoverTitle', item.label || metricId);
      setText('metricPopoverDescription', item.description || '');
      setText('metricPopoverDirection', item.direction || '');
      setText('metricPopoverExample', item.example || '');
      popover.classList.add('visible');
      popover.setAttribute('aria-hidden', 'false');
      const rect = anchor.getBoundingClientRect();
      const margin = 12;
      const top = Math.min(window.innerHeight - popover.offsetHeight - margin, rect.bottom + 8);
      const left = Math.min(window.innerWidth - popover.offsetWidth - margin, rect.left);
      popover.style.top = `${Math.max(margin, top)}px`;
      popover.style.left = `${Math.max(margin, left)}px`;
    }
    function hideMetricHelp() {
      const popover = document.getElementById('metricHelpPopover');
      if (!popover) return;
      popover.classList.remove('visible');
      popover.setAttribute('aria-hidden', 'true');
    }
    document.getElementById('metricPopoverClose').addEventListener('click', hideMetricHelp);
    document.addEventListener('click', event => {
      const popover = document.getElementById('metricHelpPopover');
      if (!popover || !popover.classList.contains('visible')) return;
      if (popover.contains(event.target) || event.target.closest('[data-metric]')) return;
      hideMetricHelp();
    });
    document.addEventListener('keydown', event => {
      if (event.key === 'Escape') hideMetricHelp();
    });
    function renderStrategyDetail(data) {
      const details = data.strategy_details || {};
      const strategies = data.summary || [];
      if (!selectedStrategy || !details[selectedStrategy]) {
        selectedStrategy = strategies[0] ? strategies[0].strategy : null;
      }
      const detail = selectedStrategy ? details[selectedStrategy] : null;
      setText('strategyDetailTitle', detail ? detail.strategy : '-');
      setText('strategyDetailTier', detail ? detail.operational_tier : 'select a row');
      setText('detailTp', detail ? fmt(detail.mean_tp, 2) : '-');
      setText('detailFp', detail ? fmt(detail.mean_fp, 2) : '-');
      setText('detailFn', detail ? fmt(detail.mean_fn, 2) : '-');
      setText('detailPrecision', detail ? fmtPrecision(detail.alert_precision) : '-');
      renderKvList('topFpList', detail ? detail.top_fp_datasets : [], 'no FP rows', row => [
        row.dataset_name || '-',
        `FP ${row.fp ?? '-'} · pred ${row.predicted_count ?? '-'}`
      ]);
      renderKvList('zeroF1List', detail ? detail.zero_f1_datasets : [], 'no zero-F1 rows', row => [
        row.dataset_name || '-',
        `FP ${row.fp ?? '-'} · FN ${row.fn ?? '-'}`
      ]);
      renderKvList('familyTroubleList', detail ? detail.family_trouble_spots : [], 'no family trouble spots', row => [
        row.family || '-',
        `FP ${row.total_fp ?? '-'} · zero ${row.zero_f1_count ?? '-'} · F1 ${fmt(row.mean_f1)}`
      ]);
    }
    function runtime(seconds) {
      if (!Number.isFinite(seconds) || seconds < 0) return '-';
      const h = Math.floor(seconds / 3600);
      const m = Math.floor((seconds % 3600) / 60);
      return h ? `${h}h ${m}m` : `${m}m`;
    }
    let refreshInFlight = false;
    let refreshDelayMs = 10000;
    let detailCache = null;
    let detailFetchedAt = 0;
    let completedFilter = 'all';
    async function copyGptHandoff() {
      const response = await fetch('/api/gpt-handoff.md', {cache:'no-store'});
      if (!response.ok) throw new Error(`handoff ${response.status}`);
      await navigator.clipboard.writeText(await response.text());
      setText('gptExportStatus', 'copied');
    }
    async function downloadGptJson() {
      const response = await fetch('/api/gpt-handoff.json', {cache:'no-store'});
      if (!response.ok) throw new Error(`handoff ${response.status}`);
      const blob = await response.blob();
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = `timeseries-gpt-handoff-${new Date().toISOString().slice(0,19).replaceAll(':','-')}.json`;
      link.click();
      URL.revokeObjectURL(link.href);
      setText('gptExportStatus', 'downloaded');
    }
    document.getElementById('copyGptHandoff').addEventListener('click', () => copyGptHandoff().catch(error => setText('gptExportStatus', error.message)));
    document.getElementById('downloadGptJson').addEventListener('click', () => downloadGptJson().catch(error => setText('gptExportStatus', error.message)));
    document.getElementById('completedFilter').addEventListener('change', event => { completedFilter = event.target.value; detailFetchedAt = 0; refresh(); });
    async function refresh() {
      if (refreshInFlight) return;
      refreshInFlight = true;
      try {
        const coreRes = await fetch('/api/overview', {cache:'no-store'});
        const core = await coreRes.json();
        if (!detailCache || Date.now() - detailFetchedAt > 60000) {
          const detailRes = await fetch('/api/status', {cache:'no-store'});
          detailCache = await detailRes.json();
          detailFetchedAt = Date.now();
        }
        const data = {...detailCache, ...core};
        refreshDelayMs = data.running ? 5000 : 10000;
        document.getElementById('statusDot').classList.toggle('running', !!data.running);
        setText('statusText', data.running ? '실험 실행 중' : '실험 미실행');
        setText('progressValue', fmtPct(data.progress_percent));
        document.getElementById('progressBar').style.width = `${Math.min(100, data.progress_percent || 0)}%`;
        setText('datasetCount', `${data.datasets_done} / ${data.expected_datasets} datasets`);
        setText('currentDataset', data.current_dataset || '-');
        setText('currentMeta', data.current_meta || 'waiting for log');
        setText('runtime', runtime(data.elapsed_seconds));
        setText('eta', data.eta_seconds ? `예상 남은 시간 ${runtime(data.eta_seconds)}` : 'ETA 계산 중');
        setText('bestStrategy', data.best ? data.best.strategy : '-');
        setText('bestMetric', data.best ? `mean F1 ${fmt(data.best.mean_f1)} / median ${fmt(data.best.median_f1)}` : 'mean F1 -');
        const procCpu = data.process ? `Proc ${fmt(data.process.cpu, 1)}% core` : 'Proc -';
        const sysCpu = data.system && data.system.cpu && data.system.cpu.available ? `Sys ${fmt(data.system.cpu.cpu_percent, 1)}%` : 'Sys -';
        setText('cpu', `${procCpu} / ${sysCpu}`);
        setText('pid', data.process ? `PID ${data.process.pid} · ${data.process.process_count || 1} procs · MEM ${fmt(data.process.mem, 1)}%` : 'PID -');
        const gpu = data.system && data.system.gpu ? data.system.gpu : null;
        setText('gpu', gpu && gpu.available ? `GPU ${fmt(gpu.gpu_percent, 1)}% · ${gpu.source}` : `GPU -${gpu && gpu.note ? ` · ${gpu.note}` : ''}`);
        const op = data.operational_best || data.best || null;
        setText('operationalPick', op ? op.strategy : '-');
        setText('operationalTier', op ? (op.operational_tier || '판정 보류') : '운영 후보 판정 대기');
        setText('meanFp', op ? fmt(op.mean_fp, 2) : '-');
        setText('alertPrecision', op ? fmtPrecision(op.alert_precision) : '-');
        setText('trainExceed', op ? fmtRate(op.mean_train_exceed_rate) : '-');
        const coverage = data.coverage || {};
        setText('coverageValue', coverage.expected_datasets ? fmtPct(coverage.coverage_percent) : '-');
        setText('coverageHint', `${coverage.datasets_done ?? 0} done · ${coverage.error_datasets ?? 0} errors · ${coverage.remaining_or_missing ?? 0} left`);
        setText('rowsInfo', `${data.rows} rows`);
        setText('updatedAt', `updated ${data.updated_at}`);
        const health = data.runtime_health || {};
        const heartbeat = health.heartbeat || {};
        const heartbeatAge = Number.isFinite(heartbeat.age_seconds) ? `${fmt(heartbeat.age_seconds, 0)}s` : '-';
        setText('runtimeHealth', `server pid ${health.server_pid || '-'} · build ${health.code_mtime || '-'} · heartbeat ${heartbeat.status || '-'} (${heartbeatAge})`);
        setText('queueInfo', `${(data.queue || []).length} queued`);
        setText('completedInfo', `${(data.completed_results || []).length} completed`);
        document.getElementById('logBox').textContent = data.log_tail || '';
        renderMetricGlossary(data.metric_glossary || []);
        const throughput = data.throughput || {};
        setText('throughputInfo', throughput.eta_basis ? `ETA basis: ${throughput.eta_basis}` : 'throughput');
        setText('recentSpeed', throughput.recent_datasets_per_minute == null ? '-' : fmt(throughput.recent_datasets_per_minute, 2));
        setText('avgSpeed', throughput.average_datasets_per_minute == null ? '-' : fmt(throughput.average_datasets_per_minute, 2));
        setText('recentEta', throughput.recent_eta_seconds == null ? '-' : runtime(throughput.recent_eta_seconds));
        setText('lastProgressMark', throughput.last_progress ? `${throughput.last_progress.done}/${throughput.last_progress.expected}` : '-');
        const context = data.dataset_context || {};
        setText('contextDatasetCount', context.dataset_count ?? '-');
        setText('contextEvaluationCount', context.evaluation_dataset_count ?? '-');
        setText('contextTrainCount', context.train_normal_instances ?? '-');
        setText('contextTestCount', context.test_total_instances == null ? '-' : `${context.test_total_instances} / ${context.test_anomaly_instances}`);
        setText('contextCandidateMeaning', context.candidate_index_meaning || '-');
        setText('contextProductionMapping', context.production_run_mapping_verified ? 'verified' : 'not verified');
        setText('datasetContextSummary', context.purpose || 'database grain and purpose');

        const tbody = document.getElementById('strategyRows');
        tbody.innerHTML = '';
        (data.summary || []).slice(0, 12).forEach((row, idx) => {
          const tr = document.createElement('tr');
          if (idx === 0) tr.className = 'best';
          tr.classList.add('selectable');
          if (row.strategy === selectedStrategy) tr.classList.add('selected');
          tr.addEventListener('click', () => {
            selectedStrategy = row.strategy;
            renderStrategyDetail(data);
          });
          tr.innerHTML = `<td>${row.strategy}</td><td class="tier">${row.operational_tier || '-'}</td><td>${fmt(row.mean_f1)}</td><td>${fmt(row.median_f1)}</td><td>${fmt(row.mean_fp, 2)}</td><td>${fmtPrecision(row.alert_precision)}</td><td>${fmtRate(row.mean_train_exceed_rate)}</td><td>${fmt(row.mean_predicted_count, 2)}</td><td>${row.zero_f1_count}</td>`;
          tbody.appendChild(tr);
        });
        renderStrategyDetail(data);

        const compareRows = document.getElementById('compareRows');
        compareRows.innerHTML = '';
        (data.experiment_compare || []).forEach(item => {
          const tr = document.createElement('tr');
          if (item.status === 'running') tr.className = 'best';
          const deltaClass = Number(item.delta_vs_best_f1) >= 0 ? 'delta-pos' : 'delta-neg';
          appendCells(tr, [
            item.status,
            item.label,
            item.best_strategy,
            fmt(item.mean_f1),
            {text: fmtSigned(item.delta_vs_best_f1), className: deltaClass},
            fmt(item.mean_fp, 2),
            fmtPrecision(item.alert_precision),
            fmtRate(item.mean_train_exceed_rate),
          ]);
          compareRows.appendChild(tr);
        });
        setText('compareInfo', `${(data.experiment_compare || []).length} rows`);

        const errors = data.error_summary || {};
        setText('errorInfo', `${errors.total_errors || 0} errors`);
        setText('coverageDone', coverage.datasets_done ?? '-');
        setText('coverageErrors', coverage.error_datasets ?? '-');
        setText('coverageRemaining', coverage.remaining_or_missing ?? '-');
        setText('coveragePercent', coverage.expected_datasets ? fmtPct(coverage.coverage_percent) : '-');
        renderKvList('errorReasonList', objectRows(errors.by_reason), 'no errors', row => [row.key, row.value]);
        renderKvList('errorFamilyList', objectRows(errors.by_family), 'no errors', row => [row.key, row.value]);
        const warnings = data.warning_summary || {};
        setText('warningInfo', `${warnings.total_warnings || 0} warnings`);
        renderKvList('warningList', warnings.recent || [], 'no warnings', row => [row.category || 'warning', row.message || '-']);
        const processRows = data.process && data.process.processes ? data.process.processes : [];
        setText('processInfo', processRows.length ? `${processRows.length} active` : 'no active process');
        renderKvList('processList', processRows, 'no active process', row => [`PID ${row.pid} · CPU ${fmt(row.cpu,1)}% · MEM ${fmt(row.mem,1)}%`, row.command || '-']);

        const timeline = document.getElementById('timeline');
        timeline.innerHTML = '';
        (data.recent_datasets || []).forEach(item => {
          const div = document.createElement('div');
          div.className = 'event';
          div.innerHTML = `<div class="time">${item.time}</div><div class="name">${item.name}</div><div class="meta">len ${item.length} · train ${item.train_size}</div>`;
          timeline.appendChild(div);
        });

        const queueRows = document.getElementById('queueRows');
        queueRows.innerHTML = '';
        (data.queue_items || []).forEach(item => {
          const tr = document.createElement('tr');
          if (item.status === 'running') tr.className = 'best';
          const progressText = item.progress_percent == null ? '-' : `${fmt(item.progress_percent, 2)}%`;
          const datasetText = item.datasets_done == null ? '-' : `${item.datasets_done}/${item.expected_datasets ?? '-'}`;
          tr.innerHTML = `<td>${item.status}</td><td>${item.label}</td><td>${progressText}</td><td>${item.rows ?? '-'}</td><td>${datasetText}</td><td>${item.blocked_reason || '-'}</td>`;
          queueRows.appendChild(tr);
        });
        setText('queueInfo', `${(data.queue_items || []).length} active / planned`);

        const completedRows = document.getElementById('completedRows');
        completedRows.innerHTML = '';
        (data.completed_results || []).filter(item => completedFilter === 'all' || (completedFilter === 'excluded' ? item.operational_excluded : item.operational_tier === (completedFilter === 'operational' ? '운영 기본 후보' : '연구 후보'))).forEach(item => {
          const tr = document.createElement('tr');
          appendCells(tr, [
            item.label,
            item.best_strategy,
            {text: item.operational_tier || '-', className: 'tier'},
            fmt(item.mean_f1),
            fmt(item.median_f1),
            fmt(item.mean_fp, 2),
            fmtPrecision(item.alert_precision),
            fmtRate(item.mean_train_exceed_rate),
            fmt(item.mean_predicted_count, 2),
            fmt(item.mean_auc_pr),
            fmt(item.mean_oracle_f1),
            item.zero_f1_count ?? '-',
            item.datasets ?? '-',
            item.rows ?? '-',
            item.elapsed_minutes == null ? '-' : `${fmt(item.elapsed_minutes, 1)}m`,
            {text: item.finished_at || '-', className: 'muted-cell'},
          ]);
          completedRows.appendChild(tr);
        });
      } catch (err) {
        document.getElementById('statusDot').classList.remove('running');
        setText('statusText', `대시보드 오류: ${err.message}`);
      } finally {
        refreshInFlight = false;
      }
    }
    function refreshLoop() {
      refresh().finally(() => setTimeout(refreshLoop, refreshDelayMs));
    }
    refreshLoop();
  </script>
</body>
</html>
"""


def read_csv_rows(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    stat = path.stat()
    key = str(path)
    cached = CSV_CACHE.get(key)
    signature = (stat.st_mtime_ns, stat.st_size)
    if cached and cached.get("signature") == signature:
        return cached["rows"]
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    CSV_CACHE[key] = {"signature": signature, "rows": rows}
    return rows


def read_state():
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text())
    except json.JSONDecodeError:
        return {}


def read_heartbeat():
    try:
        value = json.loads(HEARTBEAT_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    updated = as_float(value.get("updated_unix"))
    if updated is not None:
        value["age_seconds"] = max(0.0, time.time() - updated)
    return value


def read_external_live_run():
    """Read a non-queue experiment status without mutating queue state."""
    try:
        value = json.loads(EXTERNAL_LIVE_RUN_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if value.get("status") in {"running", "completed", "failed"} else {}


def unified_experiment_items(state, external_live_run=None, validation_plan=None):
    external_live_run = external_live_run or {}
    validation_plan = validation_plan or []
    items = {}
    completed = set(state.get("completed") or [])
    running = state.get("running") or {}
    completed_order = {exp_id: index for index, exp_id in enumerate(state.get("completed") or [])}
    for exp_id in state.get("completed") or []:
        info = experiment_info(exp_id)
        items[exp_id] = {"id": exp_id, "label": info.get("label", exp_id), "status": "completed", "completed_order": completed_order[exp_id]}
    for exp_id in state.get("queue") or []:
        info = experiment_info(exp_id)
        items[exp_id] = {"id": exp_id, "label": info.get("label", exp_id), "status": "queued"}
    if running.get("id"):
        items[running["id"]] = {**running, "label": running.get("label") or experiment_info(running["id"]).get("label", running["id"]), "status": "running"}
    for row in validation_plan:
        item = dict(row)
        if external_live_run.get("id") and item.get("runner_id") == external_live_run.get("id"):
            item["status"] = external_live_run.get("status", "running")
            item["progress_percent"] = external_live_run.get("progress_percent")
        items[item["id"]] = item
    if external_live_run.get("id") and external_live_run["id"] not in items:
        items[external_live_run["id"]] = dict(external_live_run)
    order = {"running": 0, "queued": 1, "planned": 2, "blocked": 3, "failed": 4, "completed": 5}
    return sorted(
        items.values(),
        key=lambda row: (
            order.get(row.get("status"), 9),
            row.get("completed_order", 10**9) if row.get("status") == "completed" else row.get("id", ""),
        ),
    )


def dataset_context_snapshot(db_path=DB_PATH):
    db_path = Path(db_path)
    if not db_path.exists():
        return {"available": False, "error": f"missing database: {db_path}"}
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT name, train_normal_count, test_normal_count, test_anomaly_count, test_total_count FROM datasets"
        ).fetchall()
    finally:
        conn.close()
    train_counts = sorted(int(row[1]) for row in rows)
    names = {row[0] for row in rows}
    total = lambda position: int(sum(int(row[position]) for row in rows))
    percentile = lambda value: float(np.percentile(train_counts, value)) if train_counts else 0.0
    return {
        "available": True,
        "source": str(db_path),
        "purpose": "normal-only training and rare-anomaly benchmark evaluation",
        "dataset_count": len(rows),
        "evaluation_dataset_count": len(rows) - len(names & EVALUATION_EXCLUSIONS),
        "excluded_datasets": sorted(names & EVALUATION_EXCLUSIONS),
        "train_normal_instances": total(1),
        "test_normal_instances": total(2),
        "test_anomaly_instances": total(3),
        "test_total_instances": total(4),
        "train_count_distribution": {
            "min": min(train_counts) if train_counts else 0,
            "median": percentile(50),
            "p90": percentile(90),
            "p95": percentile(95),
            "max": max(train_counts) if train_counts else 0,
            "lt20_datasets": sum(value < 20 for value in train_counts),
        },
        "dataset_grain": "benchmark dataset containing multiple TRAIN/TEST time-series instances",
        "candidate_index_meaning": "TEST time-series instance index",
        "candidate_index_is_time_position": False,
        "production_run_mapping_verified": False,
        "missing_operational_fields": ["equipment_id", "recipe_id", "sensor_id", "step_id", "run_id", "timestamp", "user_feedback"],
    }


def gpt_handoff_snapshot(status, dataset_context):
    return {
        "schema_version": "1.0",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "current_status": {
            key: status.get(key)
            for key in ("running", "active_experiment", "active_label", "progress_percent", "datasets_done", "expected_datasets", "coverage")
        },
        "experiment_work_items": status.get("unified_experiments", []),
        "current_metrics": {"best": status.get("best"), "operational_best": status.get("operational_best")},
        "external_validation": status.get("external_validation") or {},
        "data_context": dataset_context,
        "interpretation_rules": {
            "hard_alert_scope": "autonomous",
            "review_scope": "human_assisted_diagnostic_only",
            "retrospective_not_prospective": True,
            "test_metrics_must_not_select_policy": True,
            "end_to_end_strict_train_only_verified": False,
        },
    }


def gpt_handoff_markdown(handoff):
    status = handoff.get("current_status", {})
    context = handoff.get("data_context", {})
    external_validation = handoff.get("external_validation", {})
    items = handoff.get("experiment_work_items", [])
    progress = as_float(status.get("progress_percent")) or 0.0
    done = safe_int(status.get("datasets_done")) or 0
    expected = safe_int(status.get("expected_datasets")) or 0
    active_items = [item for item in items if item.get("status") != "completed"]
    completed_items = [item for item in items if item.get("status") == "completed"]
    shown_completed = completed_items[-10:]
    lines = [
        "# Time-series anomaly project handoff",
        "",
        f"Generated: {handoff.get('generated_at', '-')}",
        f"Current experiment: {status.get('active_experiment', '-')}",
        f"Progress: {done}/{expected} ({progress:.2f}%)",
        "",
        "## Experiment work items",
        "",
    ]
    lines.extend(f"- {item.get('id')}: {item.get('status')}" + (f" - {item.get('blocked_reason')}" if item.get("blocked_reason") else "") for item in active_items)
    if shown_completed:
        lines.extend(["", "## Recent completed experiments", ""])
        lines.extend(f"- {item.get('id')}: completed" for item in shown_completed)
    omitted = len(completed_items) - len(shown_completed)
    if omitted > 0:
        lines.append(f"- {omitted} additional completed experiments omitted from Markdown; retained in JSON")
    if external_validation.get("result_summary"):
        lines.extend(["", "## Current validation summary", "", "```json", json.dumps(external_validation["result_summary"], ensure_ascii=False, indent=2), "```"])
    if external_validation.get("artifacts"):
        lines.extend(["", "## Result artifacts", ""])
        lines.extend(f"- {name}: {path}" for name, path in external_validation["artifacts"].items())
    lines.extend(
        [
            "",
            "## Dataset context",
            "",
            f"- Database datasets: {context.get('dataset_count', '-')}",
            f"- Evaluation datasets: {context.get('evaluation_dataset_count', '-')}",
            f"- Candidate index meaning: {context.get('candidate_index_meaning', '-')}",
            f"- Production run mapping verified: {context.get('production_run_mapping_verified', False)}",
            "- Important caveat: production run mapping is not verified; candidate indices are benchmark TEST time-series instance indices, not time positions.",
            "",
            "## Interpretation rules",
            "",
            "- Hard alerts are autonomous metrics.",
            "- Review metrics are human-assisted diagnostics and must remain separate.",
            "- Results are retrospective unless a prospective validation artifact says otherwise.",
            "- TEST metrics must not select thresholds, budgets, routing, or feature sources.",
            "- The full pipeline is not yet verified as end-to-end strict TRAIN-only.",
        ]
    )
    return "\n".join(lines) + "\n"


def runtime_health_snapshot(heartbeat):
    try:
        code_mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(Path(__file__).stat().st_mtime))
    except OSError:
        code_mtime = "unknown"
    return {
        "server_pid": os.getpid(),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(SERVER_STARTED_UNIX)),
        "uptime_seconds": max(0.0, time.time() - SERVER_STARTED_UNIX),
        "code_mtime": code_mtime,
        "heartbeat": heartbeat,
    }


def metric_glossary_snapshot():
    return [dict(item) for item in METRIC_GLOSSARY]


def count_expected_datasets():
    if not DB_PATH.exists():
        return EXPECTED_DATASETS
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COUNT(*) FROM datasets
            WHERE name NOT IN ('CornellWhaleChallenge', 'Wafer_normal_1')
            """
        )
        value = int(cur.fetchone()[0])
        conn.close()
        return value or EXPECTED_DATASETS
    except Exception:
        return EXPECTED_DATASETS


def experiment_info(exp_id):
    return EXPERIMENTS.get(exp_id) or {}


def expected_for_experiment(info, fallback):
    return int(info.get("expected_datasets") or fallback)


def active_paths(state):
    running = state.get("running") or {}
    history = state.get("history") or []
    latest_history_id = history[-1].get("id") if history and isinstance(history[-1], dict) else None
    completed = state.get("completed") or []
    latest_completed_id = completed[-1] if completed else None
    exp_id = running.get("id") or latest_history_id or latest_completed_id or "rank_threshold_calibration"
    info = experiment_info(exp_id)
    return {
        "id": exp_id,
        "label": info.get("label", exp_id),
        "detail_csv": info.get("detail_csv", DETAIL_PATH),
        "stdout_log": info.get("stdout_log", LOG_PATH),
    }


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def first_number(row, fields):
    for field in fields:
        value = as_float(row.get(field))
        if value is not None:
            return value
    return None


def result_name(row):
    parts = []
    if row.get("config_name"):
        parts.append(row["config_name"])
    if row.get("strategy"):
        parts.append(row["strategy"])
    if row.get("threshold_method"):
        parts.append(row["threshold_method"])
    return " / ".join(parts) or "-"


def strategy_key_for_row(row):
    strategy = row.get("strategy")
    threshold_method = row.get("threshold_method")
    if strategy:
        return f"{row.get('config_name','')}::{strategy}"
    if threshold_method:
        return f"{row.get('config_name','')}::{threshold_method}"
    return row.get("config_name", "")


def parse_family(dataset_name):
    if not dataset_name:
        return "-"
    if "_normal_" in dataset_name:
        return dataset_name.rsplit("_normal_", 1)[0]
    return dataset_name


def safe_mean(rows, field):
    values = [as_float(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def safe_int(value):
    number = as_float(value)
    return int(number) if number is not None else None


def alert_precision_from(tp, fp):
    if tp is None or fp is None:
        return None
    denom = tp + fp
    if denom <= 0:
        return None
    return tp / denom


def operational_tier(row):
    if row.get("operational_excluded"):
        return "배제"
    if row.get("operational_default"):
        return "운영 기본 후보"
    mean_f1 = as_float(row.get("mean_f1"))
    mean_fp = as_float(row.get("mean_fp"))
    train_exceed = as_float(row.get("mean_train_exceed_rate"))
    precision = as_float(row.get("alert_precision"))
    if mean_f1 is None:
        return "판정 보류"
    if (
        mean_f1 >= 0.50
        and mean_fp is not None
        and mean_fp <= 2.0
        and train_exceed is not None
        and train_exceed <= 0.01
        and (precision is None or precision >= 0.50)
    ):
        return "운영 기본 후보"
    if (
        mean_f1 >= 0.50
        and mean_fp is not None
        and mean_fp <= 2.5
        and train_exceed is not None
        and train_exceed <= 0.015
    ):
        return "민감도 후보"
    if mean_f1 >= 0.50:
        return "연구 후보"
    return "보류"


def history_by_id(state):
    result = {}
    for item in state.get("history", []):
        exp_id = item.get("id")
        if exp_id:
            result[exp_id] = item
    return result


def best_summary_row(exp_id, info, history):
    rows = read_csv_rows(info.get("summary_csv"))
    if rows:
        default_selector = OPERATIONAL_DEFAULT_SELECTORS.get(exp_id)
        if default_selector:
            default_rows = [
                row
                for row in rows
                if row.get("selector_name") == default_selector
                or row.get("config_name") == default_selector
                or result_name(row).startswith(default_selector)
            ]
            if default_rows:
                return default_rows[0]
        operational_rows = [row for row in rows if str(row.get("operational_candidate", "")).strip() != "0"]
        if operational_rows:
            rows = operational_rows

        def score(row):
            value = first_number(row, ["mean_f1", "f1_evt", "mean_f1_evt", "f1"])
            return value if value is not None else float("-inf")
        return max(rows, key=score)

    configs = ((history.get(exp_id) or {}).get("summary") or {}).get("configs") or {}
    best_name = None
    best_values = None
    for name, values in configs.items():
        value = as_float(values.get("mean_f1_evt") or values.get("mean_f1") or values.get("f1_evt"))
        if value is None:
            continue
        if best_values is None or value > as_float(best_values.get("mean_f1_evt") or best_values.get("mean_f1") or best_values.get("f1_evt")):
            best_name = name
            best_values = values
    if best_name and best_values:
        return {
            "config_name": best_name,
            "mean_f1_evt": best_values.get("mean_f1_evt"),
            "median_f1_evt": best_values.get("median_f1_evt"),
        }
    return None


def completed_result_snapshot(state):
    completed = [exp_id for exp_id in state.get("completed", []) if exp_id in EXPERIMENTS]
    history = history_by_id(state)
    rows = []
    for exp_id in completed:
        info = EXPERIMENTS[exp_id]
        hist = history.get(exp_id) or {}
        best = best_summary_row(exp_id, info, history)
        summary_meta = hist.get("summary") if isinstance(hist.get("summary"), dict) else {}
        if best:
            mean_f1 = first_number(best, ["mean_f1", "f1_evt", "mean_f1_evt", "f1"])
            median_f1 = first_number(best, ["median_f1", "median_f1_evt"])
            mean_auc_roc = first_number(best, ["mean_auc_roc", "auc_roc"])
            mean_auc_pr = first_number(best, ["mean_auc_pr", "auc_pr"])
            mean_oracle_f1 = first_number(best, ["mean_oracle_f1", "oracle_f1"])
            zero_f1_count = first_number(best, ["zero_f1_count"])
            mean_predicted_count = first_number(best, ["mean_predicted_count", "predicted_count"])
            mean_tp = first_number(best, ["mean_tp", "tp"])
            mean_fp = first_number(best, ["mean_fp", "fp"])
            mean_fn = first_number(best, ["mean_fn", "fn"])
            mean_train_exceed_rate = first_number(best, ["mean_train_exceed_rate", "train_exceed_rate"])
            alert_precision = first_number(best, ["alert_precision"])
            if alert_precision is None:
                alert_precision = alert_precision_from(mean_tp, mean_fp)
            datasets = first_number(best, ["num_datasets"]) or summary_meta.get("datasets")
            best_strategy = result_name(best)
            history_configs = (summary_meta.get("configs") or {}) if isinstance(summary_meta, dict) else {}
            history_values = history_configs.get(best_strategy) or {}
            if median_f1 is None:
                median_f1 = first_number(history_values, ["median_f1_evt", "median_f1"])
        else:
            mean_f1 = median_f1 = mean_auc_roc = mean_auc_pr = mean_oracle_f1 = zero_f1_count = None
            mean_predicted_count = mean_tp = mean_fp = mean_fn = mean_train_exceed_rate = alert_precision = None
            datasets = summary_meta.get("datasets")
            best_strategy = "-"
        detail_row_count = summary_meta.get("rows")
        if detail_row_count is None:
            detail_path = info.get("detail_csv")
            try:
                detail_row_count = sum(1 for _ in detail_path.open()) - 1 if detail_path and detail_path.exists() else 0
            except OSError:
                detail_row_count = 0
        ops_row = {
            "mean_f1": mean_f1,
            "mean_fp": mean_fp,
            "mean_train_exceed_rate": mean_train_exceed_rate,
            "alert_precision": alert_precision,
            "operational_excluded": exp_id in EXCLUDED_OPERATIONAL_EXPERIMENTS,
            "operational_default": bool(
                OPERATIONAL_DEFAULT_SELECTORS.get(exp_id)
                and str(best_strategy).startswith(OPERATIONAL_DEFAULT_SELECTORS[exp_id])
            ),
        }
        rows.append(
            {
                "id": exp_id,
                "label": info.get("label", exp_id),
                "best_strategy": best_strategy,
                "mean_f1": mean_f1,
                "median_f1": median_f1,
                "mean_auc_roc": mean_auc_roc,
                "mean_auc_pr": mean_auc_pr,
                "mean_oracle_f1": mean_oracle_f1,
                "zero_f1_count": int(zero_f1_count) if zero_f1_count is not None else None,
                "mean_predicted_count": mean_predicted_count,
                "mean_tp": mean_tp,
                "mean_fp": mean_fp,
                "mean_fn": mean_fn,
                "alert_precision": alert_precision,
                "mean_train_exceed_rate": mean_train_exceed_rate,
                "operational_tier": operational_tier(ops_row),
                "operational_excluded": exp_id in EXCLUDED_OPERATIONAL_EXPERIMENTS,
                "operational_default": bool(
                    OPERATIONAL_DEFAULT_SELECTORS.get(exp_id)
                    and str(best_strategy).startswith(OPERATIONAL_DEFAULT_SELECTORS[exp_id])
                ),
                "exclusion_reason": EXCLUDED_OPERATIONAL_EXPERIMENTS.get(exp_id),
                "datasets": int(datasets) if datasets is not None else None,
                "rows": detail_row_count,
                "elapsed_minutes": hist.get("elapsed_minutes"),
                "finished_at": hist.get("finished_at"),
                "summary_csv": str(info.get("summary_csv")),
            }
        )
    return sorted(rows, key=lambda row: row.get("finished_at") or "", reverse=True)


def parse_log_tail(path):
    if not path.exists():
        return "", [], []
    lines = path.read_text(errors="replace").replace("\x00", "").splitlines()
    tail = "\n".join(lines[-80:])
    pattern = re.compile(
        r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),\d+ \[INFO\] Dataset: ([^|]+) \| Length: (\d+) \| Train size: (\d+)"
    )
    recent = []
    for line in lines[-300:]:
        match = pattern.match(line)
        if match:
            recent.append(
                {
                    "time": match.group(2),
                    "name": match.group(3).strip(),
                    "length": int(match.group(4)),
                    "train_size": int(match.group(5)),
                }
            )
    return tail, recent[-12:], lines


def recent_from_rows(rows, limit=12):
    seen = set()
    recent = []
    for row in reversed(rows):
        name = row.get("dataset_name")
        if not name or name in seen:
            continue
        seen.add(name)
        length = safe_int(row.get("sequence_length") or row.get("metadata_len") or row.get("actual_len_median"))
        train_size = safe_int(row.get("train_count") or row.get("train_score_count"))
        recent.append(
            {
                "time": "csv",
                "name": name,
                "length": length if length is not None else "-",
                "train_size": train_size if train_size is not None else "-",
            }
        )
        if len(recent) >= limit:
            break
    return list(reversed(recent))


def classify_error_reason(message):
    text = message.lower()
    if "inhomogeneous shape" in text or "setting an array element with a sequence" in text:
        return "variable_length_inhomogeneous_shape"
    if "nan" in text or "inf" in text:
        return "nan_or_inf"
    if "memory" in text or "out of memory" in text:
        return "memory"
    return "other_exception"


def parse_error_events(lines):
    pattern = re.compile(r"Error evaluating (?:dataset )?([^:]+): (.*)$")
    events = []
    for line in lines:
        match = pattern.search(line)
        if not match:
            continue
        dataset_name = match.group(1).strip()
        message = match.group(2).strip()
        events.append(
            {
                "dataset_name": dataset_name,
                "family": parse_family(dataset_name),
                "reason": classify_error_reason(message),
                "message": message,
            }
        )
    return events


def warning_summary(lines):
    events = []
    for line in lines:
        match = re.search(r"([A-Za-z]+Warning):\s*(.*)$", line)
        if not match:
            continue
        events.append({"category": match.group(1), "message": match.group(2).strip()[:180]})
    unique = []
    seen = set()
    for event in reversed(events):
        key = (event["category"], event["message"])
        if key not in seen:
            seen.add(key)
            unique.append(event)
    return {"total_warnings": len(events), "recent": list(reversed(unique[:8]))}


def error_summary(errors):
    by_reason = defaultdict(int)
    by_family = defaultdict(int)
    for item in errors:
        by_reason[item.get("reason") or "other_exception"] += 1
        by_family[item.get("family") or "-"] += 1
    return {
        "total_errors": len(errors),
        "error_datasets": len({item.get("dataset_name") for item in errors if item.get("dataset_name")}),
        "by_reason": dict(sorted(by_reason.items(), key=lambda item: (-item[1], item[0]))),
        "by_family": dict(sorted(by_family.items(), key=lambda item: (-item[1], item[0]))[:12]),
        "recent": errors[-8:],
    }


def coverage_snapshot(rows, expected, errors):
    datasets_done = len({row.get("dataset_name") for row in rows if row.get("dataset_name")})
    error_datasets = len({item.get("dataset_name") for item in errors if item.get("dataset_name")})
    remaining = max(0, int(expected or 0) - datasets_done - error_datasets)
    return {
        "datasets_done": datasets_done,
        "expected_datasets": int(expected or 0),
        "error_datasets": error_datasets,
        "remaining_or_missing": remaining,
        "coverage_percent": (datasets_done / expected * 100) if expected else 0.0,
    }


def coverage_snapshot_from_counts(done, expected, errors):
    error_datasets = len({item.get("dataset_name") for item in errors if item.get("dataset_name")})
    done = int(done or 0)
    expected = int(expected or 0)
    remaining = max(0, expected - done - error_datasets)
    return {
        "datasets_done": done,
        "expected_datasets": expected,
        "error_datasets": error_datasets,
        "remaining_or_missing": remaining,
        "coverage_percent": (done / expected * 100) if expected else 0.0,
    }


def process_info(pid):
    if not pid:
        return None
    try:
        out = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "pid=,%cpu=,%mem=,etime="],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (BlockingIOError, OSError, subprocess.CalledProcessError):
        return None
    parts = out.split(None, 3)
    if len(parts) < 4:
        return None
    return {
        "pid": int(parts[0]),
        "cpu": float(parts[1]),
        "mem": float(parts[2]),
        "etime": parts[3],
    }


def parse_ps_process_rows(ps_output):
    rows = []
    for line in (ps_output or "").splitlines():
        parts = line.split(None, 4)
        if len(parts) < 4:
            continue
        try:
            rows.append(
                {
                    "pid": int(parts[0]),
                    "ppid": int(parts[1]),
                    "cpu": float(parts[2]),
                    "mem": float(parts[3]),
                    "command": parts[4] if len(parts) > 4 else "",
                }
            )
        except ValueError:
            continue
    return rows


def process_tree_snapshot(pid, ps_output=None):
    if not pid:
        return None
    try:
        root_pid = int(pid)
    except (TypeError, ValueError):
        return None
    if ps_output is None:
        try:
            ps_output = subprocess.check_output(
                ["ps", "-axo", "pid=,ppid=,%cpu=,%mem=,comm="],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except (BlockingIOError, OSError, subprocess.CalledProcessError):
            return process_info(root_pid)
    rows = parse_ps_process_rows(ps_output)
    by_pid = {row["pid"]: row for row in rows}
    if root_pid not in by_pid:
        return None
    children = defaultdict(list)
    for row in rows:
        children[row["ppid"]].append(row["pid"])
    stack = list(children.get(root_pid, []))
    tree_pids = {root_pid}
    while stack:
        child = stack.pop()
        if child in tree_pids:
            continue
        tree_pids.add(child)
        stack.extend(children.get(child, []))
    tree_rows = [by_pid[item] for item in tree_pids if item in by_pid]
    root_info = process_info(root_pid) if ps_output is None else None
    return {
        "pid": root_pid,
        "cpu": round(sum(row["cpu"] for row in tree_rows), 3),
        "mem": round(sum(row["mem"] for row in tree_rows), 3),
        "etime": (root_info or {}).get("etime"),
        "process_count": len(tree_rows),
        "child_pids": sorted(pid for pid in tree_pids if pid != root_pid),
        "processes": sorted(
            [{"pid": row["pid"], "cpu": row["cpu"], "mem": row["mem"], "command": row["command"]} for row in tree_rows],
            key=lambda row: row["cpu"],
            reverse=True,
        ),
    }


def system_cpu_snapshot(top_output=None):
    if top_output is None:
        try:
            top_output = subprocess.check_output(
                ["top", "-l", "1", "-n", "0"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
        except (BlockingIOError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return {"available": False, "source": "top", "note": "top command temporarily unavailable"}
    match = re.search(
        r"CPU usage:\s*([0-9.]+)%\s*user,\s*([0-9.]+)%\s*sys,\s*([0-9.]+)%\s*idle",
        top_output,
    )
    if not match:
        return {"available": False, "source": "top", "note": "CPU usage line not found"}
    user = float(match.group(1))
    system = float(match.group(2))
    idle = float(match.group(3))
    return {
        "available": True,
        "source": "top",
        "cpu_percent": round(user + system, 3),
        "cpu_user_percent": user,
        "cpu_system_percent": system,
        "cpu_idle_percent": idle,
    }


def parse_nvidia_gpu_output(output):
    lines = [line.strip() for line in (output or "").splitlines() if line.strip()]
    values = []
    for line in lines:
        parts = [part.strip() for part in line.split(",")]
        try:
            values.append(
                {
                    "gpu_percent": float(parts[0]),
                    "memory_used_mb": float(parts[1]) if len(parts) > 1 else None,
                    "memory_total_mb": float(parts[2]) if len(parts) > 2 else None,
                }
            )
        except (ValueError, IndexError):
            continue
    if not values:
        return None
    return {
        "available": True,
        "source": "nvidia-smi",
        "gpu_percent": max(item["gpu_percent"] for item in values),
        "devices": values,
    }


def parse_powermetrics_gpu_output(output):
    matches = re.findall(r"GPU[^:\n]*(?:Active|active)[^:\n]*:\s*([0-9.]+)%", output or "")
    if not matches:
        matches = re.findall(r"GPU[^%\n]*?([0-9.]+)%", output or "")
    if not matches:
        return None
    return {
        "available": True,
        "source": "powermetrics",
        "gpu_percent": max(float(value) for value in matches),
    }


def gpu_snapshot(nvidia_output=None, powermetrics_output=None, errors=None):
    errors = list(errors or [])
    if nvidia_output is not None:
        parsed = parse_nvidia_gpu_output(nvidia_output)
        if parsed:
            return parsed
    elif shutil_which("nvidia-smi"):
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            parsed = parse_nvidia_gpu_output(output)
            if parsed:
                return parsed
        except subprocess.TimeoutExpired:
            errors.append("nvidia-smi timed out")
        except (BlockingIOError, OSError, subprocess.CalledProcessError, FileNotFoundError):
            errors.append("nvidia-smi unavailable")
    if powermetrics_output is not None:
        parsed = parse_powermetrics_gpu_output(powermetrics_output)
        if parsed:
            return parsed
    else:
        cached = RESOURCE_CACHE["gpu"]
        if cached["value"] is not None and cached["expires_at"] > time.time():
            return cached["value"]
        if shutil_which("powermetrics"):
            try:
                output = subprocess.check_output(
                    ["powermetrics", "--samplers", "gpu_power", "-n", "1", "-i", "200"],
                    text=True,
                    stderr=subprocess.STDOUT,
                    timeout=3,
                )
                parsed = parse_powermetrics_gpu_output(output)
                if parsed:
                    RESOURCE_CACHE["gpu"] = {"expires_at": time.time() + 10, "value": parsed}
                    return parsed
            except subprocess.TimeoutExpired:
                errors.append("powermetrics timed out")
            except (BlockingIOError, OSError, subprocess.CalledProcessError):
                errors.append("powermetrics 권한 필요 또는 GPU sampler 사용 불가")
            except FileNotFoundError:
                errors.append("powermetrics unavailable")
    note = "; ".join(errors[-2:]) if errors else "GPU usage source unavailable"
    unavailable = {"available": False, "source": None, "gpu_percent": None, "note": note}
    RESOURCE_CACHE["gpu"] = {"expires_at": time.time() + 30, "value": unavailable}
    return unavailable


def shutil_which(command):
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        path = Path(directory) / command
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
    return None


def system_resource_snapshot():
    return {
        "cpu": system_cpu_snapshot(),
        "gpu": gpu_snapshot(),
    }


def elapsed_from_started(started_at):
    if not started_at:
        return None
    try:
        start = time.mktime(time.strptime(started_at, "%Y-%m-%d %H:%M:%S"))
        return max(0, time.time() - start)
    except ValueError:
        return None


def summarize_by_config(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[strategy_key_for_row(row)].append(row)
    summary = []
    for key, subset in groups.items():
        strategy = key
        def mean(field):
            vals = [float(r[field]) for r in subset if r.get(field) not in ("", None)]
            return sum(vals) / len(vals) if vals else 0.0
        f1s = sorted(float(r["f1"]) for r in subset if r.get("f1") not in ("", None))
        median = f1s[len(f1s) // 2] if f1s else 0.0
        mean_tp = safe_mean(subset, "tp")
        mean_fp = safe_mean(subset, "fp")
        mean_fn = safe_mean(subset, "fn")
        alert_precision = alert_precision_from(mean_tp, mean_fp)
        row = {
            "strategy": strategy,
            "mean_f1": mean("f1"),
            "median_f1": median,
            "mean_auc_roc": mean("auc_roc"),
            "mean_auc_pr": mean("auc_pr"),
            "zero_f1_count": sum(1 for r in subset if float(r.get("f1") or 0) == 0),
            "mean_predicted_count": safe_mean(subset, "predicted_count"),
            "mean_tp": mean_tp,
            "mean_fp": mean_fp,
            "mean_fn": mean_fn,
            "alert_precision": alert_precision,
            "mean_train_exceed_rate": safe_mean(subset, "train_exceed_rate"),
        }
        row["operational_tier"] = operational_tier(row)
        summary.append(row)
    return sorted(summary, key=lambda r: r["mean_f1"], reverse=True)


def family_trouble_spots(rows, strategy, limit=8):
    groups = defaultdict(
        lambda: {
            "datasets": set(),
            "f1s": [],
            "total_tp": 0.0,
            "total_fp": 0.0,
            "total_fn": 0.0,
            "total_predicted": 0.0,
            "zero_f1_count": 0,
        }
    )
    for row in rows:
        if strategy_key_for_row(row) != strategy:
            continue
        family = parse_family(row.get("dataset_name"))
        bucket = groups[family]
        if row.get("dataset_name"):
            bucket["datasets"].add(row["dataset_name"])
        f1 = as_float(row.get("f1"))
        if f1 is not None:
            bucket["f1s"].append(f1)
            if f1 == 0:
                bucket["zero_f1_count"] += 1
        bucket["total_tp"] += as_float(row.get("tp")) or 0.0
        bucket["total_fp"] += as_float(row.get("fp")) or 0.0
        bucket["total_fn"] += as_float(row.get("fn")) or 0.0
        bucket["total_predicted"] += as_float(row.get("predicted_count")) or 0.0
    spots = []
    for family, bucket in groups.items():
        total_tp = bucket["total_tp"]
        total_fp = bucket["total_fp"]
        total_fn = bucket["total_fn"]
        mean_f1 = sum(bucket["f1s"]) / len(bucket["f1s"]) if bucket["f1s"] else None
        spots.append(
            {
                "family": family,
                "datasets": len(bucket["datasets"]),
                "mean_f1": mean_f1,
                "zero_f1_count": bucket["zero_f1_count"],
                "total_tp": int(total_tp) if total_tp.is_integer() else total_tp,
                "total_fp": int(total_fp) if total_fp.is_integer() else total_fp,
                "total_fn": int(total_fn) if total_fn.is_integer() else total_fn,
                "total_predicted": int(bucket["total_predicted"]) if bucket["total_predicted"].is_integer() else bucket["total_predicted"],
                "alert_precision": alert_precision_from(total_tp, total_fp),
            }
        )
    return sorted(
        spots,
        key=lambda row: (
            -(as_float(row.get("total_fp")) or 0),
            -(as_float(row.get("zero_f1_count")) or 0),
            -(as_float(row.get("total_fn")) or 0),
            as_float(row.get("mean_f1")) if row.get("mean_f1") is not None else 999,
            row.get("family") or "",
        ),
    )[:limit]


def strategy_detail_snapshot(rows, summary, limit=12):
    result = {}
    for item in summary[:limit]:
        strategy = item["strategy"]
        subset = []
        for row in rows:
            if strategy_key_for_row(row) == strategy:
                subset.append(row)
        top_fp = sorted(
            subset,
            key=lambda row: (as_float(row.get("fp")) or 0, as_float(row.get("predicted_count")) or 0),
            reverse=True,
        )[:10]
        zero_f1 = [row for row in subset if (as_float(row.get("f1")) or 0) == 0][:10]
        result[strategy] = {
            "strategy": strategy,
            "operational_tier": item.get("operational_tier"),
            "mean_tp": item.get("mean_tp"),
            "mean_fp": item.get("mean_fp"),
            "mean_fn": item.get("mean_fn"),
            "alert_precision": item.get("alert_precision"),
            "mean_train_exceed_rate": item.get("mean_train_exceed_rate"),
            "top_fp_datasets": [
                {
                    "dataset_name": row.get("dataset_name"),
                    "family": parse_family(row.get("dataset_name")),
                    "f1": as_float(row.get("f1")),
                    "tp": safe_int(row.get("tp")),
                    "fp": safe_int(row.get("fp")),
                    "fn": safe_int(row.get("fn")),
                    "predicted_count": safe_int(row.get("predicted_count")),
                }
                for row in top_fp
            ],
            "zero_f1_datasets": [
                {
                    "dataset_name": row.get("dataset_name"),
                    "family": parse_family(row.get("dataset_name")),
                    "fp": safe_int(row.get("fp")),
                    "fn": safe_int(row.get("fn")),
                    "predicted_count": safe_int(row.get("predicted_count")),
                }
                for row in zero_f1
            ],
            "family_trouble_spots": family_trouble_spots(rows, strategy),
        }
    return result


def parse_progress_events(lines):
    canonical_pattern = re.compile(
        r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),\d+ \[INFO\] Progress: \[\s*(\d+)/(\d+)\] rows=(\d+)"
    )
    compact_pattern = re.compile(r"\bprogress\s+(\d+)/(\d+)\b(?:\s+last=([^\s]+))?", re.IGNORECASE)
    plain_pattern = re.compile(r"^Progress:\s*\[\s*(\d+)/(\d+)\]\s*rows=(\d+)", re.IGNORECASE)
    events = []
    for line in lines:
        match = canonical_pattern.match(line)
        if match:
            timestamp = time.mktime(time.strptime(f"{match.group(1)} {match.group(2)}", "%Y-%m-%d %H:%M:%S"))
            events.append(
                {
                    "timestamp": timestamp,
                    "time": match.group(2),
                    "done": int(match.group(3)),
                    "expected": int(match.group(4)),
                    "rows": int(match.group(5)),
                    "source": "canonical",
                }
            )
            continue
        match = plain_pattern.match(line)
        if match:
            events.append(
                {
                    "timestamp": None,
                    "time": "log",
                    "done": int(match.group(1)),
                    "expected": int(match.group(2)),
                    "rows": int(match.group(3)),
                    "source": "plain",
                }
            )
            continue
        match = compact_pattern.search(line)
        if match:
            events.append(
                {
                    "timestamp": None,
                    "time": "log",
                    "done": int(match.group(1)),
                    "expected": int(match.group(2)),
                    "rows": None,
                    "name": match.group(3) or "",
                    "source": "compact",
                }
            )
    return events


def throughput_snapshot(events, done, expected, elapsed):
    average_speed = None
    if elapsed and elapsed > 0 and done is not None:
        average_speed = done / (elapsed / 60)
    recent_speed = None
    recent_eta = None
    if len(events) >= 2:
        previous = events[-2]
        latest = events[-1]
        if latest.get("timestamp") is not None and previous.get("timestamp") is not None:
            seconds = latest["timestamp"] - previous["timestamp"]
            delta_done = latest["done"] - previous["done"]
            if seconds > 0 and delta_done >= 0:
                recent_speed = delta_done / (seconds / 60)
    remaining = max(0, int(expected or 0) - int(done or 0))
    if recent_speed and recent_speed > 0:
        recent_eta = remaining / recent_speed * 60
        eta_basis = "recent"
    elif average_speed and average_speed > 0:
        recent_eta = remaining / average_speed * 60
        eta_basis = "average"
    else:
        eta_basis = "waiting"
    latest = events[-1] if events else None
    return {
        "recent_datasets_per_minute": recent_speed,
        "average_datasets_per_minute": average_speed,
        "recent_eta_seconds": recent_eta,
        "eta_basis": eta_basis,
        "last_progress": latest,
        "progress_log_gap": (done - latest["done"]) if latest else None,
    }


def experiment_compare_snapshot(active_id, active_label, active_summary, completed_results, is_running):
    rows = []
    completed_ids = {item.get("id") for item in completed_results}
    if active_summary and (is_running or active_id not in completed_ids):
        active = active_summary[0]
        rows.append(
            {
                "id": active_id,
                "label": active_label,
                "status": "running" if is_running else "selected",
                "best_strategy": active.get("strategy"),
                "mean_f1": active.get("mean_f1"),
                "median_f1": active.get("median_f1"),
                "mean_fp": active.get("mean_fp"),
                "alert_precision": active.get("alert_precision"),
                "mean_train_exceed_rate": active.get("mean_train_exceed_rate"),
                "operational_tier": active.get("operational_tier"),
                "operational_excluded": active.get("operational_excluded"),
            }
        )
    for item in sorted(completed_results, key=lambda row: as_float(row.get("mean_f1")) or float("-inf"), reverse=True):
        rows.append(
            {
                "id": item.get("id"),
                "label": item.get("label"),
                "status": "completed",
                "best_strategy": item.get("best_strategy"),
                "mean_f1": item.get("mean_f1"),
                "median_f1": item.get("median_f1"),
                "mean_fp": item.get("mean_fp"),
                "alert_precision": item.get("alert_precision"),
                "mean_train_exceed_rate": item.get("mean_train_exceed_rate"),
                "operational_tier": item.get("operational_tier"),
                "operational_excluded": item.get("operational_excluded"),
            }
        )
    best_f1 = max(
        [
            as_float(row.get("mean_f1"))
            for row in rows
            if not row.get("operational_excluded") and as_float(row.get("mean_f1")) is not None
        ]
        or [None]
    )
    for row in rows:
        mean_f1 = as_float(row.get("mean_f1"))
        row["delta_vs_best_f1"] = round(mean_f1 - best_f1, 6) if mean_f1 is not None and best_f1 is not None else None
    return rows


def build_status(include_details=True):
    state = read_state()
    running = state.get("running") or {}
    external_live_run = read_external_live_run()
    if external_live_run.get("status") == "running":
        running = external_live_run
    heartbeat = read_heartbeat()
    paths = active_paths(state)
    if external_live_run:
        paths = {
            "id": external_live_run.get("id", "external_validation"),
            "label": external_live_run.get("label", "External validation"),
            "detail_csv": Path(external_live_run.get("detail_csv", DATA_DIR / "missing_external_results.csv")),
            "stdout_log": Path(external_live_run.get("stdout_log", LOG_PATH)),
        }
    rows = read_csv_rows(paths["detail_csv"])
    default_expected = count_expected_datasets()
    expected = expected_for_experiment(experiment_info(paths["id"]), default_expected)
    summary = summarize_by_config(rows)
    if paths["id"] in EXCLUDED_OPERATIONAL_EXPERIMENTS:
        for row in summary:
            row["operational_excluded"] = True
            row["exclusion_reason"] = EXCLUDED_OPERATIONAL_EXPERIMENTS[paths["id"]]
            row["operational_tier"] = "배제"
    log_tail, recent, log_lines = parse_log_tail(paths["stdout_log"])
    errors = parse_error_events(log_lines)
    warnings = warning_summary(log_lines)
    progress_events = parse_progress_events(log_lines)
    csv_done = len({r.get("dataset_name") for r in rows if r.get("dataset_name")})
    progress_latest = progress_events[-1] if progress_events else None
    done = csv_done
    heartbeat_matches = bool(running and heartbeat.get("experiment_id") == running.get("id") and heartbeat.get("status") == "running")
    if heartbeat_matches and heartbeat.get("done") is not None:
        done = safe_int(heartbeat.get("done")) or done
        expected = safe_int(heartbeat.get("expected")) or expected
    elif running and progress_latest:
        done = progress_latest["done"]
        expected = progress_latest.get("expected") or expected
    elif progress_latest and progress_latest.get("done", 0) > done:
        done = progress_latest["done"]
        expected = progress_latest.get("expected") or expected
    progress = (done / expected * 100) if expected else 0.0
    if not recent:
        recent = recent_from_rows(rows)
    if not recent and progress_latest:
        recent = [
            {
                "time": progress_latest.get("time") or "log",
                "name": progress_latest.get("name")
                or f"progress {progress_latest.get('done')}/{progress_latest.get('expected')}",
                "length": "-",
                "train_size": "-",
            }
        ]
    pid = running.get("child_pid") or running.get("pid")
    process_snapshot = process_tree_snapshot(pid)
    is_running = bool(process_snapshot)
    current = recent[-1] if recent and is_running else {}
    if heartbeat_matches and heartbeat.get("current_dataset"):
        current = {"name": heartbeat.get("current_dataset"), "length": "-", "train_size": "-"}
    elapsed = elapsed_from_started(running.get("started_at")) if is_running else None
    eta = None
    if is_running and elapsed and done:
        eta = elapsed / done * max(0, expected - done)
    operational_candidates = [
        row for row in summary if row.get("operational_tier") == "운영 기본 후보" and not row.get("operational_excluded")
    ]
    operational_best = None
    if operational_candidates:
        operational_best = sorted(
            operational_candidates,
            key=lambda row: (
                row.get("mean_fp") if row.get("mean_fp") is not None else 999999,
                -(row.get("mean_f1") or 0),
            ),
        )[0]
    completed_results = completed_result_snapshot(state) if include_details else []
    if include_details:
        primary_completed = next(
            (
                row
                for row in completed_results
                if row.get("id") == PRIMARY_OPERATIONAL_EXPERIMENT and not row.get("operational_excluded")
            ),
            None,
        )
        if primary_completed:
            operational_best = primary_completed
        if not operational_best:
            completed_operational_candidates = [
                row
                for row in completed_results
                if row.get("operational_tier") == "운영 기본 후보" and not row.get("operational_excluded")
            ]
            if completed_operational_candidates:
                operational_best = sorted(
                    completed_operational_candidates,
                    key=lambda row: (
                        row.get("mean_fp") if row.get("mean_fp") is not None else 999999,
                        -(row.get("mean_f1") or 0),
                    ),
                )[0]
    completed_ids = set(state.get("completed", []))
    running_id = running.get("id")
    pending_queue = [
        exp_id
        for exp_id in state.get("queue", [])
        if exp_id and exp_id != running_id and exp_id not in completed_ids
    ]
    queue_ids = [running_id, *pending_queue]
    experiment_snapshot = experiment_queue_snapshot(state, default_expected, experiment_ids=queue_ids, heartbeat=heartbeat)
    external_for_items = dict(external_live_run)
    if external_for_items:
        external_for_items["progress_percent"] = progress
        external_for_items["datasets_done"] = done
        external_for_items["expected_datasets"] = expected
        external_for_items["rows"] = len(rows)
    unified = unified_experiment_items(state, external_for_items, VALIDATION_PLAN)
    queue_items = [row for row in unified if row.get("status") in {"running", "queued", "planned", "blocked", "failed"}]
    dataset_context = dataset_context_snapshot()
    payload = {
        "running": is_running,
        "active_experiment": paths["id"],
        "active_label": paths["label"],
        "running_state": running,
        "process": process_snapshot,
        "system": system_resource_snapshot(),
        "queue": pending_queue,
        "queue_items": queue_items,
        "unified_experiments": unified,
        "dataset_context": dataset_context,
        "catalog_count": len(EXPERIMENTS),
        "datasets_done": done,
        "expected_datasets": expected,
        "progress_percent": progress,
        "rows": len(rows),
        "current_dataset": current.get("name"),
        "current_meta": (
            "progress log fallback"
            if current and current.get("name", "").startswith("progress ")
            else f"length {current.get('length')} · train {current.get('train_size')}"
            if current
            else ""
        ),
        "recent_datasets": recent,
        "summary": summary,
        "coverage": coverage_snapshot_from_counts(done, expected, errors),
        "error_summary": error_summary(errors),
        "warning_summary": warnings,
        "throughput": throughput_snapshot(progress_events, done, expected, elapsed),
        "best": summary[0] if summary else None,
        "runtime_health": runtime_health_snapshot(heartbeat),
        "external_validation": external_live_run,
        "log_tail": log_tail,
        "elapsed_seconds": elapsed,
        "eta_seconds": eta,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if include_details:
        payload.update(
            {
                "metric_glossary": metric_glossary_snapshot(),
                "completed_results": completed_results,
                "experiment_compare": experiment_compare_snapshot(
                    paths["id"], paths["label"], summary, completed_results, is_running
                ),
                "strategy_details": strategy_detail_snapshot(rows, summary),
                "operational_best": operational_best,
            }
        )
    return payload


def experiment_queue_snapshot(state, expected, experiment_ids=None, heartbeat=None):
    completed = set(state.get("completed", []))
    history = history_by_id(state)
    queued = list(state.get("queue", []))
    running = state.get("running") or {}
    running_id = running.get("id")
    ordered = []
    source_ids = experiment_ids if experiment_ids is not None else [running_id] + queued + list(EXPERIMENTS)
    for exp_id in source_ids:
        if exp_id and exp_id not in ordered:
            ordered.append(exp_id)
    rows = []
    for exp_id in ordered:
        info = experiment_info(exp_id)
        if not info:
            continue
        detail_rows = read_csv_rows(info["detail_csv"])
        item_expected = expected_for_experiment(info, expected)
        done = len({r.get("dataset_name") for r in detail_rows if r.get("dataset_name")})
        row_count = len(detail_rows)
        hist_summary = ((history.get(exp_id) or {}).get("summary") or {})
        if exp_id in completed and done == 0 and hist_summary.get("datasets"):
            done = safe_int(hist_summary.get("datasets")) or done
        if exp_id in completed and row_count == 0 and hist_summary.get("rows"):
            row_count = safe_int(hist_summary.get("rows")) or row_count
        if info.get("stdout_log") and Path(info["stdout_log"]).exists():
            _, _, log_lines = parse_log_tail(info["stdout_log"])
            progress_events = parse_progress_events(log_lines)
            if progress_events and progress_events[-1].get("done", 0) > done:
                done = progress_events[-1]["done"]
                item_expected = progress_events[-1].get("expected") or item_expected
        status = "pending"
        if exp_id == running_id:
            status = "running"
        elif exp_id in completed:
            status = "completed"
        elif exp_id in queued:
            status = "queued"
        if status == "queued":
            # Fresh queue runs archive any smoke or stale CSV before launch.
            # Do not present those old rows as live queued progress.
            done = 0
            row_count = 0
        if status == "running" and heartbeat and heartbeat.get("experiment_id") == exp_id:
            done = safe_int(heartbeat.get("done")) or done
            item_expected = safe_int(heartbeat.get("expected")) or item_expected
            row_count = safe_int(heartbeat.get("rows")) or row_count
        rows.append(
            {
                "id": exp_id,
                "label": info.get("label", exp_id),
                "status": status,
                "datasets_done": done,
                "expected_datasets": item_expected,
                "progress_percent": (done / item_expected * 100) if item_expected else 0.0,
                "rows": row_count,
            }
        )
    return rows


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def auth_credentials(self):
        return (
            os.environ.get("RANK_DASHBOARD_USER", ""),
            os.environ.get("RANK_DASHBOARD_PASSWORD", ""),
        )

    def auth_enabled(self):
        user, password = self.auth_credentials()
        return bool(user and password)

    def is_authenticated(self):
        if not self.auth_enabled():
            return True
        expected_user, expected_password = self.auth_credentials()
        header = self.headers.get("Authorization", "")
        if not header.startswith("Basic "):
            return False
        try:
            decoded = base64.b64decode(header.split(" ", 1)[1], validate=True).decode("utf-8")
        except Exception:
            return False
        username, separator, password = decoded.partition(":")
        if not separator:
            return False
        return hmac.compare_digest(username, expected_user) and hmac.compare_digest(password, expected_password)

    def send_unauthorized(self):
        self.send_response(401)
        self.send_header("WWW-Authenticate", f'Basic realm="{AUTH_REALM}", charset="UTF-8"')
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b"authentication required")

    def send_payload(self, status, content_type, payload):
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
        except BrokenPipeError:
            pass
        except ConnectionResetError:
            pass

    def do_GET(self):
        if not self.is_authenticated():
            self.send_unauthorized()
            return
        path = urlparse(self.path).path
        if path == "/":
            self.send_payload(200, "text/html; charset=utf-8", INDEX_HTML.encode())
            return
        if path == "/api/status":
            self.send_payload(200, "application/json; charset=utf-8", json.dumps(build_status()).encode())
            return
        if path == "/api/overview":
            self.send_payload(200, "application/json; charset=utf-8", json.dumps(build_status(include_details=False)).encode())
            return
        if path == "/api/dataset-context":
            self.send_payload(200, "application/json; charset=utf-8", json.dumps(dataset_context_snapshot()).encode())
            return
        if path == "/api/gpt-handoff.json":
            status = build_status(include_details=True)
            payload = gpt_handoff_snapshot(status, status.get("dataset_context") or dataset_context_snapshot())
            self.send_payload(200, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False, indent=2).encode())
            return
        if path == "/api/gpt-handoff.md":
            status = build_status(include_details=True)
            payload = gpt_handoff_snapshot(status, status.get("dataset_context") or dataset_context_snapshot())
            self.send_payload(200, "text/markdown; charset=utf-8", gpt_handoff_markdown(payload).encode())
            return
        if path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Cache-Control", "max-age=86400")
            self.end_headers()
            return
        self.send_payload(404, "text/plain; charset=utf-8", b"not found")

    def do_HEAD(self):
        if not self.is_authenticated():
            self.send_unauthorized()
            return
        path = urlparse(self.path).path
        if path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        if path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Cache-Control", "max-age=86400")
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()


def main():
    port = int(os.environ.get("RANK_DASHBOARD_PORT", "8765"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    PID_PATH.write_text(str(os.getpid()))
    print(f"Rank dashboard listening at http://127.0.0.1:{port}/", flush=True)
    if os.environ.get("RANK_DASHBOARD_USER") and os.environ.get("RANK_DASHBOARD_PASSWORD"):
        print("Rank dashboard HTTP Basic Auth enabled.", flush=True)
    try:
        server.serve_forever()
    finally:
        try:
            PID_PATH.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
