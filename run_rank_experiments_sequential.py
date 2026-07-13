import argparse
import csv
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = Path("/Users/minho/Documents/Dataset")
PYTHON_BIN = "/opt/homebrew/bin/python3"
STATE_PATH = DATA_DIR / "rank_experiments_sequential_state.json"
LOG_PATH = DATA_DIR / "rank_experiments_sequential.log"
QUEUE_LOCK_PATH = DATA_DIR / "rank_experiments_sequential.lock"
HEARTBEAT_PATH = DATA_DIR / "rank_experiments_dashboard_heartbeat.json"

EXPERIMENTS = [
    {
        "id": "rank_v1_train_evt",
        "script": ROOT / "run_rank_ensemble_calibration.py",
        "detail_csv": DATA_DIR / "vae_results_rank_ensemble_calibration_train_evt.csv",
        "summary_csv": DATA_DIR / "vae_results_rank_ensemble_calibration_train_evt_summary.csv",
        "stdout_log": DATA_DIR / "rank_ensemble_train_evt_stdout.log",
    },
    {
        "id": "rank_threshold_calibration",
        "script": ROOT / "run_rank_ensemble_threshold_calibration.py",
        "detail_csv": DATA_DIR / "rank_ensemble_threshold_calibration.csv",
        "summary_csv": DATA_DIR / "rank_ensemble_threshold_calibration_summary.csv",
        "stdout_log": DATA_DIR / "rank_ensemble_threshold_calibration_stdout.log",
    },
    {
        "id": "experiment_26_rocket",
        "script": ROOT / "run_experiment_26_rocket_parallel.py",
        "detail_csv": DATA_DIR / "experiment_26_rocket_results.csv",
        "summary_csv": DATA_DIR / "experiment_26_rocket_summary.csv",
        "stdout_log": DATA_DIR / "experiment_26_rocket_stdout.log",
    },
    {
        "id": "experiment_27_rocket_score_variants",
        "script": ROOT / "run_experiment_27_rocket_score_variants.py",
        "detail_csv": DATA_DIR / "experiment_27_rocket_score_variants_results.csv",
        "summary_csv": DATA_DIR / "experiment_27_rocket_score_variants_summary.csv",
        "stdout_log": DATA_DIR / "experiment_27_rocket_score_variants_stdout.log",
    },
    {
        "id": "experiment_27e_rocket_1024_top32",
        "script": ROOT / "run_experiment_27e_rocket_1024_top32.py",
        "detail_csv": DATA_DIR / "experiment_27e_rocket_1024_top32_results.csv",
        "summary_csv": DATA_DIR / "experiment_27e_rocket_1024_top32_summary.csv",
        "stdout_log": DATA_DIR / "experiment_27e_rocket_1024_top32_stdout.log",
    },
    {
        "id": "experiment_27f_rstsf_interval",
        "script": ROOT / "run_experiment_27f_rstsf_interval.py",
        "detail_csv": DATA_DIR / "experiment_27f_rstsf_interval_results.csv",
        "summary_csv": DATA_DIR / "experiment_27f_rstsf_interval_summary.csv",
        "stdout_log": DATA_DIR / "experiment_27f_rstsf_interval_stdout.log",
    },
    {
        "id": "experiment_28_minirocket_multirocket_features",
        "script": ROOT / "run_experiment_28_minirocket_multirocket_features.py",
        "detail_csv": DATA_DIR / "experiment_28_minirocket_multirocket_features_results.csv",
        "summary_csv": DATA_DIR / "experiment_28_minirocket_multirocket_features_summary.csv",
        "stdout_log": DATA_DIR / "experiment_28_minirocket_multirocket_features_stdout.log",
    },
    {
        "id": "experiment_29_train_normal_threshold_calibration",
        "script": ROOT / "run_experiment_29_train_normal_threshold_calibration.py",
        "detail_csv": DATA_DIR / "experiment_29_train_normal_threshold_calibration_results.csv",
        "summary_csv": DATA_DIR / "experiment_29_train_normal_threshold_calibration_summary.csv",
        "stdout_log": DATA_DIR / "experiment_29_train_normal_threshold_calibration_stdout.log",
    },
    {
        "id": "experiment_30_knn_threshold_sweep",
        "script": ROOT / "run_experiment_30_knn_threshold_sweep.py",
        "detail_csv": DATA_DIR / "experiment_30_knn_threshold_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_30_knn_threshold_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_30_knn_threshold_sweep_stdout.log",
    },
    {
        "id": "experiment_31_knn_operational_budget_sweep",
        "script": ROOT / "run_experiment_31_knn_operational_budget_sweep.py",
        "detail_csv": DATA_DIR / "experiment_31_knn_operational_budget_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_31_knn_operational_budget_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_31_knn_operational_budget_sweep_stdout.log",
    },
    {
        "id": "experiment_32_knn_score_capacity_sweep",
        "script": ROOT / "run_experiment_32_knn_score_capacity_sweep.py",
        "detail_csv": DATA_DIR / "experiment_32_knn_score_capacity_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_32_knn_score_capacity_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_32_knn_score_capacity_sweep_stdout.log",
    },
    {
        "id": "experiment_34_balanced_feature_capacity_sweep",
        "script": ROOT / "run_experiment_34_balanced_feature_capacity_sweep.py",
        "detail_csv": DATA_DIR / "experiment_34_balanced_feature_capacity_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_34_balanced_feature_capacity_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_34_balanced_feature_capacity_sweep_stdout.log",
    },
    {
        "id": "experiment_35_balanced_threshold_policy_sweep",
        "script": ROOT / "run_experiment_35_balanced_threshold_policy_sweep.py",
        "detail_csv": DATA_DIR / "experiment_35_balanced_threshold_policy_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_35_balanced_threshold_policy_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_35_balanced_threshold_policy_sweep_stdout.log",
    },
    {
        "id": "experiment_36_balanced_score_normalization_sweep",
        "script": ROOT / "run_experiment_36_balanced_score_normalization_sweep.py",
        "detail_csv": DATA_DIR / "experiment_36_balanced_score_normalization_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_36_balanced_score_normalization_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_36_balanced_score_normalization_sweep_stdout.log",
    },
    {
        "id": "experiment_37_balanced_bagged_rocket_ensemble",
        "script": ROOT / "run_experiment_37_balanced_bagged_rocket_ensemble.py",
        "detail_csv": DATA_DIR / "experiment_37_balanced_bagged_rocket_ensemble_results.csv",
        "summary_csv": DATA_DIR / "experiment_37_balanced_bagged_rocket_ensemble_summary.csv",
        "stdout_log": DATA_DIR / "experiment_37_balanced_bagged_rocket_ensemble_stdout.log",
    },
    {
        "id": "experiment_38_balanced_actual_length_handling",
        "script": ROOT / "run_experiment_38_balanced_actual_length_handling.py",
        "detail_csv": DATA_DIR / "experiment_38_balanced_actual_length_handling_results.csv",
        "summary_csv": DATA_DIR / "experiment_38_balanced_actual_length_handling_summary.csv",
        "stdout_log": DATA_DIR / "experiment_38_balanced_actual_length_handling_stdout.log",
    },
    {
        "id": "experiment_39_balanced_candidate_retest",
        "script": ROOT / "run_experiment_39_balanced_candidate_retest.py",
        "detail_csv": DATA_DIR / "experiment_39_balanced_candidate_retest_results.csv",
        "summary_csv": DATA_DIR / "experiment_39_balanced_candidate_retest_summary.csv",
        "stdout_log": DATA_DIR / "experiment_39_balanced_candidate_retest_stdout.log",
    },
    {
        "id": "experiment_35_original_threshold_policy_sweep",
        "script": ROOT / "run_experiment_35_original_threshold_policy_sweep.py",
        "detail_csv": DATA_DIR / "experiment_35_original_threshold_policy_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_35_original_threshold_policy_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_35_original_threshold_policy_sweep_stdout.log",
    },
    {
        "id": "experiment_36_original_score_normalization_sweep",
        "script": ROOT / "run_experiment_36_original_score_normalization_sweep.py",
        "detail_csv": DATA_DIR / "experiment_36_original_score_normalization_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_36_original_score_normalization_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_36_original_score_normalization_sweep_stdout.log",
    },
    {
        "id": "experiment_37_original_bagged_rocket_ensemble",
        "script": ROOT / "run_experiment_37_original_bagged_rocket_ensemble.py",
        "detail_csv": DATA_DIR / "experiment_37_original_bagged_rocket_ensemble_results.csv",
        "summary_csv": DATA_DIR / "experiment_37_original_bagged_rocket_ensemble_summary.csv",
        "stdout_log": DATA_DIR / "experiment_37_original_bagged_rocket_ensemble_stdout.log",
    },
    {
        "id": "experiment_38_original_actual_length_handling",
        "script": ROOT / "run_experiment_38_original_actual_length_handling.py",
        "detail_csv": DATA_DIR / "experiment_38_original_actual_length_handling_results.csv",
        "summary_csv": DATA_DIR / "experiment_38_original_actual_length_handling_summary.csv",
        "stdout_log": DATA_DIR / "experiment_38_original_actual_length_handling_stdout.log",
    },
    {
        "id": "experiment_39_original_candidate_retest",
        "script": ROOT / "run_experiment_39_original_candidate_retest.py",
        "detail_csv": DATA_DIR / "experiment_39_original_candidate_retest_results.csv",
        "summary_csv": DATA_DIR / "experiment_39_original_candidate_retest_summary.csv",
        "stdout_log": DATA_DIR / "experiment_39_original_candidate_retest_stdout.log",
    },
    {
        "id": "experiment_40_original_score_normalization_sweep",
        "script": ROOT / "run_experiment_40_original_score_normalization_sweep.py",
        "detail_csv": DATA_DIR / "experiment_40_original_score_normalization_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_40_original_score_normalization_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_40_original_score_normalization_sweep_stdout.log",
    },
    {
        "id": "experiment_41_multi_aug_robust_baseline",
        "script": ROOT / "run_experiment_41_multi_aug_robust_baseline.py",
        "detail_csv": DATA_DIR / "experiment_41_multi_aug_robust_baseline_results.csv",
        "summary_csv": DATA_DIR / "experiment_41_multi_aug_robust_baseline_summary.csv",
        "stdout_log": DATA_DIR / "experiment_41_multi_aug_robust_baseline_stdout.log",
    },
    {
        "id": "experiment_42_multi_aug_robust_operational",
        "script": ROOT / "run_experiment_42_multi_aug_robust_operational.py",
        "detail_csv": DATA_DIR / "experiment_42_multi_aug_robust_operational_results.csv",
        "summary_csv": DATA_DIR / "experiment_42_multi_aug_robust_operational_summary.csv",
        "stdout_log": DATA_DIR / "experiment_42_multi_aug_robust_operational_stdout.log",
    },
    {
        "id": "experiment_43_explanation_space_transforms",
        "script": ROOT / "run_experiment_43_explanation_space_transforms.py",
        "detail_csv": DATA_DIR / "experiment_43_explanation_space_transforms_results.csv",
        "summary_csv": DATA_DIR / "experiment_43_explanation_space_transforms_summary.csv",
        "stdout_log": DATA_DIR / "experiment_43_explanation_space_transforms_stdout.log",
    },
    {
        "id": "experiment_44_classical_embedding_baselines",
        "script": ROOT / "run_experiment_44_classical_embedding_baselines.py",
        "detail_csv": DATA_DIR / "experiment_44_classical_embedding_baselines_results.csv",
        "summary_csv": DATA_DIR / "experiment_44_classical_embedding_baselines_summary.csv",
        "stdout_log": DATA_DIR / "experiment_44_classical_embedding_baselines_stdout.log",
    },
    {
        "id": "experiment_45_model_hard_diagnostic_harness",
        "script": ROOT / "run_experiment_45_model_hard_diagnostic_harness.py",
        "detail_csv": DATA_DIR / "experiment_45_model_hard_diagnostic_harness_results.csv",
        "summary_csv": DATA_DIR / "experiment_45_model_hard_diagnostic_harness_summary.csv",
        "stdout_log": DATA_DIR / "experiment_45_model_hard_diagnostic_harness_stdout.log",
    },
    {
        "id": "experiment_46_model_hard_interval_drcif_lite",
        "script": ROOT / "run_experiment_46_model_hard_interval_drcif_lite.py",
        "detail_csv": DATA_DIR / "experiment_46_model_hard_interval_drcif_lite_results.csv",
        "summary_csv": DATA_DIR / "experiment_46_model_hard_interval_drcif_lite_summary.csv",
        "stdout_log": DATA_DIR / "experiment_46_model_hard_interval_drcif_lite_stdout.log",
    },
    {
        "id": "experiment_47_model_hard_frequency_rocket",
        "script": ROOT / "run_experiment_47_model_hard_frequency_rocket.py",
        "detail_csv": DATA_DIR / "experiment_47_model_hard_frequency_rocket_results.csv",
        "summary_csv": DATA_DIR / "experiment_47_model_hard_frequency_rocket_summary.csv",
        "stdout_log": DATA_DIR / "experiment_47_model_hard_frequency_rocket_stdout.log",
    },
    {
        "id": "experiment_48_model_hard_shapelet_prototype",
        "script": ROOT / "run_experiment_48_model_hard_shapelet_prototype.py",
        "detail_csv": DATA_DIR / "experiment_48_model_hard_shapelet_prototype_results.csv",
        "summary_csv": DATA_DIR / "experiment_48_model_hard_shapelet_prototype_summary.csv",
        "stdout_log": DATA_DIR / "experiment_48_model_hard_shapelet_prototype_stdout.log",
    },
    {
        "id": "experiment_49_model_hard_anomaly_injection",
        "script": ROOT / "run_experiment_49_model_hard_anomaly_injection.py",
        "detail_csv": DATA_DIR / "experiment_49_model_hard_anomaly_injection_results.csv",
        "summary_csv": DATA_DIR / "experiment_49_model_hard_anomaly_injection_summary.csv",
        "stdout_log": DATA_DIR / "experiment_49_model_hard_anomaly_injection_stdout.log",
    },
    {
        "id": "experiment_50_model_hard_timeseries_imaging",
        "script": ROOT / "run_experiment_50_model_hard_timeseries_imaging.py",
        "detail_csv": DATA_DIR / "experiment_50_model_hard_timeseries_imaging_results.csv",
        "summary_csv": DATA_DIR / "experiment_50_model_hard_timeseries_imaging_summary.csv",
        "stdout_log": DATA_DIR / "experiment_50_model_hard_timeseries_imaging_stdout.log",
    },
    {
        "id": "experiment_51_full_timeseries_imaging_selector_probe",
        "script": ROOT / "run_experiment_51_full_timeseries_imaging_selector_probe.py",
        "detail_csv": DATA_DIR / "experiment_51_full_timeseries_imaging_selector_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_51_full_timeseries_imaging_selector_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_51_full_timeseries_imaging_selector_probe_stdout.log",
    },
    {
        "id": "experiment_52_imaging_multiscale_fusion_probe",
        "script": ROOT / "run_experiment_52_imaging_multiscale_fusion_probe.py",
        "detail_csv": DATA_DIR / "experiment_52_imaging_multiscale_fusion_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_52_imaging_multiscale_fusion_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_52_imaging_multiscale_fusion_probe_stdout.log",
    },
    {
        "id": "experiment_53a_imaging_texture_features_probe",
        "script": ROOT / "run_experiment_53a_imaging_texture_features_probe.py",
        "detail_csv": DATA_DIR / "experiment_53a_imaging_texture_features_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_53a_imaging_texture_features_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_53a_imaging_texture_features_probe_stdout.log",
    },
    {
        "id": "experiment_53b_imaging_train_robust_scaling_probe",
        "script": ROOT / "run_experiment_53b_imaging_train_robust_scaling_probe.py",
        "detail_csv": DATA_DIR / "experiment_53b_imaging_train_robust_scaling_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_53b_imaging_train_robust_scaling_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_53b_imaging_train_robust_scaling_probe_stdout.log",
    },
    {
        "id": "experiment_54_imaging_resolution_pca_sweep",
        "script": ROOT / "run_experiment_54_imaging_resolution_pca_sweep.py",
        "detail_csv": DATA_DIR / "experiment_54_imaging_resolution_pca_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_54_imaging_resolution_pca_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_54_imaging_resolution_pca_sweep_stdout.log",
    },
    {
        "id": "experiment_55_imaging_scaling_sweep",
        "script": ROOT / "run_experiment_55_imaging_scaling_sweep.py",
        "detail_csv": DATA_DIR / "experiment_55_imaging_scaling_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_55_imaging_scaling_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_55_imaging_scaling_sweep_stdout.log",
    },
    {
        "id": "experiment_56_imaging_glcm_texture_probe",
        "script": ROOT / "run_experiment_56_imaging_glcm_texture_probe.py",
        "detail_csv": DATA_DIR / "experiment_56_imaging_glcm_texture_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_56_imaging_glcm_texture_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_56_imaging_glcm_texture_probe_stdout.log",
    },
    {
        "id": "experiment_57_imaging_small_cnn_mps_probe",
        "script": ROOT / "run_experiment_57_imaging_small_cnn_mps_probe.py",
        "detail_csv": DATA_DIR / "experiment_57_imaging_small_cnn_mps_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_57_imaging_small_cnn_mps_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_57_imaging_small_cnn_mps_probe_stdout.log",
    },
    {
        "id": "experiment_58_imaging_pretrained_cnn_feature_probe",
        "script": ROOT / "run_experiment_58_imaging_pretrained_cnn_feature_probe.py",
        "detail_csv": DATA_DIR / "experiment_58_imaging_pretrained_cnn_feature_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_58_imaging_pretrained_cnn_feature_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_58_imaging_pretrained_cnn_feature_probe_stdout.log",
    },
    {
        "id": "experiment_59_rocket_imaging_selector",
        "script": ROOT / "run_experiment_59_rocket_imaging_selector.py",
        "detail_csv": DATA_DIR / "experiment_59_rocket_imaging_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_59_rocket_imaging_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_59_rocket_imaging_selector_stdout.log",
    },
    {
        "id": "experiment_60_selector_fp_guard_variants",
        "script": ROOT / "run_experiment_60_selector_fp_guard_variants.py",
        "detail_csv": DATA_DIR / "experiment_60_selector_fp_guard_variants_results.csv",
        "summary_csv": DATA_DIR / "experiment_60_selector_fp_guard_variants_summary.csv",
        "stdout_log": DATA_DIR / "experiment_60_selector_fp_guard_variants_stdout.log",
    },
    {
        "id": "experiment_61_selector_index_agreement",
        "script": ROOT / "run_experiment_61_selector_index_agreement.py",
        "detail_csv": DATA_DIR / "experiment_61_selector_index_agreement_results.csv",
        "summary_csv": DATA_DIR / "experiment_61_selector_index_agreement_summary.csv",
        "stdout_log": DATA_DIR / "experiment_61_selector_index_agreement_stdout.log",
    },
    {
        "id": "experiment_62_selector_guarded_index_agreement",
        "script": ROOT / "run_experiment_62_selector_guarded_index_agreement.py",
        "detail_csv": DATA_DIR / "experiment_62_selector_guarded_index_agreement_results.csv",
        "summary_csv": DATA_DIR / "experiment_62_selector_guarded_index_agreement_summary.csv",
        "stdout_log": DATA_DIR / "experiment_62_selector_guarded_index_agreement_stdout.log",
    },
    {
        "id": "experiment_63_guarded_cap_sweep",
        "script": ROOT / "run_experiment_63_guarded_cap_sweep.py",
        "detail_csv": DATA_DIR / "experiment_63_guarded_cap_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_63_guarded_cap_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_63_guarded_cap_sweep_stdout.log",
    },
    {
        "id": "experiment_64_guarded_with_fallback",
        "script": ROOT / "run_experiment_64_guarded_with_fallback.py",
        "detail_csv": DATA_DIR / "experiment_64_guarded_with_fallback_results.csv",
        "summary_csv": DATA_DIR / "experiment_64_guarded_with_fallback_summary.csv",
        "stdout_log": DATA_DIR / "experiment_64_guarded_with_fallback_stdout.log",
    },
    {
        "id": "experiment_65_confidence_tier_selector",
        "script": ROOT / "run_experiment_65_confidence_tier_selector.py",
        "detail_csv": DATA_DIR / "experiment_65_confidence_tier_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_65_confidence_tier_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_65_confidence_tier_selector_stdout.log",
    },
    {
        "id": "experiment_66_train_normal_alert_budget",
        "script": ROOT / "run_experiment_66_train_normal_alert_budget.py",
        "detail_csv": DATA_DIR / "experiment_66_train_normal_alert_budget_results.csv",
        "summary_csv": DATA_DIR / "experiment_66_train_normal_alert_budget_summary.csv",
        "stdout_log": DATA_DIR / "experiment_66_train_normal_alert_budget_stdout.log",
    },
    {
        "id": "experiment_67_hard_family_fallback_selector",
        "script": ROOT / "run_experiment_67_hard_family_fallback_selector.py",
        "detail_csv": DATA_DIR / "experiment_67_hard_family_fallback_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_67_hard_family_fallback_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_67_hard_family_fallback_selector_stdout.log",
    },
    {
        "id": "experiment_68_final_operational_selector",
        "script": ROOT / "run_experiment_68_final_operational_selector.py",
        "detail_csv": DATA_DIR / "experiment_68_final_operational_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_68_final_operational_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_68_final_operational_selector_stdout.log",
    },
    {
        "id": "experiment_68b_final_operational_fallback_sweep",
        "script": ROOT / "run_experiment_68b_final_operational_fallback_sweep.py",
        "detail_csv": DATA_DIR / "experiment_68b_final_operational_fallback_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_68b_final_operational_fallback_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_68b_final_operational_fallback_sweep_stdout.log",
    },
    {
        "id": "experiment_69_operational_train_exceed_calibration",
        "script": ROOT / "run_experiment_69_operational_train_exceed_calibration.py",
        "detail_csv": DATA_DIR / "experiment_69_operational_train_exceed_calibration_results.csv",
        "summary_csv": DATA_DIR / "experiment_69_operational_train_exceed_calibration_summary.csv",
        "stdout_log": DATA_DIR / "experiment_69_operational_train_exceed_calibration_stdout.log",
    },
    {
        "id": "experiment_69b_no_prediction_fallback_calibration",
        "script": ROOT / "run_experiment_69b_no_prediction_fallback_calibration.py",
        "detail_csv": DATA_DIR / "experiment_69b_no_prediction_fallback_calibration_results.csv",
        "summary_csv": DATA_DIR / "experiment_69b_no_prediction_fallback_calibration_summary.csv",
        "stdout_log": DATA_DIR / "experiment_69b_no_prediction_fallback_calibration_stdout.log",
    },
    {
        "id": "experiment_70_zero_mode_family_repair_selector",
        "script": ROOT / "run_experiment_70_zero_mode_family_repair_selector.py",
        "detail_csv": DATA_DIR / "experiment_70_zero_mode_family_repair_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_70_zero_mode_family_repair_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_70_zero_mode_family_repair_selector_stdout.log",
    },
    {
        "id": "experiment_71a_large_data_rocket_fallback",
        "script": ROOT / "run_experiment_71a_large_data_rocket_fallback.py",
        "detail_csv": DATA_DIR / "experiment_71a_large_data_rocket_fallback_results.csv",
        "summary_csv": DATA_DIR / "experiment_71a_large_data_rocket_fallback_summary.csv",
        "stdout_log": DATA_DIR / "experiment_71a_large_data_rocket_fallback_stdout.log",
    },
    {
        "id": "experiment_71b_large_data_rocket_review_tier",
        "script": ROOT / "run_experiment_71b_large_data_rocket_review_tier.py",
        "detail_csv": DATA_DIR / "experiment_71b_large_data_rocket_review_tier_results.csv",
        "summary_csv": DATA_DIR / "experiment_71b_large_data_rocket_review_tier_summary.csv",
        "stdout_log": DATA_DIR / "experiment_71b_large_data_rocket_review_tier_stdout.log",
    },
    {
        "id": "experiment_72a_large_data_rank_ensemble",
        "script": ROOT / "run_experiment_72a_large_data_rank_ensemble.py",
        "detail_csv": DATA_DIR / "experiment_72a_large_data_rank_ensemble_results.csv",
        "summary_csv": DATA_DIR / "experiment_72a_large_data_rank_ensemble_summary.csv",
        "stdout_log": DATA_DIR / "experiment_72a_large_data_rank_ensemble_stdout.log",
    },
    {
        "id": "experiment_72b_large_data_source_disagreement",
        "script": ROOT / "run_experiment_72b_large_data_source_disagreement.py",
        "detail_csv": DATA_DIR / "experiment_72b_large_data_source_disagreement_results.csv",
        "summary_csv": DATA_DIR / "experiment_72b_large_data_source_disagreement_summary.csv",
        "stdout_log": DATA_DIR / "experiment_72b_large_data_source_disagreement_stdout.log",
    },
    {
        "id": "experiment_73a_large_rank_rocket_guard",
        "script": ROOT / "run_experiment_73a_large_rank_rocket_guard.py",
        "detail_csv": DATA_DIR / "experiment_73a_large_rank_rocket_guard_results.csv",
        "summary_csv": DATA_DIR / "experiment_73a_large_rank_rocket_guard_summary.csv",
        "stdout_log": DATA_DIR / "experiment_73a_large_rank_rocket_guard_stdout.log",
    },
    {
        "id": "experiment_73b_large_rank_two_model_guard",
        "script": ROOT / "run_experiment_73b_large_rank_two_model_guard.py",
        "detail_csv": DATA_DIR / "experiment_73b_large_rank_two_model_guard_results.csv",
        "summary_csv": DATA_DIR / "experiment_73b_large_rank_two_model_guard_summary.csv",
        "stdout_log": DATA_DIR / "experiment_73b_large_rank_two_model_guard_stdout.log",
    },
    {
        "id": "experiment_73c_large_rank_budget_guard",
        "script": ROOT / "run_experiment_73c_large_rank_budget_guard.py",
        "detail_csv": DATA_DIR / "experiment_73c_large_rank_budget_guard_results.csv",
        "summary_csv": DATA_DIR / "experiment_73c_large_rank_budget_guard_summary.csv",
        "stdout_log": DATA_DIR / "experiment_73c_large_rank_budget_guard_stdout.log",
    },
    {
        "id": "experiment_73d_large_rank_combined_guard",
        "script": ROOT / "run_experiment_73d_large_rank_combined_guard.py",
        "detail_csv": DATA_DIR / "experiment_73d_large_rank_combined_guard_results.csv",
        "summary_csv": DATA_DIR / "experiment_73d_large_rank_combined_guard_summary.csv",
        "stdout_log": DATA_DIR / "experiment_73d_large_rank_combined_guard_stdout.log",
    },
    {
        "id": "experiment_74a_large_rank_margin_guard",
        "script": ROOT / "run_experiment_74a_large_rank_margin_guard.py",
        "detail_csv": DATA_DIR / "experiment_74a_large_rank_margin_guard_results.csv",
        "summary_csv": DATA_DIR / "experiment_74a_large_rank_margin_guard_summary.csv",
        "stdout_log": DATA_DIR / "experiment_74a_large_rank_margin_guard_stdout.log",
    },
    {
        "id": "experiment_74b_large_rank_family_budget",
        "script": ROOT / "run_experiment_74b_large_rank_family_budget.py",
        "detail_csv": DATA_DIR / "experiment_74b_large_rank_family_budget_results.csv",
        "summary_csv": DATA_DIR / "experiment_74b_large_rank_family_budget_summary.csv",
        "stdout_log": DATA_DIR / "experiment_74b_large_rank_family_budget_stdout.log",
    },
    {
        "id": "experiment_74c_large_rank_margin_family_guard",
        "script": ROOT / "run_experiment_74c_large_rank_margin_family_guard.py",
        "detail_csv": DATA_DIR / "experiment_74c_large_rank_margin_family_guard_results.csv",
        "summary_csv": DATA_DIR / "experiment_74c_large_rank_margin_family_guard_summary.csv",
        "stdout_log": DATA_DIR / "experiment_74c_large_rank_margin_family_guard_stdout.log",
    },
    {
        "id": "experiment_74d_large_rank_review_tier_split",
        "script": ROOT / "run_experiment_74d_large_rank_review_tier_split.py",
        "detail_csv": DATA_DIR / "experiment_74d_large_rank_review_tier_split_results.csv",
        "summary_csv": DATA_DIR / "experiment_74d_large_rank_review_tier_split_summary.csv",
        "stdout_log": DATA_DIR / "experiment_74d_large_rank_review_tier_split_stdout.log",
    },
    {
        "id": "experiment_75_score_source_diagnostic_harness",
        "script": ROOT / "run_experiment_75_score_source_diagnostic_harness.py",
        "detail_csv": DATA_DIR / "experiment_75_score_source_diagnostic_harness_results.csv",
        "summary_csv": DATA_DIR / "experiment_75_score_source_diagnostic_harness_summary.csv",
        "stdout_log": DATA_DIR / "experiment_75_score_source_diagnostic_harness_stdout.log",
    },
    {
        "id": "experiment_76_spectral_derivative_rocket_hard_family",
        "script": ROOT / "run_experiment_76_spectral_derivative_rocket_hard_family.py",
        "detail_csv": DATA_DIR / "experiment_76_spectral_derivative_rocket_hard_family_results.csv",
        "summary_csv": DATA_DIR / "experiment_76_spectral_derivative_rocket_hard_family_summary.csv",
        "stdout_log": DATA_DIR / "experiment_76_spectral_derivative_rocket_hard_family_stdout.log",
    },
    {
        "id": "experiment_77_shapelet_prototype_low_train_guard",
        "script": ROOT / "run_experiment_77_shapelet_prototype_low_train_guard.py",
        "detail_csv": DATA_DIR / "experiment_77_shapelet_prototype_low_train_guard_results.csv",
        "summary_csv": DATA_DIR / "experiment_77_shapelet_prototype_low_train_guard_summary.csv",
        "stdout_log": DATA_DIR / "experiment_77_shapelet_prototype_low_train_guard_stdout.log",
    },
    {
        "id": "experiment_78_robust_density_corrected_score",
        "script": ROOT / "run_experiment_78_robust_density_corrected_score.py",
        "detail_csv": DATA_DIR / "experiment_78_robust_density_corrected_score_results.csv",
        "summary_csv": DATA_DIR / "experiment_78_robust_density_corrected_score_summary.csv",
        "stdout_log": DATA_DIR / "experiment_78_robust_density_corrected_score_stdout.log",
    },
    {
        "id": "experiment_79_vae_epoch_sweep_guarded",
        "script": ROOT / "run_experiment_79_vae_epoch_sweep_guarded.py",
        "detail_csv": DATA_DIR / "experiment_79_vae_epoch_sweep_guarded_results.csv",
        "summary_csv": DATA_DIR / "experiment_79_vae_epoch_sweep_guarded_summary.csv",
        "stdout_log": DATA_DIR / "experiment_79_vae_epoch_sweep_guarded_stdout.log",
    },
    {
        "id": "experiment_80_train_only_score_selector",
        "script": ROOT / "run_experiment_80_train_only_score_selector.py",
        "detail_csv": DATA_DIR / "experiment_80_train_only_score_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_80_train_only_score_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_80_train_only_score_selector_stdout.log",
    },
    {
        "id": "experiment_81_aeon_multirocket_official_full",
        "script": ROOT / "run_experiment_81_aeon_multirocket_official_full.py",
        "detail_csv": DATA_DIR / "experiment_81_aeon_multirocket_official_full_results.csv",
        "summary_csv": DATA_DIR / "experiment_81_aeon_multirocket_official_full_summary.csv",
        "stdout_log": DATA_DIR / "experiment_81_aeon_multirocket_official_full_stdout.log",
    },
    {
        "id": "experiment_82_hydra_hard_family_subset",
        "script": ROOT / "run_experiment_82_hydra_hard_family_subset.py",
        "detail_csv": DATA_DIR / "experiment_82_hydra_hard_family_subset_results.csv",
        "summary_csv": DATA_DIR / "experiment_82_hydra_hard_family_subset_summary.csv",
        "stdout_log": DATA_DIR / "experiment_82_hydra_hard_family_subset_stdout.log",
    },
    {
        "id": "experiment_83_multirocket_hydra_hard339",
        "script": ROOT / "run_experiment_83_multirocket_hydra_hard339.py",
        "detail_csv": DATA_DIR / "experiment_83_multirocket_hydra_hard339_results.csv",
        "summary_csv": DATA_DIR / "experiment_83_multirocket_hydra_hard339_summary.csv",
        "stdout_log": DATA_DIR / "experiment_83_multirocket_hydra_hard339_stdout.log",
    },
    {
        "id": "experiment_84_feature_pruning_operational_stability",
        "script": ROOT / "run_experiment_84_feature_pruning_operational_stability.py",
        "detail_csv": DATA_DIR / "experiment_84_feature_pruning_operational_stability_results.csv",
        "summary_csv": DATA_DIR / "experiment_84_feature_pruning_operational_stability_summary.csv",
        "stdout_log": DATA_DIR / "experiment_84_feature_pruning_operational_stability_stdout.log",
    },
    {
        "id": "experiment_85_exp84_hard_specialist_selector",
        "script": ROOT / "run_experiment_85_exp84_hard_specialist_selector.py",
        "detail_csv": DATA_DIR / "experiment_85_exp84_hard_specialist_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_85_exp84_hard_specialist_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_85_exp84_hard_specialist_selector_stdout.log",
    },
    {
        "id": "experiment_86_train_only_agreement_exp84_selector",
        "script": ROOT / "run_experiment_86_train_only_agreement_exp84_selector.py",
        "detail_csv": DATA_DIR / "experiment_86_train_only_agreement_exp84_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_86_train_only_agreement_exp84_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_86_train_only_agreement_exp84_selector_stdout.log",
    },
    {
        "id": "experiment_87_exp84_index_diagnostics",
        "script": ROOT / "run_experiment_87_exp84_index_diagnostics.py",
        "detail_csv": DATA_DIR / "experiment_87_exp84_index_diagnostics_results.csv",
        "summary_csv": DATA_DIR / "experiment_87_exp84_index_diagnostics_summary.csv",
        "stdout_log": DATA_DIR / "experiment_87_exp84_index_diagnostics_stdout.log",
    },
    {
        "id": "experiment_88_true_agreement_exp84_selector",
        "script": ROOT / "run_experiment_88_true_agreement_exp84_selector.py",
        "detail_csv": DATA_DIR / "experiment_88_true_agreement_exp84_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_88_true_agreement_exp84_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_88_true_agreement_exp84_selector_stdout.log",
    },
    {
        "id": "experiment_89_74d_with_exp84_candidate",
        "script": ROOT / "run_experiment_89_74d_with_exp84_candidate.py",
        "detail_csv": DATA_DIR / "experiment_89_74d_with_exp84_candidate_results.csv",
        "summary_csv": DATA_DIR / "experiment_89_74d_with_exp84_candidate_summary.csv",
        "stdout_log": DATA_DIR / "experiment_89_74d_with_exp84_candidate_stdout.log",
    },
    {
        "id": "experiment_90_zero_f1_repair_selector",
        "script": ROOT / "run_experiment_90_zero_f1_repair_selector.py",
        "detail_csv": DATA_DIR / "experiment_90_zero_f1_repair_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_90_zero_f1_repair_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_90_zero_f1_repair_selector_stdout.log",
    },
    {
        "id": "experiment_91_guarded_candidate_union_repair",
        "script": ROOT / "run_experiment_91_guarded_candidate_union_repair.py",
        "detail_csv": DATA_DIR / "experiment_91_guarded_candidate_union_repair_results.csv",
        "summary_csv": DATA_DIR / "experiment_91_guarded_candidate_union_repair_summary.csv",
        "stdout_log": DATA_DIR / "experiment_91_guarded_candidate_union_repair_stdout.log",
    },
    {
        "id": "experiment_92_operational_hybrid_selector",
        "script": ROOT / "run_experiment_92_operational_hybrid_selector.py",
        "detail_csv": DATA_DIR / "experiment_92_operational_hybrid_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_92_operational_hybrid_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_92_operational_hybrid_selector_stdout.log",
    },
    {
        "id": "experiment_93_nonpos_candidate_reranker",
        "script": ROOT / "run_experiment_93_nonpos_candidate_reranker.py",
        "detail_csv": DATA_DIR / "experiment_93_nonpos_candidate_reranker_results.csv",
        "summary_csv": DATA_DIR / "experiment_93_nonpos_candidate_reranker_summary.csv",
        "stdout_log": DATA_DIR / "experiment_93_nonpos_candidate_reranker_stdout.log",
    },
    {
        "id": "experiment_94_nonpos_rank_consensus_v2",
        "script": ROOT / "run_experiment_94_nonpos_rank_consensus_v2.py",
        "detail_csv": DATA_DIR / "experiment_94_nonpos_rank_consensus_v2_results.csv",
        "summary_csv": DATA_DIR / "experiment_94_nonpos_rank_consensus_v2_summary.csv",
        "stdout_log": DATA_DIR / "experiment_94_nonpos_rank_consensus_v2_stdout.log",
    },
    {
        "id": "experiment_95_topk_review_tier",
        "script": ROOT / "run_experiment_95_topk_review_tier.py",
        "detail_csv": DATA_DIR / "experiment_95_topk_review_tier_results.csv",
        "summary_csv": DATA_DIR / "experiment_95_topk_review_tier_summary.csv",
        "stdout_log": DATA_DIR / "experiment_95_topk_review_tier_stdout.log",
    },
    {
        "id": "experiment_96_review_tier_operational_workflow",
        "script": ROOT / "run_experiment_96_review_tier_operational_workflow.py",
        "detail_csv": DATA_DIR / "experiment_96_review_tier_operational_workflow_results.csv",
        "summary_csv": DATA_DIR / "experiment_96_review_tier_operational_workflow_summary.csv",
        "stdout_log": DATA_DIR / "experiment_96_review_tier_operational_workflow_stdout.log",
    },
    {
        "id": "experiment_94b_corrected_nonpos_rank_consensus",
        "script": ROOT / "run_experiment_94b_corrected_nonpos_rank_consensus.py",
        "detail_csv": DATA_DIR / "experiment_94b_corrected_nonpos_rank_consensus_results.csv",
        "summary_csv": DATA_DIR / "experiment_94b_corrected_nonpos_rank_consensus_summary.csv",
        "stdout_log": DATA_DIR / "experiment_94b_corrected_nonpos_rank_consensus_stdout.log",
    },
    {
        "id": "experiment_97_zero_f1_feature_need_diagnosis",
        "script": ROOT / "run_experiment_97_zero_f1_feature_need_diagnosis.py",
        "detail_csv": DATA_DIR / "experiment_97_zero_f1_feature_need_diagnosis_results.csv",
        "summary_csv": DATA_DIR / "experiment_97_zero_f1_feature_need_diagnosis_summary.csv",
        "stdout_log": DATA_DIR / "experiment_97_zero_f1_feature_need_diagnosis_stdout.log",
    },
    {
        "id": "experiment_98_tiny_train_normal_pooling",
        "script": ROOT / "run_experiment_98_tiny_train_normal_pooling.py",
        "detail_csv": DATA_DIR / "experiment_98_tiny_train_normal_pooling_results.csv",
        "summary_csv": DATA_DIR / "experiment_98_tiny_train_normal_pooling_summary.csv",
        "stdout_log": DATA_DIR / "experiment_98_tiny_train_normal_pooling_stdout.log",
    },
    {
        "id": "experiment_98b_train_only_tiny_pooling",
        "script": ROOT / "run_experiment_98b_train_only_tiny_pooling.py",
        "detail_csv": DATA_DIR / "experiment_98b_train_only_tiny_pooling_results.csv",
        "summary_csv": DATA_DIR / "experiment_98b_train_only_tiny_pooling_summary.csv",
        "stdout_log": DATA_DIR / "experiment_98b_train_only_tiny_pooling_stdout.log",
    },
    {
        "id": "experiment_99_spectral_derivative_feature_score",
        "script": ROOT / "run_experiment_99_spectral_derivative_feature_score.py",
        "detail_csv": DATA_DIR / "experiment_99_spectral_derivative_feature_score_results.csv",
        "summary_csv": DATA_DIR / "experiment_99_spectral_derivative_feature_score_summary.csv",
        "stdout_log": DATA_DIR / "experiment_99_spectral_derivative_feature_score_stdout.log",
    },
    {
        "id": "experiment_100_spectral_review_and_guard",
        "script": ROOT / "run_experiment_100_spectral_review_and_guard.py",
        "detail_csv": DATA_DIR / "experiment_100_spectral_review_and_guard_results.csv",
        "summary_csv": DATA_DIR / "experiment_100_spectral_review_and_guard_summary.csv",
        "stdout_log": DATA_DIR / "experiment_100_spectral_review_and_guard_stdout.log",
    },
    {
        "id": "experiment_101_shapelet_normal_prototype",
        "script": ROOT / "run_experiment_101_shapelet_normal_prototype.py",
        "detail_csv": DATA_DIR / "experiment_101_shapelet_normal_prototype_results.csv",
        "summary_csv": DATA_DIR / "experiment_101_shapelet_normal_prototype_summary.csv",
        "stdout_log": DATA_DIR / "experiment_101_shapelet_normal_prototype_stdout.log",
    },
    {
        "id": "experiment_102_feature_source_selector",
        "script": ROOT / "run_experiment_102_feature_source_selector.py",
        "detail_csv": DATA_DIR / "experiment_102_feature_source_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_102_feature_source_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_102_feature_source_selector_stdout.log",
    },
    {
        "id": "experiment_103_higher_dim_review_sources",
        "script": ROOT / "run_experiment_103_higher_dim_review_sources.py",
        "detail_csv": DATA_DIR / "experiment_103_higher_dim_review_sources_results.csv",
        "summary_csv": DATA_DIR / "experiment_103_higher_dim_review_sources_summary.csv",
        "stdout_log": DATA_DIR / "experiment_103_higher_dim_review_sources_stdout.log",
    },
    {
        "id": "experiment_104_score_dimensionality_sweep",
        "script": ROOT / "run_experiment_104_score_dimensionality_sweep.py",
        "detail_csv": DATA_DIR / "experiment_104_score_dimensionality_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_104_score_dimensionality_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_104_score_dimensionality_sweep_stdout.log",
    },
    {
        "id": "experiment_105_score_combination_methods",
        "script": ROOT / "run_experiment_105_score_combination_methods.py",
        "detail_csv": DATA_DIR / "experiment_105_score_combination_methods_results.csv",
        "summary_csv": DATA_DIR / "experiment_105_score_combination_methods_summary.csv",
        "stdout_log": DATA_DIR / "experiment_105_score_combination_methods_stdout.log",
    },
    {
        "id": "experiment_106_gated_score_combo_selector",
        "script": ROOT / "run_experiment_106_gated_score_combo_selector.py",
        "detail_csv": DATA_DIR / "experiment_106_gated_score_combo_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_106_gated_score_combo_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_106_gated_score_combo_selector_stdout.log",
    },
    {
        "id": "experiment_107_exp103_combo_disagreement",
        "script": ROOT / "run_experiment_107_exp103_combo_disagreement.py",
        "detail_csv": DATA_DIR / "experiment_107_exp103_combo_disagreement_results.csv",
        "summary_csv": DATA_DIR / "experiment_107_exp103_combo_disagreement_summary.csv",
        "stdout_log": DATA_DIR / "experiment_107_exp103_combo_disagreement_stdout.log",
    },
    {
        "id": "experiment_108_imaging_pretrained_vit_feature_probe",
        "script": ROOT / "run_experiment_108_imaging_pretrained_vit_feature_probe.py",
        "detail_csv": DATA_DIR / "experiment_108_imaging_pretrained_vit_feature_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_108_imaging_pretrained_vit_feature_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_108_imaging_pretrained_vit_feature_probe_stdout.log",
    },
    {
        "id": "experiment_109_vit_compression_alternatives",
        "script": ROOT / "run_experiment_109_vit_compression_alternatives.py",
        "detail_csv": DATA_DIR / "experiment_109_vit_compression_alternatives_results.csv",
        "summary_csv": DATA_DIR / "experiment_109_vit_compression_alternatives_summary.csv",
        "stdout_log": DATA_DIR / "experiment_109_vit_compression_alternatives_stdout.log",
    },
    {
        "id": "experiment_110_exp84_score_backend_probe",
        "script": ROOT / "run_experiment_110_exp84_score_backend_probe.py",
        "detail_csv": DATA_DIR / "experiment_110_exp84_score_backend_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_110_exp84_score_backend_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_110_exp84_score_backend_probe_stdout.log",
    },
    {
        "id": "experiment_111_vit_fast_compression_probe",
        "script": ROOT / "run_experiment_111_vit_fast_compression_probe.py",
        "detail_csv": DATA_DIR / "experiment_111_vit_fast_compression_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_111_vit_fast_compression_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_111_vit_fast_compression_probe_stdout.log",
    },
    {
        "id": "experiment_111b_vit_manifold_compression_probe",
        "script": ROOT / "run_experiment_111b_vit_manifold_compression_probe.py",
        "detail_csv": DATA_DIR / "experiment_111b_vit_manifold_compression_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_111b_vit_manifold_compression_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_111b_vit_manifold_compression_probe_stdout.log",
    },
    {
        "id": "experiment_112_parametric_umap_oof_probe",
        "script": ROOT / "run_experiment_112_parametric_umap_oof_probe.py",
        "detail_csv": DATA_DIR / "experiment_112_parametric_umap_oof_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_112_parametric_umap_oof_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_112_parametric_umap_oof_probe_stdout.log",
        "python_bin": ROOT / ".venv-parametric-umap" / "bin" / "python",
    },
    {
        "id": "experiment_113_train_normal_conformal_fusion",
        "script": ROOT / "run_experiment_113_train_normal_conformal_fusion.py",
        "detail_csv": DATA_DIR / "experiment_113_train_normal_conformal_fusion_results.csv",
        "summary_csv": DATA_DIR / "experiment_113_train_normal_conformal_fusion_summary.csv",
        "stdout_log": DATA_DIR / "experiment_113_train_normal_conformal_fusion_stdout.log",
    },
    {
        "id": "experiment_114_pseudo_anomaly_score_probe",
        "script": ROOT / "run_experiment_114_pseudo_anomaly_score_probe.py",
        "detail_csv": DATA_DIR / "experiment_114_pseudo_anomaly_score_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_114_pseudo_anomaly_score_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_114_pseudo_anomaly_score_probe_stdout.log",
    },
    {
        "id": "experiment_115_local_normal_state_score",
        "script": ROOT / "run_experiment_115_local_normal_state_score.py",
        "detail_csv": DATA_DIR / "experiment_115_local_normal_state_score_results.csv",
        "summary_csv": DATA_DIR / "experiment_115_local_normal_state_score_summary.csv",
        "stdout_log": DATA_DIR / "experiment_115_local_normal_state_score_stdout.log",
    },
    {
        "id": "experiment_116_train_only_reliability_reranker",
        "script": ROOT / "run_experiment_116_train_only_reliability_reranker.py",
        "detail_csv": DATA_DIR / "experiment_116_train_only_reliability_reranker_results.csv",
        "summary_csv": DATA_DIR / "experiment_116_train_only_reliability_reranker_summary.csv",
        "stdout_log": DATA_DIR / "experiment_116_train_only_reliability_reranker_stdout.log",
    },
    {
        "id": "experiment_117_train_only_candidate_budget",
        "script": ROOT / "run_experiment_117_train_only_candidate_budget.py",
        "detail_csv": DATA_DIR / "experiment_117_train_only_candidate_budget_results.csv",
        "summary_csv": DATA_DIR / "experiment_117_train_only_candidate_budget_summary.csv",
        "stdout_log": DATA_DIR / "experiment_117_train_only_candidate_budget_stdout.log",
    },
    {
        "id": "experiment_118_rocket512_knn3_exp93_source_probe",
        "script": ROOT / "run_experiment_118_rocket512_knn3_exp93_source_probe.py",
        "detail_csv": DATA_DIR / "experiment_118_rocket512_knn3_exp93_source_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_118_rocket512_knn3_exp93_source_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_118_rocket512_knn3_exp93_source_probe_stdout.log",
    },
    {
        "id": "experiment_119a_exp93_rank_order_validation",
        "script": ROOT / "run_experiment_119a_exp93_rank_order_validation.py",
        "detail_csv": DATA_DIR / "experiment_119a_exp93_rank_order_validation_results.csv",
        "summary_csv": DATA_DIR / "experiment_119a_exp93_rank_order_validation_summary.csv",
        "stdout_log": DATA_DIR / "experiment_119a_exp93_rank_order_validation_stdout.log",
    },
    {
        "id": "experiment_119b_rocket256_512_validated_rank_compare",
        "script": ROOT / "run_experiment_119b_rocket256_512_validated_rank_compare.py",
        "detail_csv": DATA_DIR / "experiment_119b_rocket256_512_validated_rank_compare_results.csv",
        "summary_csv": DATA_DIR / "experiment_119b_rocket256_512_validated_rank_compare_summary.csv",
        "stdout_log": DATA_DIR / "experiment_119b_rocket256_512_validated_rank_compare_stdout.log",
    },
    {
        "id": "experiment_120_exp93_reference_rebaseline",
        "script": ROOT / "run_experiment_120_exp93_reference_rebaseline.py",
        "detail_csv": DATA_DIR / "experiment_120_exp93_reference_rebaseline_results.csv",
        "summary_csv": DATA_DIR / "experiment_120_exp93_reference_rebaseline_summary.csv",
        "stdout_log": DATA_DIR / "experiment_120_exp93_reference_rebaseline_stdout.log",
    },
    {
        "id": "experiment_121_exp95_validated_rank_review",
        "script": ROOT / "run_experiment_121_exp95_validated_rank_review.py",
        "detail_csv": DATA_DIR / "experiment_121_exp95_validated_rank_review_results.csv",
        "summary_csv": DATA_DIR / "experiment_121_exp95_validated_rank_review_summary.csv",
        "stdout_log": DATA_DIR / "experiment_121_exp95_validated_rank_review_stdout.log",
    },
    {
        "id": "experiment_122_exp96_validated_rank_workflow",
        "script": ROOT / "run_experiment_122_exp96_validated_rank_workflow.py",
        "detail_csv": DATA_DIR / "experiment_122_exp96_validated_rank_workflow_results.csv",
        "summary_csv": DATA_DIR / "experiment_122_exp96_validated_rank_workflow_summary.csv",
        "stdout_log": DATA_DIR / "experiment_122_exp96_validated_rank_workflow_stdout.log",
    },
    {
        "id": "experiment_123_exp103_validated_rank_review_sources",
        "script": ROOT / "run_experiment_123_exp103_validated_rank_review_sources.py",
        "detail_csv": DATA_DIR / "experiment_123_exp103_validated_rank_review_sources_results.csv",
        "summary_csv": DATA_DIR / "experiment_123_exp103_validated_rank_review_sources_summary.csv",
        "stdout_log": DATA_DIR / "experiment_123_exp103_validated_rank_review_sources_stdout.log",
    },
    {
        "id": "experiment_124_exp104_validated_rank_dimensions",
        "script": ROOT / "run_experiment_124_exp104_validated_rank_dimensions.py",
        "detail_csv": DATA_DIR / "experiment_124_exp104_validated_rank_dimensions_results.csv",
        "summary_csv": DATA_DIR / "experiment_124_exp104_validated_rank_dimensions_summary.csv",
        "stdout_log": DATA_DIR / "experiment_124_exp104_validated_rank_dimensions_stdout.log",
    },
    {
        "id": "experiment_125_exp105_validated_rank_combinations",
        "script": ROOT / "run_experiment_125_exp105_validated_rank_combinations.py",
        "detail_csv": DATA_DIR / "experiment_125_exp105_validated_rank_combinations_results.csv",
        "summary_csv": DATA_DIR / "experiment_125_exp105_validated_rank_combinations_summary.csv",
        "stdout_log": DATA_DIR / "experiment_125_exp105_validated_rank_combinations_stdout.log",
    },
    {
        "id": "experiment_126_exp106_validated_rank_gated_combo",
        "script": ROOT / "run_experiment_126_exp106_validated_rank_gated_combo.py",
        "detail_csv": DATA_DIR / "experiment_126_exp106_validated_rank_gated_combo_results.csv",
        "summary_csv": DATA_DIR / "experiment_126_exp106_validated_rank_gated_combo_summary.csv",
        "stdout_log": DATA_DIR / "experiment_126_exp106_validated_rank_gated_combo_stdout.log",
    },
    {
        "id": "experiment_127_exp107_validated_rank_disagreement",
        "script": ROOT / "run_experiment_127_exp107_validated_rank_disagreement.py",
        "detail_csv": DATA_DIR / "experiment_127_exp107_validated_rank_disagreement_results.csv",
        "summary_csv": DATA_DIR / "experiment_127_exp107_validated_rank_disagreement_summary.csv",
        "stdout_log": DATA_DIR / "experiment_127_exp107_validated_rank_disagreement_stdout.log",
    },
    {
        "id": "experiment_128_rocket512_only_review_selector",
        "script": ROOT / "run_experiment_128_rocket512_only_review_selector.py",
        "detail_csv": DATA_DIR / "experiment_128_rocket512_only_review_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_128_rocket512_only_review_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_128_rocket512_only_review_selector_stdout.log",
    },
    {
        "id": "experiment_129_rocket512_base_selector_rebuild",
        "script": ROOT / "run_experiment_129_rocket512_base_selector_rebuild.py",
        "detail_csv": DATA_DIR / "experiment_129_rocket512_base_selector_rebuild_results.csv",
        "summary_csv": DATA_DIR / "experiment_129_rocket512_base_selector_rebuild_summary.csv",
        "stdout_log": DATA_DIR / "experiment_129_rocket512_base_selector_rebuild_stdout.log",
    },
    {
        "id": "experiment_130_rocket512_two_block_knn",
        "script": ROOT / "run_experiment_130_rocket512_two_block_knn.py",
        "detail_csv": DATA_DIR / "experiment_130_rocket512_two_block_knn_results.csv",
        "summary_csv": DATA_DIR / "experiment_130_rocket512_two_block_knn_summary.csv",
        "stdout_log": DATA_DIR / "experiment_130_rocket512_two_block_knn_stdout.log",
    },
    {
        "id": "experiment_131_rocket_block_b_calibration",
        "script": ROOT / "run_experiment_131_rocket_block_b_calibration.py",
        "detail_csv": DATA_DIR / "experiment_131_rocket_block_b_calibration_results.csv",
        "summary_csv": DATA_DIR / "experiment_131_rocket_block_b_calibration_summary.csv",
        "stdout_log": DATA_DIR / "experiment_131_rocket_block_b_calibration_stdout.log",
    },
    {
        "id": "experiment_132_block_b_review_integration",
        "script": ROOT / "run_experiment_132_block_b_review_integration.py",
        "detail_csv": DATA_DIR / "experiment_132_block_b_review_integration_results.csv",
        "summary_csv": DATA_DIR / "experiment_132_block_b_review_integration_summary.csv",
        "stdout_log": DATA_DIR / "experiment_132_block_b_review_integration_stdout.log",
    },
    {
        "id": "experiment_133_block_b_confidence_tiers",
        "script": ROOT / "run_experiment_133_block_b_confidence_tiers.py",
        "detail_csv": DATA_DIR / "experiment_133_block_b_confidence_tiers_results.csv",
        "summary_csv": DATA_DIR / "experiment_133_block_b_confidence_tiers_summary.csv",
        "stdout_log": DATA_DIR / "experiment_133_block_b_confidence_tiers_stdout.log",
    },
    {
        "id": "experiment_134_block_b_review_tail_guard",
        "script": ROOT / "run_experiment_134_block_b_review_tail_guard.py",
        "detail_csv": DATA_DIR / "experiment_134_block_b_review_tail_guard_results.csv",
        "summary_csv": DATA_DIR / "experiment_134_block_b_review_tail_guard_summary.csv",
        "stdout_log": DATA_DIR / "experiment_134_block_b_review_tail_guard_stdout.log",
    },
    {
        "id": "experiment_135_block_c_review_confirmation",
        "script": ROOT / "run_experiment_135_block_c_review_confirmation.py",
        "detail_csv": DATA_DIR / "experiment_135_block_c_review_confirmation_results.csv",
        "summary_csv": DATA_DIR / "experiment_135_block_c_review_confirmation_summary.csv",
        "stdout_log": DATA_DIR / "experiment_135_block_c_review_confirmation_stdout.log",
    },
    {
        "id": "experiment_136_family_holdout_review_audit",
        "script": ROOT / "run_experiment_136_family_holdout_review_audit.py",
        "detail_csv": DATA_DIR / "experiment_136_family_holdout_review_audit_results.csv",
        "summary_csv": DATA_DIR / "experiment_136_family_holdout_review_audit_summary.csv",
        "stdout_log": DATA_DIR / "experiment_136_family_holdout_review_audit_stdout.log",
    },
    {
        "id": "experiment_137_operational_triage",
        "script": ROOT / "run_experiment_137_operational_triage.py",
        "detail_csv": DATA_DIR / "experiment_137_operational_triage_results.csv",
        "summary_csv": DATA_DIR / "experiment_137_operational_triage_summary.csv",
        "stdout_log": DATA_DIR / "experiment_137_operational_triage_stdout.log",
    },
]

DEFAULT_QUEUE = ["experiment_26_rocket"]


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message):
    line = f"[{now()}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")


def load_state():
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text())
    else:
        state = {"completed": [], "running": None, "history": []}
    state.setdefault("completed", [])
    state.setdefault("running", None)
    state.setdefault("history", [])
    state.setdefault("queue", [])
    return state


def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def python_processes():
    out = subprocess.run(
        ["ps", "-axo", "pid,ppid,stat,etime,command"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    matches = []
    for line in out[1:]:
        if (
            "run_rank_ensemble_calibration" in line
            or "run_rank_ensemble_threshold_calibration" in line
            or "run_experiment_" in line
            or "run_rank_experiments_sequential.py" in line
        ):
            if str(os.getpid()) not in line:
                matches.append(line.strip())
    return matches


def active_rank_processes():
    experiment_scripts = {exp["script"].name for exp in EXPERIMENTS}
    matches = []
    for line in python_processes():
        if "python" not in line:
            continue
        command = line.split(None, 4)[-1] if len(line.split(None, 4)) >= 5 else line
        parts = command.split()
        script_names = {Path(part).name for part in parts if part.endswith(".py")}
        if script_names & experiment_scripts:
            matches.append(line)
    return matches


def pid_alive(pid):
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def reconcile_running_state(state):
    running = state.get("running")
    if not running:
        return state
    pids = [running.get("child_pid"), running.get("pid")]
    if any(pid_alive(pid) for pid in pids):
        return state
    stale = dict(running)
    state["running"] = None
    recovered_summary = None
    recovered = False
    try:
        exp = get_experiment(stale.get("id"))
        log_text = exp["stdout_log"].read_text(errors="replace") if exp["stdout_log"].exists() else ""
        recovered_summary = summarize_csv(exp["detail_csv"])
        recovered = bool(recovered_summary and f"{exp['id']} finished" in log_text)
    except (KeyError, OSError, ValueError):
        exp = None
    if recovered:
        state.setdefault("history", []).append(
            {
                "id": stale.get("id"),
                "finished_at": now(),
                "return_code": 0,
                "recovered_state": True,
                "summary": recovered_summary,
            }
        )
        if stale.get("id") not in state.get("completed", []):
            state.setdefault("completed", []).append(stale.get("id"))
        state["queue"] = [item for item in state.get("queue", []) if item != stale.get("id")]
        save_state(state)
        log(f"RECOVER completed {stale.get('id')} from finished output")
        return state
    state.setdefault("history", []).append(
        {
            "id": stale.get("id"),
            "finished_at": now(),
            "return_code": "stale_running_state_cleared",
            "summary": {
                "reason": "No live runner or child PID remained for recorded running state.",
                "stale_running": stale,
            },
        }
    )
    save_state(state)
    log(f"STALE running state cleared id={stale.get('id')} pid={stale.get('pid')} child_pid={stale.get('child_pid')}")
    return state


def get_experiment(exp_id):
    for exp in EXPERIMENTS:
        if exp["id"] == exp_id:
            return exp
    raise SystemExit(f"Unknown experiment id: {exp_id}")


def next_experiment(state):
    queue = [exp_id for exp_id in state.get("queue", []) if exp_id not in state.get("completed", [])]
    if queue:
        return get_experiment(queue[0])
    completed = set(state.get("completed", []))
    for exp in EXPERIMENTS:
        if exp["id"] not in completed:
            return exp
    return None


def queue_experiment(exp_id):
    get_experiment(exp_id)
    state = load_state()
    if exp_id in state.get("completed", []):
        log(f"QUEUE skip already completed {exp_id}")
        return
    if exp_id not in state.get("queue", []):
        state.setdefault("queue", []).append(exp_id)
        save_state(state)
        log(f"QUEUE add {exp_id}")
    else:
        log(f"QUEUE already pending {exp_id}")


def remove_from_queue(exp_id):
    state = load_state()
    queue = state.get("queue", [])
    if exp_id in queue:
        state["queue"] = [item for item in queue if item != exp_id]
        save_state(state)


def summarize_csv(path):
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return {"rows": 0, "datasets": 0, "configs": {}}
    configs = {}
    for row in rows:
        f1_key = "f1_evt" if "f1_evt" in row else ("f1" if "f1" in row else "validated_f1")
        group_name = row.get("config_name") or row.get("source_experiment") or "default"
        if "strategy" in row:
            group_name = f"{row['config_name']}::{row['strategy']}"
        elif row.get("threshold_method"):
            group_name = f"{row['config_name']}::{row['threshold_method']}"
        try:
            configs.setdefault(group_name, []).append(float(row.get(f1_key, 0.0)))
        except (TypeError, ValueError):
            configs.setdefault(group_name, []).append(0.0)
    return {
        "rows": len(rows),
        "datasets": len({row.get("dataset_name") for row in rows if row.get("dataset_name")}),
        "configs": {
            name: {
                "mean_f1_evt": sum(values) / len(values),
                "median_f1_evt": sorted(values)[len(values) // 2],
            }
            for name, values in sorted(configs.items())
        },
    }


def write_heartbeat(update, replace=False):
    current = {} if replace else load_heartbeat()
    current.update(update)
    current["updated_unix"] = time.time()
    temporary = HEARTBEAT_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2))
    temporary.replace(HEARTBEAT_PATH)


def load_heartbeat():
    try:
        return json.loads(HEARTBEAT_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def update_heartbeat_from_output(experiment_id, line):
    match = re.search(
        r"Progress:\s*\[\s*(\d+)\s*/\s*(\d+)\s*\]\s*rows=(\d+)\s+(?:last|dataset)=([^\s]+)(?:\s+errors=(\d+))?",
        line,
    )
    if not match:
        return
    write_heartbeat(
        {
            "experiment_id": experiment_id,
            "status": "running",
            "done": int(match.group(1)),
            "expected": int(match.group(2)),
            "rows": int(match.group(3)),
            "current_dataset": match.group(4),
            "errors": int(match.group(5) or 0),
            "updated_at": now(),
        }
    )


def run_experiment(exp, overlap_checked=False):
    if not overlap_checked:
        active = active_rank_processes()
        if active:
            raise SystemExit("Another rank experiment is already running:\n" + "\n".join(active))
    if not exp["script"].exists():
        raise SystemExit(f"Script does not exist: {exp['script']}")

    archive_existing_outputs(exp)

    state = load_state()
    state["running"] = {"id": exp["id"], "started_at": now(), "pid": os.getpid()}
    save_state(state)
    write_heartbeat(
        {
            "experiment_id": exp["id"],
            "status": "starting",
            "done": 0,
            "expected": None,
            "rows": 0,
            "current_dataset": "",
            "errors": 0,
            "updated_at": now(),
        }
    )

    worker_count = os.environ.get("RANK_EXPERIMENT_WORKERS", "6")
    log(f"START {exp['id']} script={exp['script']} rank_workers={worker_count}")
    start = time.time()
    with exp["stdout_log"].open("w") as stdout_log:
        process = subprocess.Popen(
            [str(exp.get("python_bin", PYTHON_BIN)), str(exp["script"])],
            cwd=str(ROOT),
            env={**os.environ, "RANK_EXPERIMENT_WORKERS": worker_count},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        state["running"]["child_pid"] = process.pid
        save_state(state)
        write_heartbeat({"child_pid": process.pid, "status": "running", "updated_at": now()})
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                stdout_log.write(line)
                stdout_log.flush()
                update_heartbeat_from_output(exp["id"], line)
            return_code = process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGTERM)
            return_code = process.wait()
            raise

    elapsed = time.time() - start
    state = load_state()
    state["running"] = None
    state.setdefault("history", []).append(
        {
            "id": exp["id"],
            "finished_at": now(),
            "return_code": return_code,
            "elapsed_minutes": round(elapsed / 60, 2),
            "summary": summarize_csv(exp["detail_csv"]),
        }
    )
    if return_code == 0 and exp["id"] not in state.get("completed", []):
        state.setdefault("completed", []).append(exp["id"])
    if return_code == 0:
        state["queue"] = [item for item in state.get("queue", []) if item != exp["id"]]
    save_state(state)
    write_heartbeat(
        {
            "experiment_id": exp["id"],
            "status": "completed" if return_code == 0 else "failed",
            "return_code": return_code,
            "updated_at": now(),
        }
    )
    log(f"END {exp['id']} return_code={return_code} elapsed_min={elapsed / 60:.2f}")
    if return_code != 0:
        raise SystemExit(return_code)


def archive_existing_outputs(exp):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for key in ["detail_csv", "summary_csv", "stdout_log"]:
        path = exp[key]
        if path.exists():
            archive_path = path.with_name(f"{path.stem}.before_{exp['id']}_{stamp}{path.suffix}")
            path.rename(archive_path)
            log(f"ARCHIVE {path} -> {archive_path}")


def cmd_status():
    state = load_state()
    print(json.dumps(state, indent=2, sort_keys=True))
    active = active_rank_processes()
    if active:
        print("\nActive rank processes:")
        print("\n".join(active))
    for exp in EXPERIMENTS:
        summary = summarize_csv(exp["detail_csv"])
        if summary:
            print(f"\n{exp['id']} partial summary:")
            print(json.dumps(summary, indent=2, sort_keys=True))


def cmd_list():
    state = load_state()
    completed = set(state.get("completed", []))
    queued = set(state.get("queue", []))
    running_id = (state.get("running") or {}).get("id")
    for exp in EXPERIMENTS:
        if exp["id"] == running_id:
            marker = "running"
        elif exp["id"] in completed:
            marker = "done"
        elif exp["id"] in queued:
            marker = "queued"
        else:
            marker = "pending"
        print(f"{marker:7s} {exp['id']} -> {exp['script'].name}")


def cmd_run_queue(poll_seconds=30):
    with QUEUE_LOCK_PATH.open("w") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("QUEUE runner already active; exiting duplicate runner")
            return
        lock_file.write(f"{os.getpid()}\n")
        lock_file.flush()
        log("QUEUE runner started")
        while True:
            state = reconcile_running_state(load_state())
            queue = [exp_id for exp_id in state.get("queue", []) if exp_id not in state.get("completed", [])]
            if not queue:
                log("QUEUE empty; runner exiting")
                return
            active = active_rank_processes()
            if state.get("running") or active:
                time.sleep(poll_seconds)
                continue
            exp_id = queue[0]
            log(f"QUEUE starting {exp_id}")
            run_experiment(get_experiment(exp_id), overlap_checked=True)


def main():
    parser = argparse.ArgumentParser(description="Run rank experiments one at a time.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    sub.add_parser("status")
    sub.add_parser("run-next")
    sub.add_parser("run-queue")
    queue_one = sub.add_parser("queue")
    queue_one.add_argument("experiment_id")
    run_one = sub.add_parser("run")
    run_one.add_argument("experiment_id")
    args = parser.parse_args()

    if args.command == "list":
        cmd_list()
    elif args.command == "status":
        cmd_status()
    elif args.command == "run-next":
        exp = next_experiment(load_state())
        if exp is None:
            log("No pending experiments.")
            return
        run_experiment(exp)
    elif args.command == "queue":
        queue_experiment(args.experiment_id)
    elif args.command == "run-queue":
        cmd_run_queue()
    elif args.command == "run":
        run_experiment(get_experiment(args.experiment_id))


if __name__ == "__main__":
    main()
