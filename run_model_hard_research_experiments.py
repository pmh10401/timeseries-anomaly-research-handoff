import argparse
import copy
import csv
import logging
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA, PCA, TruncatedSVD
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import RobustScaler
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.svm import OneClassSVM
from scipy.signal import fftconvolve

from run_balanced_improvement_experiment import (
    count_cap_threshold,
    density_knn_score_pair,
    evaluate_threshold,
    rank_normalize_scores,
    score_metrics,
)
from run_experiment_29_train_normal_threshold_calibration import (
    knn_score_pair,
    train_false_positive_stats,
)
from run_experiment_26_rocket import make_kernels, rocket_transform
from run_original_improvement_experiment import (
    DATA_DIR,
    DB_PATH,
    load_original_record,
    rate_for_threshold,
    target_len_for_record,
)
from run_rank_ensemble_calibration import align_series_lengths, z_normalize


DIFFICULTY_PATH = DATA_DIR / "dataset_anomaly_validation_difficulty_20260707.csv"
RNG_SEED = 20260707
DEFAULT_WORKERS = int(os.environ.get("MODEL_HARD_RESEARCH_WORKERS", "4"))
DEFAULT_THRESHOLDS = ["count_cap_2pct", "count_cap_3pct", "adaptive_v0", "family_guard_v1"]
EXCLUDED_DATASETS = {"CornellWhaleChallenge", "Wafer_normal_1"}
_PRETRAINED_CNN_CACHE = {}
_PRETRAINED_VIT_CACHE = {}

POWER_DEVICE_FAMILIES = {
    "ScreenType",
    "LargeKitchenAppliances",
    "SmallKitchenAppliances",
    "ElectricDevices",
    "Computers",
    "RefrigerationDevices",
}
FREQUENCY_FAMILIES = {
    "EthanolLevel",
    "ScreenType",
    "FordA",
    "FordB",
    "Earthquakes",
    "Phoneme",
}
SHAPE_FAMILIES = {
    "HandOutlines",
    "DistalPhalanxOutlineCorrect",
    "MiddlePhalanxOutlineCorrect",
    "ProximalPhalanxOutlineCorrect",
    "PhalangesOutlinesCorrect",
    "Fish",
    "Worms",
    "ArrowHead",
}
INJECTION_FAMILIES = FREQUENCY_FAMILIES | SHAPE_FAMILIES | POWER_DEVICE_FAMILIES
IMAGING_FAMILIES = POWER_DEVICE_FAMILIES | FREQUENCY_FAMILIES | SHAPE_FAMILIES | {
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "GestureMidAirD1",
    "GestureMidAirD2",
    "GestureMidAirD3",
    "MelbournePedestrian",
}
HARD_SCORE_FAMILIES = {
    "Phoneme",
    "PigAirwayPressure",
    "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2",
    "CricketX",
    "CricketY",
    "CricketZ",
    "InlineSkate",
    "GestureMidAirD3",
    "WordSynonyms",
    "ECG5000",
    "Crop",
    "StarLightCurves",
    "MelbournePedestrian",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "FordA",
    "FreezerRegularTrain",
}
HYDRA_PRIORITY_FAMILIES = {
    "Phoneme",
    "CricketZ",
    "InlineSkate",
    "GestureMidAirD3",
}


def env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value else default


AEON_N_JOBS = env_int("AEON_ROCKET_N_JOBS", 1)


EXPERIMENT_SPECS = {
    "experiment_45_model_hard_diagnostic_harness": {
        "label": "Exp 45 - Model-hard diagnostic harness",
        "source_experiment_id": "model_hard_research_plan",
        "target": {"scope": "hard_core", "families": None, "min_train": 0},
        "thresholds": DEFAULT_THRESHOLDS,
        "configs": [
            {"name": "baseline_rocket_256_knn3", "kind": "rocket_knn", "num_kernels": 256, "neighbors": 3},
            {
                "name": "baseline_interval_quantile_24_knn3",
                "kind": "interval_quantile_knn",
                "intervals": 24,
                "neighbors": 3,
            },
            {
                "name": "baseline_fft_band_24_knn3",
                "kind": "frequency_knn",
                "n_bands": 24,
                "neighbors": 3,
            },
        ],
    },
    "experiment_46_model_hard_interval_drcif_lite": {
        "label": "Exp 46 - Model-hard interval quantile DrCIF-lite",
        "source_experiment_id": "interval_quantile_drcif_lite",
        "target": {"scope": "model_hard", "families": POWER_DEVICE_FAMILIES, "min_train": 0},
        "thresholds": DEFAULT_THRESHOLDS,
        "configs": [
            {
                "name": "interval_q16_raw_diff_periodogram_knn3",
                "kind": "interval_quantile_knn",
                "intervals": 16,
                "neighbors": 3,
                "include_diff": True,
                "include_periodogram": True,
            },
            {
                "name": "interval_q24_raw_diff_periodogram_knn5",
                "kind": "interval_quantile_knn",
                "intervals": 24,
                "neighbors": 5,
                "include_diff": True,
                "include_periodogram": True,
            },
            {
                "name": "interval_q32_raw_only_knn3",
                "kind": "interval_quantile_knn",
                "intervals": 32,
                "neighbors": 3,
                "include_diff": False,
                "include_periodogram": False,
            },
        ],
    },
    "experiment_47_model_hard_frequency_rocket": {
        "label": "Exp 47 - Model-hard frequency decomposition ROCKET",
        "source_experiment_id": "frequency_decomposition_rocket",
        "target": {"scope": "model_hard", "families": FREQUENCY_FAMILIES, "min_train": 0},
        "thresholds": DEFAULT_THRESHOLDS,
        "configs": [
            {
                "name": "fft_band_32_knn3",
                "kind": "frequency_knn",
                "n_bands": 32,
                "neighbors": 3,
            },
            {
                "name": "raw_diff_fft_rank_ensemble_knn3",
                "kind": "frequency_rocket_rank_ensemble",
                "num_kernels": 192,
                "neighbors": 3,
                "n_bands": 24,
            },
            {
                "name": "bandpass_rocket_rank_ensemble_knn3",
                "kind": "bandpass_rocket_rank_ensemble",
                "num_kernels": 128,
                "neighbors": 3,
                "bands": [(0.0, 0.18), (0.18, 0.42), (0.42, 1.0)],
            },
        ],
    },
    "experiment_48_model_hard_shapelet_prototype": {
        "label": "Exp 48 - Model-hard shapelet normal prototypes",
        "source_experiment_id": "shapelet_normal_prototype",
        "target": {"scope": "model_hard", "families": SHAPE_FAMILIES, "min_train": 0},
        "thresholds": DEFAULT_THRESHOLDS,
        "configs": [
            {
                "name": "shapelet_proto_24_len16_32_knn3",
                "kind": "shapelet_prototype_knn",
                "num_shapelets": 24,
                "shapelet_lengths": [16, 32],
                "neighbors": 3,
                "max_len": 512,
            },
            {
                "name": "shapelet_proto_36_len16_32_64_knn3",
                "kind": "shapelet_prototype_knn",
                "num_shapelets": 36,
                "shapelet_lengths": [16, 32, 64],
                "neighbors": 3,
                "max_len": 512,
            },
        ],
    },
    "experiment_49_model_hard_anomaly_injection": {
        "label": "Exp 49 - Model-hard CARLA-style anomaly injection score",
        "source_experiment_id": "carla_style_anomaly_injection",
        "target": {"scope": "model_hard", "families": INJECTION_FAMILIES, "min_train": 40},
        "thresholds": DEFAULT_THRESHOLDS,
        "configs": [
            {
                "name": "injection_interval_frequency_centroid",
                "kind": "injection_centroid_score",
                "feature_kind": "interval_frequency",
                "synthetic_per_train": 1,
                "neighbors": 3,
            },
            {
                "name": "injection_interval_frequency_knn",
                "kind": "injection_knn_score",
                "feature_kind": "interval_frequency",
                "synthetic_per_train": 1,
                "neighbors": 3,
            },
        ],
    },
    "experiment_50_model_hard_timeseries_imaging": {
        "label": "Exp 50 - Model-hard time-series imaging smoke test",
        "source_experiment_id": "time_series_imaging_smoke",
        "target": {"scope": "model_hard", "families": IMAGING_FAMILIES, "min_train": 0},
        "thresholds": DEFAULT_THRESHOLDS,
        "configs": [
            {"name": "image_gasf_32_pca32_knn3", "kind": "imaging_knn", "image": "gasf", "size": 32, "pca": 32, "neighbors": 3},
            {"name": "image_mtf_32_pca32_knn3", "kind": "imaging_knn", "image": "mtf", "size": 32, "pca": 32, "neighbors": 3},
            {"name": "image_rp_32_pca32_knn3", "kind": "imaging_knn", "image": "rp", "size": 32, "pca": 32, "neighbors": 3},
            {
                "name": "image_spectrogram_32_pca32_knn3",
                "kind": "imaging_knn",
                "image": "spectrogram",
                "size": 32,
                "pca": 32,
                "neighbors": 3,
            },
        ],
    },
    "experiment_51_full_timeseries_imaging_selector_probe": {
        "label": "Exp 51 - Full original time-series imaging selector probe",
        "source_experiment_id": "time_series_imaging_full_probe",
        "target": {"scope": "full_original", "families": None, "min_train": 0},
        "thresholds": DEFAULT_THRESHOLDS,
        "configs": [
            {"name": "image_gasf_32_pca32_knn3", "kind": "imaging_knn", "image": "gasf", "size": 32, "pca": 32, "neighbors": 3},
            {"name": "image_mtf_32_pca32_knn3", "kind": "imaging_knn", "image": "mtf", "size": 32, "pca": 32, "neighbors": 3},
            {"name": "image_rp_32_pca32_knn3", "kind": "imaging_knn", "image": "rp", "size": 32, "pca": 32, "neighbors": 3},
            {
                "name": "image_spectrogram_32_pca32_knn3",
                "kind": "imaging_knn",
                "image": "spectrogram",
                "size": 32,
                "pca": 32,
                "neighbors": 3,
            },
        ],
    },
    "experiment_52_imaging_multiscale_fusion_probe": {
        "label": "Exp 52 - Imaging multiscale fusion probe",
        "source_experiment_id": "time_series_imaging_multiscale_fusion_probe",
        "target": {"scope": "full_original", "families": None, "min_train": 0},
        "thresholds": ["count_cap_2pct", "count_cap_3pct", "family_guard_v1"],
        "configs": [
            {
                "name": "fusion_spectrogram_gasf_rp_64_pca64_knn3",
                "kind": "imaging_knn",
                "image": ["spectrogram", "gasf", "rp"],
                "size": 64,
                "pca": 64,
                "neighbors": 3,
            },
            {
                "name": "image_scalogram_64_pca64_knn3",
                "kind": "imaging_knn",
                "image": "scalogram",
                "size": 64,
                "pca": 64,
                "neighbors": 3,
            },
            {
                "name": "fusion_gasf_gadf_64_pca64_knn3",
                "kind": "imaging_knn",
                "image": ["gasf", "gadf"],
                "size": 64,
                "pca": 64,
                "neighbors": 3,
            },
            {
                "name": "image_multiscale_rp_64_pca64_knn3",
                "kind": "imaging_knn",
                "image": "multiscale_rp",
                "size": 64,
                "pca": 64,
                "neighbors": 3,
            },
            {
                "name": "image_spectrogram_64_pca64_knn3",
                "kind": "imaging_knn",
                "image": "spectrogram",
                "size": 64,
                "pca": 64,
                "neighbors": 3,
            },
        ],
    },
    "experiment_53a_imaging_texture_features_probe": {
        "label": "Exp 53a - Imaging HOG/LBP texture feature probe",
        "source_experiment_id": "time_series_imaging_texture_features_probe",
        "target": {"scope": "full_original", "families": None, "min_train": 0},
        "thresholds": ["count_cap_2pct", "count_cap_3pct", "family_guard_v1"],
        "configs": [
            {
                "name": "hog_spectrogram_32_pca32_knn3",
                "kind": "imaging_knn",
                "image": "spectrogram",
                "feature_extractor": "hog",
                "size": 32,
                "pca": 32,
                "neighbors": 3,
            },
            {
                "name": "hog_gasf_32_pca32_knn3",
                "kind": "imaging_knn",
                "image": "gasf",
                "feature_extractor": "hog",
                "size": 32,
                "pca": 32,
                "neighbors": 3,
            },
            {
                "name": "lbp_rp_32_pca32_knn3",
                "kind": "imaging_knn",
                "image": "rp",
                "feature_extractor": "lbp",
                "size": 32,
                "pca": 32,
                "neighbors": 3,
            },
            {
                "name": "lbp_spectrogram_32_pca32_knn3",
                "kind": "imaging_knn",
                "image": "spectrogram",
                "feature_extractor": "lbp",
                "size": 32,
                "pca": 32,
                "neighbors": 3,
            },
        ],
    },
    "experiment_53b_imaging_train_robust_scaling_probe": {
        "label": "Exp 53b - Imaging train-normal robust scaling probe",
        "source_experiment_id": "time_series_imaging_train_robust_scaling_probe",
        "target": {"scope": "full_original", "families": None, "min_train": 0},
        "thresholds": ["count_cap_2pct", "count_cap_3pct", "family_guard_v1"],
        "configs": [
            {
                "name": "trainrobust_spectrogram_32_pca32_knn3",
                "kind": "imaging_knn",
                "image": "spectrogram",
                "series_scale": "train_robust",
                "size": 32,
                "pca": 32,
                "neighbors": 3,
            },
            {
                "name": "trainrobust_gasf_32_pca32_knn3",
                "kind": "imaging_knn",
                "image": "gasf",
                "series_scale": "train_robust",
                "size": 32,
                "pca": 32,
                "neighbors": 3,
            },
            {
                "name": "trainrobust_rp_32_pca32_knn3",
                "kind": "imaging_knn",
                "image": "rp",
                "series_scale": "train_robust",
                "size": 32,
                "pca": 32,
                "neighbors": 3,
            },
            {
                "name": "trainrobust_fusion_spectrogram_gasf_rp_32_pca32_knn3",
                "kind": "imaging_knn",
                "image": ["spectrogram", "gasf", "rp"],
                "series_scale": "train_robust",
                "size": 32,
                "pca": 32,
                "neighbors": 3,
            },
        ],
    },
    "experiment_54_imaging_resolution_pca_sweep": {
        "label": "Exp 54 - Imaging resolution and PCA compression sweep",
        "source_experiment_id": "time_series_imaging_resolution_pca_sweep",
        "target": {"scope": "full_original", "families": None, "min_train": 0},
        "thresholds": ["count_cap_2pct", "count_cap_3pct", "family_guard_v1"],
        "configs": [
            {
                "name": f"image_spectrogram_{size}_pca{pca}_knn3",
                "kind": "imaging_knn",
                "image": "spectrogram",
                "size": size,
                "pca": pca,
                "neighbors": 3,
            }
            for size in (32, 48, 64, 96)
            for pca in (32, 64)
        ],
    },
    "experiment_55_imaging_scaling_sweep": {
        "label": "Exp 55 - Imaging train-normal and per-series scaling sweep",
        "source_experiment_id": "time_series_imaging_scaling_sweep",
        "target": {"scope": "full_original", "families": None, "min_train": 0},
        "thresholds": ["count_cap_2pct", "count_cap_3pct", "family_guard_v1"],
        "configs": [
            {
                "name": f"{scale}_spectrogram_32_pca32_knn3",
                "kind": "imaging_knn",
                "image": "spectrogram",
                "series_scale": scale,
                "size": 32,
                "pca": 32,
                "neighbors": 3,
            }
            for scale in (
                "per_series_z",
                "per_series_z_clip3",
                "per_series_quantile_05_95",
                "train_point_quantile_05_95",
                "train_point_quantile_01_99",
                "train_global_quantile_05_95",
                "train_global_quantile_01_99",
                "train_global_minmax_clip",
            )
        ],
    },
    "experiment_56_imaging_glcm_texture_probe": {
        "label": "Exp 56 - Imaging GLCM texture feature probe",
        "source_experiment_id": "time_series_imaging_glcm_texture_probe",
        "target": {"scope": "full_original", "families": None, "min_train": 0},
        "thresholds": ["count_cap_2pct", "count_cap_3pct", "family_guard_v1"],
        "configs": [
            {
                "name": "glcm_spectrogram_32_pca32_knn3",
                "kind": "imaging_knn",
                "image": "spectrogram",
                "feature_extractor": "glcm",
                "size": 32,
                "pca": 32,
                "neighbors": 3,
            },
            {
                "name": "glcm_gasf_32_pca32_knn3",
                "kind": "imaging_knn",
                "image": "gasf",
                "feature_extractor": "glcm",
                "size": 32,
                "pca": 32,
                "neighbors": 3,
            },
            {
                "name": "glcm_rp_32_pca32_knn3",
                "kind": "imaging_knn",
                "image": "rp",
                "feature_extractor": "glcm",
                "size": 32,
                "pca": 32,
                "neighbors": 3,
            },
            {
                "name": "glcm_fusion_spectrogram_gasf_rp_32_pca32_knn3",
                "kind": "imaging_knn",
                "image": ["spectrogram", "gasf", "rp"],
                "feature_extractor": "glcm",
                "size": 32,
                "pca": 32,
                "neighbors": 3,
            },
        ],
    },
    "experiment_57_imaging_small_cnn_mps_probe": {
        "label": "Exp 57 - Imaging small CNN MPS normal autoencoder probe",
        "source_experiment_id": "time_series_imaging_small_cnn_mps_probe",
        "target": {"scope": "full_original", "families": None, "min_train": 0},
        "thresholds": ["count_cap_2pct", "count_cap_3pct", "family_guard_v1"],
        "configs": [
            {
                "name": "cnn_ae_spectrogram_32_mps",
                "kind": "imaging_cnn_autoencoder",
                "image": "spectrogram",
                "size": 32,
                "epochs": 8,
                "batch_size": 128,
            },
            {
                "name": "cnn_ae_gasf_32_mps",
                "kind": "imaging_cnn_autoencoder",
                "image": "gasf",
                "size": 32,
                "epochs": 8,
                "batch_size": 128,
            },
            {
                "name": "cnn_ae_rp_32_mps",
                "kind": "imaging_cnn_autoencoder",
                "image": "rp",
                "size": 32,
                "epochs": 8,
                "batch_size": 128,
            },
            {
                "name": "cnn_ae_fusion_spectrogram_gasf_rp_32_mps",
                "kind": "imaging_cnn_autoencoder",
                "image": ["spectrogram", "gasf", "rp"],
                "size": 32,
                "epochs": 8,
                "batch_size": 128,
            },
        ],
    },
    "experiment_58_imaging_pretrained_cnn_feature_probe": {
        "label": "Exp 58 - Imaging pretrained CNN feature probe",
        "source_experiment_id": "time_series_imaging_pretrained_cnn_feature_probe",
        "target": {"scope": "full_original", "families": None, "min_train": 0},
        "thresholds": ["count_cap_2pct", "count_cap_3pct", "family_guard_v1"],
        "configs": [
            {
                "name": "pretrained_resnet18_spectrogram_32_pca64_knn3",
                "kind": "imaging_pretrained_cnn_knn",
                "image": "spectrogram",
                "size": 32,
                "pca": 64,
                "neighbors": 3,
                "batch_size": 96,
            },
            {
                "name": "pretrained_resnet18_rp_32_pca64_knn3",
                "kind": "imaging_pretrained_cnn_knn",
                "image": "rp",
                "size": 32,
                "pca": 64,
                "neighbors": 3,
                "batch_size": 96,
            },
            {
                "name": "pretrained_resnet18_fusion_spectrogram_gasf_rp_32_pca64_knn3",
                "kind": "imaging_pretrained_cnn_knn",
                "image": ["spectrogram", "gasf", "rp"],
                "size": 32,
                "pca": 64,
                "neighbors": 3,
                "batch_size": 96,
            },
        ],
    },
    "experiment_108_imaging_pretrained_vit_feature_probe": {
        "label": "Exp 108 - Imaging pretrained ViT hard-subset feature probe",
        "source_experiment_id": "time_series_imaging_pretrained_vit_feature_probe",
        "target": {"scope": "full_original", "families": HARD_SCORE_FAMILIES, "min_train": 0},
        "thresholds": ["count_cap_2pct", "count_cap_3pct", "family_guard_v1"],
        "configs": [
            {
                "name": "pretrained_vit_b32_spectrogram_32_pca64_knn3",
                "kind": "imaging_pretrained_vit_knn",
                "vit_arch": "vit_b_32",
                "image": "spectrogram",
                "size": 32,
                "pca": 64,
                "neighbors": 3,
                "batch_size": 32,
            },
            {
                "name": "pretrained_vit_b32_rp_32_pca64_knn3",
                "kind": "imaging_pretrained_vit_knn",
                "vit_arch": "vit_b_32",
                "image": "rp",
                "size": 32,
                "pca": 64,
                "neighbors": 3,
                "batch_size": 32,
            },
            {
                "name": "pretrained_vit_b32_fusion_spectrogram_gasf_rp_32_pca64_knn3",
                "kind": "imaging_pretrained_vit_knn",
                "vit_arch": "vit_b_32",
                "image": ["spectrogram", "gasf", "rp"],
                "size": 32,
                "pca": 64,
                "neighbors": 3,
                "batch_size": 32,
            },
        ],
    },
    "experiment_109_vit_compression_alternatives": {
        "label": "Exp 109 - ViT compression alternatives hard-subset probe",
        "source_experiment_id": "vit_compression_alternatives",
        "target": {"scope": "full_original", "families": HARD_SCORE_FAMILIES, "min_train": 0},
        "thresholds": ["count_cap_2pct", "count_cap_3pct", "family_guard_v1"],
        "configs": [
            {
                "name": "vit_spectrogram_pca64_knn3",
                "kind": "imaging_pretrained_vit_knn",
                "vit_arch": "vit_b_32",
                "image": "spectrogram",
                "size": 32,
                "compression": "pca",
                "components": 64,
                "neighbors": 3,
                "batch_size": 32,
            },
            {
                "name": "vit_spectrogram_pca128_knn3",
                "kind": "imaging_pretrained_vit_knn",
                "vit_arch": "vit_b_32",
                "image": "spectrogram",
                "size": 32,
                "compression": "pca",
                "components": 128,
                "neighbors": 3,
                "batch_size": 32,
            },
            {
                "name": "vit_spectrogram_gaussian_rp64_knn3",
                "kind": "imaging_pretrained_vit_knn",
                "vit_arch": "vit_b_32",
                "image": "spectrogram",
                "size": 32,
                "compression": "gaussian_rp",
                "components": 64,
                "neighbors": 3,
                "batch_size": 32,
            },
            {
                "name": "vit_spectrogram_sparse_rp64_knn3",
                "kind": "imaging_pretrained_vit_knn",
                "vit_arch": "vit_b_32",
                "image": "spectrogram",
                "size": 32,
                "compression": "sparse_rp",
                "components": 64,
                "neighbors": 3,
                "batch_size": 32,
            },
            {
                "name": "vit_spectrogram_svd64_knn3",
                "kind": "imaging_pretrained_vit_knn",
                "vit_arch": "vit_b_32",
                "image": "spectrogram",
                "size": 32,
                "compression": "svd",
                "components": 64,
                "neighbors": 3,
                "batch_size": 32,
            },
            {
                "name": "vit_spectrogram_stability_select64_knn3",
                "kind": "imaging_pretrained_vit_knn",
                "vit_arch": "vit_b_32",
                "image": "spectrogram",
                "size": 32,
                "compression": "stability_select",
                "components": 64,
                "neighbors": 3,
                "batch_size": 32,
            },
            {
                "name": "vit_spectrogram_ae64_knn3",
                "kind": "imaging_pretrained_vit_knn",
                "vit_arch": "vit_b_32",
                "image": "spectrogram",
                "size": 32,
                "compression": "ae",
                "components": 64,
                "ae_epochs": 20,
                "neighbors": 3,
                "batch_size": 32,
            },
            {
                "name": "vit_spectrogram_umap_adaptive_up_to64_knn3",
                "kind": "imaging_pretrained_vit_knn",
                "vit_arch": "vit_b_32",
                "image": "spectrogram",
                "size": 32,
                "compression": "umap",
                "components": 64,
                "umap_neighbors": 10,
                "neighbors": 3,
                "batch_size": 32,
            },
            {
                "name": "vit_rp_pca64_knn3",
                "kind": "imaging_pretrained_vit_knn",
                "vit_arch": "vit_b_32",
                "image": "rp",
                "size": 32,
                "compression": "pca",
                "components": 64,
                "neighbors": 3,
                "batch_size": 32,
            },
            {
                "name": "vit_rp_gaussian_rp64_knn3",
                "kind": "imaging_pretrained_vit_knn",
                "vit_arch": "vit_b_32",
                "image": "rp",
                "size": 32,
                "compression": "gaussian_rp",
                "components": 64,
                "neighbors": 3,
                "batch_size": 32,
            },
            {
                "name": "vit_rp_stability_select64_knn3",
                "kind": "imaging_pretrained_vit_knn",
                "vit_arch": "vit_b_32",
                "image": "rp",
                "size": 32,
                "compression": "stability_select",
                "components": 64,
                "neighbors": 3,
                "batch_size": 32,
            },
            {
                "name": "vit_rp_ae64_knn3",
                "kind": "imaging_pretrained_vit_knn",
                "vit_arch": "vit_b_32",
                "image": "rp",
                "size": 32,
                "compression": "ae",
                "components": 64,
                "ae_epochs": 20,
                "neighbors": 3,
                "batch_size": 32,
            },
            {
                "name": "vit_rp_umap_adaptive_up_to64_knn3",
                "kind": "imaging_pretrained_vit_knn",
                "vit_arch": "vit_b_32",
                "image": "rp",
                "size": 32,
                "compression": "umap",
                "components": 64,
                "umap_neighbors": 10,
                "neighbors": 3,
                "batch_size": 32,
            },
        ],
    },
    "experiment_110_exp84_score_backend_probe": {
        "label": "Exp 110 - Exp84 train-only score backend probe",
        "source_experiment_id": "exp84_score_backend_probe",
        "target": {
            "scope": "full_original",
            "families": HARD_SCORE_FAMILIES,
            "min_train": 0,
            "train_count_stratified_total": 90,
        },
        "thresholds": ["count_cap_2pct", "count_cap_3pct", "family_guard_v1"],
        "configs": [
            {
                "name": "exp84_local_gap_knn3_baseline",
                "kind": "aeon_multirocket_hydra",
                "num_kernels": 1024,
                "hydra_kernels": 4,
                "hydra_groups": 32,
                "feature_prune": "stable_tail",
                "feature_keep": 1024,
                "score_mode": "local_gap",
                "neighbors": 3,
                "random_state": 20260717,
            },
            {
                "name": "exp84_kmeans_centroid1_crossfit",
                "kind": "aeon_multirocket_hydra",
                "num_kernels": 1024,
                "hydra_kernels": 4,
                "hydra_groups": 32,
                "feature_prune": "stable_tail",
                "feature_keep": 1024,
                "score_mode": "kmeans_crossfit",
                "score_clusters": 1,
                "random_state": 20260717,
            },
            {
                "name": "exp84_kmeans_centroid3_crossfit",
                "kind": "aeon_multirocket_hydra",
                "num_kernels": 1024,
                "hydra_kernels": 4,
                "hydra_groups": 32,
                "feature_prune": "stable_tail",
                "feature_keep": 1024,
                "score_mode": "kmeans_crossfit",
                "score_clusters": 3,
                "random_state": 20260717,
            },
            {
                "name": "exp84_gmm_diag_bic1to3_crossfit",
                "kind": "aeon_multirocket_hydra",
                "num_kernels": 1024,
                "hydra_kernels": 4,
                "hydra_groups": 32,
                "feature_prune": "stable_tail",
                "feature_keep": 1024,
                "score_mode": "gmm_diag_crossfit",
                "score_clusters": 3,
                "random_state": 20260717,
            },
            {
                "name": "exp84_lof5_novelty_crossfit",
                "kind": "aeon_multirocket_hydra",
                "num_kernels": 1024,
                "hydra_kernels": 4,
                "hydra_groups": 32,
                "feature_prune": "stable_tail",
                "feature_keep": 1024,
                "score_mode": "lof_novelty_crossfit",
                "neighbors": 5,
                "random_state": 20260717,
            },
            {
                "name": "exp84_ocsvm_rbf_nu005_crossfit",
                "kind": "aeon_multirocket_hydra",
                "num_kernels": 1024,
                "hydra_kernels": 4,
                "hydra_groups": 32,
                "feature_prune": "stable_tail",
                "feature_keep": 1024,
                "score_mode": "ocsvm_crossfit",
                "ocsvm_nu": 0.05,
                "random_state": 20260717,
            },
        ],
    },
    "experiment_111_vit_fast_compression_probe": {
        "label": "Exp 111 - Cached ViT fast compression probe",
        "source_experiment_id": "vit_fast_compression_probe",
        "target": {
            "scope": "full_original",
            "families": HARD_SCORE_FAMILIES,
            "min_train": 0,
            "train_count_stratified_total": 90,
        },
        "thresholds": ["count_cap_2pct", "count_cap_3pct", "family_guard_v1"],
        "configs": [
            {
                "name": "vit_spectrogram_full768_knn3",
                "kind": "imaging_pretrained_vit_knn",
                "vit_arch": "vit_b_32",
                "image": "spectrogram",
                "size": 32,
                "compression": "none",
                "components": 768,
                "neighbors": 3,
                "batch_size": 32,
            },
            {
                "name": "vit_spectrogram_pca64_knn3",
                "kind": "imaging_pretrained_vit_knn",
                "vit_arch": "vit_b_32",
                "image": "spectrogram",
                "size": 32,
                "compression": "pca",
                "components": 64,
                "neighbors": 3,
                "batch_size": 32,
            },
            {
                "name": "vit_spectrogram_pca128_knn3",
                "kind": "imaging_pretrained_vit_knn",
                "vit_arch": "vit_b_32",
                "image": "spectrogram",
                "size": 32,
                "compression": "pca",
                "components": 128,
                "neighbors": 3,
                "batch_size": 32,
            },
            {
                "name": "vit_spectrogram_svd64_knn3",
                "kind": "imaging_pretrained_vit_knn",
                "vit_arch": "vit_b_32",
                "image": "spectrogram",
                "size": 32,
                "compression": "svd",
                "components": 64,
                "neighbors": 3,
                "batch_size": 32,
            },
            {
                "name": "vit_spectrogram_sparse_rp64_knn3",
                "kind": "imaging_pretrained_vit_knn",
                "vit_arch": "vit_b_32",
                "image": "spectrogram",
                "size": 32,
                "compression": "sparse_rp",
                "components": 64,
                "neighbors": 3,
                "batch_size": 32,
            },
            {
                "name": "vit_spectrogram_gaussian_rp64_knn3",
                "kind": "imaging_pretrained_vit_knn",
                "vit_arch": "vit_b_32",
                "image": "spectrogram",
                "size": 32,
                "compression": "gaussian_rp",
                "components": 64,
                "neighbors": 3,
                "batch_size": 32,
            },
            {
                "name": "vit_spectrogram_stability_select64_knn3",
                "kind": "imaging_pretrained_vit_knn",
                "vit_arch": "vit_b_32",
                "image": "spectrogram",
                "size": 32,
                "compression": "stability_select",
                "components": 64,
                "neighbors": 3,
                "batch_size": 32,
            },
        ],
    },
    "experiment_111b_vit_manifold_compression_probe": {
        "label": "Exp 111b - ViT nonlinear manifold compression probe",
        "source_experiment_id": "vit_manifold_compression_probe",
        "target": {
            "scope": "full_original",
            "families": HARD_SCORE_FAMILIES,
            "min_train": 0,
            "train_count_strata": [
                {"min": 20, "max": 50, "count": 15},
                {"min": 51, "max": 200, "count": 15},
                {"min": 201, "max": None, "count": 15},
            ],
        },
        "thresholds": ["count_cap_2pct", "count_cap_3pct", "family_guard_v1"],
        "configs": [
            {
                "name": "vit_spectrogram_pca64_knn3",
                "kind": "imaging_pretrained_vit_knn",
                "vit_arch": "vit_b_32",
                "image": "spectrogram",
                "size": 32,
                "compression": "pca",
                "components": 64,
                "neighbors": 3,
                "batch_size": 32,
            },
            {
                "name": "vit_spectrogram_gaussian_rp64_knn3",
                "kind": "imaging_pretrained_vit_knn",
                "vit_arch": "vit_b_32",
                "image": "spectrogram",
                "size": 32,
                "compression": "gaussian_rp",
                "components": 64,
                "neighbors": 3,
                "batch_size": 32,
            },
            {
                "name": "vit_spectrogram_umap8_nn5_seed1_knn3",
                "kind": "imaging_pretrained_vit_knn",
                "vit_arch": "vit_b_32",
                "image": "spectrogram",
                "size": 32,
                "compression": "umap",
                "components": 8,
                "umap_neighbors": 5,
                "random_state": 20260718,
                "neighbors": 3,
                "batch_size": 32,
            },
            {
                "name": "vit_spectrogram_umap8_nn5_seed2_knn3",
                "kind": "imaging_pretrained_vit_knn",
                "vit_arch": "vit_b_32",
                "image": "spectrogram",
                "size": 32,
                "compression": "umap",
                "components": 8,
                "umap_neighbors": 5,
                "random_state": 20260719,
                "neighbors": 3,
                "batch_size": 32,
            },
            {
                "name": "vit_spectrogram_umap16_nn10_knn3",
                "kind": "imaging_pretrained_vit_knn",
                "vit_arch": "vit_b_32",
                "image": "spectrogram",
                "size": 32,
                "compression": "umap",
                "components": 16,
                "umap_neighbors": 10,
                "random_state": 20260718,
                "neighbors": 3,
                "batch_size": 32,
            },
            {
                "name": "vit_spectrogram_isomap8_nn5_knn3",
                "kind": "imaging_pretrained_vit_knn",
                "vit_arch": "vit_b_32",
                "image": "spectrogram",
                "size": 32,
                "compression": "isomap",
                "components": 8,
                "manifold_neighbors": 5,
                "neighbors": 3,
                "batch_size": 32,
            },
            {
                "name": "vit_spectrogram_isomap16_nn10_knn3",
                "kind": "imaging_pretrained_vit_knn",
                "vit_arch": "vit_b_32",
                "image": "spectrogram",
                "size": 32,
                "compression": "isomap",
                "components": 16,
                "manifold_neighbors": 10,
                "neighbors": 3,
                "batch_size": 32,
            },
            {
                "name": "vit_spectrogram_kernel_pca_rbf16_knn3",
                "kind": "imaging_pretrained_vit_knn",
                "vit_arch": "vit_b_32",
                "image": "spectrogram",
                "size": 32,
                "compression": "kernel_pca_rbf",
                "components": 16,
                "neighbors": 3,
                "batch_size": 32,
            },
            {
                "name": "vit_spectrogram_lle8_nn10_knn3",
                "kind": "imaging_pretrained_vit_knn",
                "vit_arch": "vit_b_32",
                "image": "spectrogram",
                "size": 32,
                "compression": "lle",
                "components": 8,
                "manifold_neighbors": 10,
                "neighbors": 3,
                "batch_size": 32,
            },
        ],
    },
    "experiment_76_spectral_derivative_rocket_hard_family": {
        "label": "Exp 76 - Spectral/derivative ROCKET hard-family score",
        "source_experiment_id": "spectral_derivative_rocket_hard_family",
        "target": {"scope": "full_original", "families": HARD_SCORE_FAMILIES, "min_train": 0},
        "thresholds": ["count_cap_1pct", "count_cap_2pct", "count_cap_3pct", "family_guard_v1"],
        "configs": [
            {
                "name": "fft_band_48_knn3",
                "kind": "frequency_knn",
                "n_bands": 48,
                "neighbors": 3,
            },
            {
                "name": "raw_diff_fft_rank_ensemble_256_knn3",
                "kind": "frequency_rocket_rank_ensemble",
                "num_kernels": 256,
                "neighbors": 3,
                "n_bands": 32,
            },
            {
                "name": "bandpass_rocket_rank_ensemble_192_knn3",
                "kind": "bandpass_rocket_rank_ensemble",
                "num_kernels": 192,
                "neighbors": 3,
                "bands": [(0.0, 0.12), (0.12, 0.32), (0.32, 0.62), (0.62, 1.0)],
            },
        ],
    },
    "experiment_77_shapelet_prototype_low_train_guard": {
        "label": "Exp 77 - Shapelet/prototype hard-family score",
        "source_experiment_id": "shapelet_prototype_low_train_guard",
        "target": {"scope": "full_original", "families": HARD_SCORE_FAMILIES, "min_train": 0},
        "thresholds": ["count_cap_1pct", "count_cap_2pct", "count_cap_3pct", "family_guard_v1"],
        "configs": [
            {
                "name": "shapelet_proto_32_len8_16_32_knn3",
                "kind": "shapelet_prototype_knn",
                "num_shapelets": 32,
                "shapelet_lengths": [8, 16, 32],
                "neighbors": 3,
                "max_len": 512,
            },
            {
                "name": "shapelet_proto_48_len16_32_64_knn3",
                "kind": "shapelet_prototype_knn",
                "num_shapelets": 48,
                "shapelet_lengths": [16, 32, 64],
                "neighbors": 3,
                "max_len": 768,
            },
        ],
    },
    "experiment_78_robust_density_corrected_score": {
        "label": "Exp 78 - Robust density-corrected hard-family score",
        "source_experiment_id": "robust_density_corrected_score",
        "target": {"scope": "full_original", "families": HARD_SCORE_FAMILIES, "min_train": 0},
        "thresholds": ["count_cap_1pct", "count_cap_2pct", "count_cap_3pct", "family_guard_v1"],
        "configs": [
            {
                "name": "rocket_256_local_gap_ratio_rank_knn3",
                "kind": "rocket_density_corrected",
                "num_kernels": 256,
                "neighbors": 3,
            },
            {
                "name": "rocket_512_local_gap_ratio_rank_knn5",
                "kind": "rocket_density_corrected",
                "num_kernels": 512,
                "neighbors": 5,
            },
        ],
    },
    "experiment_80_train_only_score_selector": {
        "label": "Exp 80 - Train-only score-source selector",
        "source_experiment_id": "train_only_score_selector",
        "target": {"scope": "full_original", "families": HARD_SCORE_FAMILIES, "min_train": 0},
        "thresholds": ["count_cap_1pct", "count_cap_2pct", "count_cap_3pct", "family_guard_v1"],
        "configs": [
            {
                "name": "train_stability_rank_ensemble_rocket_fft_shapelet",
                "kind": "train_stability_rank_ensemble",
                "num_kernels": 192,
                "neighbors": 3,
                "n_bands": 32,
                "num_shapelets": 24,
                "shapelet_lengths": [16, 32],
                "max_len": 512,
            },
        ],
    },
    "experiment_81_aeon_multirocket_official_full": {
        "label": "Exp 81 - aeon official MultiROCKET feature extractor full sweep",
        "source_experiment_id": "aeon_official_multirocket_feature_extractor",
        "target": {"scope": "full_original", "families": None, "min_train": 0},
        "thresholds": ["count_cap_2pct", "count_cap_3pct", "family_guard_v1"],
        "configs": [
            {
                "name": "aeon_multirocket_1024_local_gap_knn3",
                "kind": "aeon_multirocket",
                "num_kernels": 1024,
                "neighbors": 3,
                "score_mode": "local_gap",
                "random_state": 20260709,
            },
            {
                "name": "aeon_multirocket_2048_local_gap_knn3",
                "kind": "aeon_multirocket",
                "num_kernels": 2048,
                "neighbors": 3,
                "score_mode": "local_gap",
                "random_state": 20260710,
            },
        ],
    },
    "experiment_82_hydra_hard_family_subset": {
        "label": "Exp 82 - HYDRA hard-family subset feature extractor",
        "source_experiment_id": "aeon_hydra_hard_family_feature_extractor",
        "target": {"scope": "full_original", "families": HYDRA_PRIORITY_FAMILIES, "min_train": 0},
        "thresholds": ["count_cap_1pct", "count_cap_2pct", "count_cap_3pct", "family_guard_v1"],
        "configs": [
            {
                "name": "aeon_hydra_k4_g32_local_gap_knn3",
                "kind": "aeon_hydra",
                "hydra_kernels": 4,
                "hydra_groups": 32,
                "neighbors": 3,
                "score_mode": "local_gap",
                "random_state": 20260711,
            },
            {
                "name": "aeon_hydra_k8_g64_local_gap_knn3",
                "kind": "aeon_hydra",
                "hydra_kernels": 8,
                "hydra_groups": 64,
                "neighbors": 3,
                "score_mode": "local_gap",
                "random_state": 20260712,
            },
        ],
    },
    "experiment_83_multirocket_hydra_hard339": {
        "label": "Exp 83 - MultiROCKET+HYDRA feature extractor on Exp75 hard subset",
        "source_experiment_id": "aeon_multirocket_hydra_hard339_feature_extractor",
        "target": {"scope": "full_original", "families": HARD_SCORE_FAMILIES, "min_train": 0},
        "thresholds": ["count_cap_1pct", "count_cap_2pct", "count_cap_3pct", "family_guard_v1"],
        "configs": [
            {
                "name": "aeon_mrh_mr1024_hk4_g32_local_gap_knn3",
                "kind": "aeon_multirocket_hydra",
                "num_kernels": 1024,
                "hydra_kernels": 4,
                "hydra_groups": 32,
                "neighbors": 3,
                "score_mode": "local_gap",
                "random_state": 20260713,
            },
            {
                "name": "aeon_mrh_mr2048_hk8_g32_local_gap_knn3",
                "kind": "aeon_multirocket_hydra",
                "num_kernels": 2048,
                "hydra_kernels": 8,
                "hydra_groups": 32,
                "neighbors": 3,
                "score_mode": "local_gap",
                "random_state": 20260714,
            },
        ],
    },
    "experiment_84_feature_pruning_operational_stability": {
        "label": "Exp 84 - Feature pruning for FP and operational stability",
        "source_experiment_id": "aeon_feature_pruning_operational_stability",
        "target": {"scope": "full_original", "families": HARD_SCORE_FAMILIES, "min_train": 0},
        "thresholds": ["count_cap_1pct", "count_cap_2pct", "count_cap_3pct", "family_guard_v1"],
        "configs": [
            {
                "name": "aeon_multirocket_2048_prune512_stable_tail_local_gap_knn3",
                "kind": "aeon_multirocket",
                "num_kernels": 2048,
                "neighbors": 3,
                "score_mode": "local_gap",
                "feature_prune": "stable_tail",
                "feature_keep": 512,
                "random_state": 20260715,
            },
            {
                "name": "aeon_multirocket_2048_prune1024_stable_tail_local_gap_knn3",
                "kind": "aeon_multirocket",
                "num_kernels": 2048,
                "neighbors": 3,
                "score_mode": "local_gap",
                "feature_prune": "stable_tail",
                "feature_keep": 1024,
                "random_state": 20260716,
            },
            {
                "name": "aeon_mrh_mr1024_hk4_g32_prune1024_stable_tail_local_gap_knn3",
                "kind": "aeon_multirocket_hydra",
                "num_kernels": 1024,
                "hydra_kernels": 4,
                "hydra_groups": 32,
                "neighbors": 3,
                "score_mode": "local_gap",
                "feature_prune": "stable_tail",
                "feature_keep": 1024,
                "random_state": 20260717,
            },
        ],
    },
    "experiment_87_exp84_index_diagnostics": {
        "label": "Exp 87 - Exp84 selected-index diagnostics for true agreement",
        "source_experiment_id": "aeon_exp84_index_diagnostics",
        "target": {"scope": "full_original", "families": HARD_SCORE_FAMILIES, "min_train": 0},
        "thresholds": ["count_cap_2pct", "count_cap_3pct", "family_guard_v1"],
        "configs": [
            {
                "name": "aeon_mrh_mr1024_hk4_g32_prune1024_stable_tail_local_gap_knn3",
                "kind": "aeon_multirocket_hydra",
                "num_kernels": 1024,
                "hydra_kernels": 4,
                "hydra_groups": 32,
                "neighbors": 3,
                "score_mode": "local_gap",
                "feature_prune": "stable_tail",
                "feature_keep": 1024,
                "random_state": 20260717,
            },
        ],
    },
}


def experiment_paths(exp_id):
    return {
        "detail": DATA_DIR / f"{exp_id}_results.csv",
        "summary": DATA_DIR / f"{exp_id}_summary.csv",
        "log": DATA_DIR / f"{exp_id}.log",
    }


def get_spec(exp_id):
    if exp_id not in EXPERIMENT_SPECS:
        raise SystemExit(f"Unknown model-hard research experiment: {exp_id}")
    spec = copy.deepcopy(EXPERIMENT_SPECS[exp_id])
    spec["id"] = exp_id
    spec["paths"] = experiment_paths(exp_id)
    return spec


def available_experiment_ids():
    return sorted(EXPERIMENT_SPECS)


def read_difficulty_rows(path=DIFFICULTY_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(f"Difficulty file does not exist: {path}")
    rows = []
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            if row["dataset_name"] in EXCLUDED_DATASETS:
                continue
            converted = dict(row)
            for key in [
                "difficulty_score",
                "train_count",
                "test_total_count",
                "test_anomaly_count",
                "clean_test_total_count",
            ]:
                converted[key] = int(float(row.get(key) or 0))
            for key in [
                "best_original_f1",
                "max_original_auc_pr",
                "max_original_oracle_f1",
                "best_clean_f1",
                "max_clean_auc_pr",
                "max_clean_oracle_f1",
            ]:
                converted[key] = float(row.get(key) or 0.0)
            converted["clean_balanced_eligible"] = str(row.get("clean_balanced_eligible")) == "1"
            rows.append(converted)
    return rows


def family_round_robin_sample(rows, count):
    by_family = defaultdict(list)
    for row in sorted(rows, key=lambda item: (item["family"], item["dataset_name"])):
        by_family[row["family"]].append(row)
    selected = []
    families = sorted(by_family)
    while families and len(selected) < count:
        remaining = []
        for family in families:
            if by_family[family] and len(selected) < count:
                selected.append(by_family[family].pop(0))
            if by_family[family]:
                remaining.append(family)
        families = remaining
    return selected


def train_count_stratified_sample(rows, total):
    total = max(3, int(total))
    bin_sizes = [total // 3, total // 3, total - 2 * (total // 3)]
    bins = [
        [row for row in rows if row["train_count"] <= 10],
        [row for row in rows if 11 <= row["train_count"] <= 50],
        [row for row in rows if row["train_count"] > 50],
    ]
    selected = []
    for candidates, count in zip(bins, bin_sizes):
        selected.extend(family_round_robin_sample(candidates, count))
    if len(selected) < total:
        selected_names = {row["dataset_name"] for row in selected}
        remaining = [row for row in rows if row["dataset_name"] not in selected_names]
        selected.extend(family_round_robin_sample(remaining, total - len(selected)))
    return selected[:total]


def train_count_custom_strata_sample(rows, strata):
    selected = []
    selected_names = set()
    for stratum in strata:
        lower = int(stratum.get("min", 0))
        upper = stratum.get("max")
        upper = int(upper) if upper is not None else None
        count = max(0, int(stratum.get("count", 0)))
        candidates = [
            row
            for row in rows
            if row["dataset_name"] not in selected_names
            and row["train_count"] >= lower
            and (upper is None or row["train_count"] <= upper)
        ]
        chosen = family_round_robin_sample(candidates, count)
        selected.extend(chosen)
        selected_names.update(row["dataset_name"] for row in chosen)
    return selected


def target_datasets(spec, limit=None):
    target = spec.get("target", {})
    if target.get("scope") == "full_original":
        rows = read_difficulty_rows()
    else:
        rows = [row for row in read_difficulty_rows() if row["difficulty_type"] == "model_hard"]
    families = target.get("families")
    if families:
        rows = [row for row in rows if row["family"] in set(families)]
    if target.get("scope") == "hard_core":
        rows = [
            row
            for row in rows
            if row["clean_balanced_eligible"]
            and row["max_original_auc_pr"] < 0.20
            and row["max_clean_auc_pr"] < 0.20
        ]
    min_train = int(target.get("min_train", 0))
    if min_train:
        rows = [row for row in rows if row["train_count"] >= min_train]
    custom_strata = target.get("train_count_strata")
    stratified_total = target.get("train_count_stratified_total")
    if custom_strata:
        rows = train_count_custom_strata_sample(rows, custom_strata)
        rows.sort(key=lambda row: (row["train_count"], row["family"], row["dataset_name"]))
    elif stratified_total:
        rows = train_count_stratified_sample(rows, stratified_total)
        rows.sort(key=lambda row: (row["train_count"], row["family"], row["dataset_name"]))
    else:
        rows.sort(key=lambda row: (-row["difficulty_score"], row["family"], row["dataset_name"]))
    if limit is not None:
        rows = rows[:limit]
    return rows


def resample_rows(X, target_len):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("resample_rows expects a 2D array")
    target_len = max(8, int(target_len))
    if X.shape[1] == target_len:
        return X.astype(np.float32, copy=False)
    old_x = np.linspace(0.0, 1.0, X.shape[1], dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    return np.vstack([np.interp(new_x, old_x, row).astype(np.float32) for row in X])


def fixed_width_features(features, width):
    features = np.asarray(features, dtype=np.float32)
    width = max(1, int(width))
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if features.shape[1] == width:
        return features.astype(np.float32, copy=False)
    if features.shape[1] > width:
        return features[:, :width].astype(np.float32, copy=False)
    pad = np.zeros((features.shape[0], width - features.shape[1]), dtype=np.float32)
    return np.hstack([features, pad]).astype(np.float32)


def pca_pair(X_train, X_test, n_components):
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    n_components = max(1, int(n_components))
    center = X_train.mean(axis=0, keepdims=True)
    train_centered = X_train - center
    test_centered = X_test - center
    effective = max(1, min(n_components, train_centered.shape[0], train_centered.shape[1]))
    large_matrix = train_centered.shape[0] * train_centered.shape[1] >= 1_000_000
    if large_matrix and effective < min(train_centered.shape):
        try:
            pca = PCA(n_components=effective, svd_solver="randomized", random_state=RNG_SEED)
            return (
                fixed_width_features(pca.fit_transform(train_centered), n_components),
                fixed_width_features(pca.transform(test_centered), n_components),
            )
        except Exception:
            logging.getLogger("model_hard_research").exception("Randomized PCA failed; falling back to SVD.")
    try:
        _, _, vt = np.linalg.svd(train_centered, full_matrices=False)
        components = vt[:effective].T
    except np.linalg.LinAlgError:
        components = np.eye(train_centered.shape[1], effective, dtype=np.float32)
    return (
        fixed_width_features(train_centered @ components, n_components),
        fixed_width_features(test_centered @ components, n_components),
    )


def random_projection_pair(X_train, X_test, n_components, sparse=False):
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    n_components = max(1, int(n_components))
    effective = max(1, min(n_components, X_train.shape[1]))
    projector_cls = SparseRandomProjection if sparse else GaussianRandomProjection
    projector = projector_cls(n_components=effective, random_state=RNG_SEED)
    return (
        fixed_width_features(projector.fit_transform(X_train), n_components),
        fixed_width_features(projector.transform(X_test), n_components),
    )


def svd_pair(X_train, X_test, n_components):
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    n_components = max(1, int(n_components))
    if X_train.shape[0] < 2 or X_train.shape[1] < 2:
        return fixed_width_features(X_train, n_components), fixed_width_features(X_test, n_components)
    effective = min(n_components, X_train.shape[0] - 1, X_train.shape[1] - 1)
    svd = TruncatedSVD(n_components=effective, random_state=RNG_SEED)
    return (
        fixed_width_features(svd.fit_transform(X_train), n_components),
        fixed_width_features(svd.transform(X_test), n_components),
    )


def stability_select_pair(X_train, X_test, n_components):
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    n_components = max(1, int(n_components))
    median = np.median(X_train, axis=0, keepdims=True)
    mad = np.median(np.abs(X_train - median), axis=0) + 1e-6
    spread = np.percentile(X_train, 90, axis=0) - np.percentile(X_train, 10, axis=0)
    score = spread / mad
    order = np.argsort(score)[::-1]
    selected = order[: min(n_components, X_train.shape[1])]
    return fixed_width_features(X_train[:, selected], n_components), fixed_width_features(X_test[:, selected], n_components)


def umap_pair(X_train, X_test, config):
    try:
        import umap
    except Exception as exc:
        raise RuntimeError("umap-learn is required for UMAP compression experiments.") from exc

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    config["_compression_effective_method"] = "umap"
    requested_components = max(2, int(config.get("components", 64)))
    n_train = int(X_train.shape[0])
    if n_train < 3:
        # UMAP cannot build a meaningful neighborhood graph from fewer than
        # three normal samples. Preserve a deterministic train-only fallback.
        return pca_pair(X_train, X_test, requested_components)

    # Spectral initialization solves for n_components + 1 eigenvectors, which
    # must remain strictly smaller than the number of training samples.
    effective_components = max(1, min(requested_components, n_train - 2))
    n_neighbors = max(2, min(int(config.get("umap_neighbors", 10)), n_train - 1))
    config["_manifold_neighbors_effective"] = n_neighbors
    random_state = int(config.get("random_state", RNG_SEED))
    reducer = umap.UMAP(
        n_components=effective_components,
        n_neighbors=n_neighbors,
        min_dist=float(config.get("umap_min_dist", 0.1)),
        metric=str(config.get("umap_metric", "euclidean")),
        random_state=random_state,
        transform_seed=random_state,
        n_jobs=1,
    )
    return (
        fixed_width_features(reducer.fit_transform(X_train), requested_components),
        fixed_width_features(reducer.transform(X_test), requested_components),
    )


def isomap_pair(X_train, X_test, config):
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    config["_compression_effective_method"] = "isomap"
    config.pop("_manifold_neighbors_effective", None)
    requested_components = max(1, int(config.get("components", 8)))
    n_train = len(X_train)
    if n_train < 3:
        return pca_pair(X_train, X_test, requested_components)
    effective_components = max(1, min(requested_components, n_train - 2, X_train.shape[1]))
    requested_neighbors = max(2, min(int(config.get("manifold_neighbors", 5)), n_train - 1))
    neighbor_attempts = [
        requested_neighbors,
        min(n_train - 1, max(requested_neighbors, effective_components + 2)),
        max(2, n_train - 2),
        n_train - 1,
    ]
    last_error = None
    for n_neighbors in dict.fromkeys(neighbor_attempts):
        try:
            reducer = Isomap(
                n_neighbors=n_neighbors,
                n_components=effective_components,
                eigen_solver="arpack",
                path_method="auto",
                neighbors_algorithm="auto",
                n_jobs=1,
            )
            train_embedding = reducer.fit_transform(X_train)
            test_embedding = reducer.transform(X_test)
            config["_manifold_neighbors_effective"] = n_neighbors
            return (
                fixed_width_features(train_embedding, requested_components),
                fixed_width_features(test_embedding, requested_components),
            )
        except ValueError as exc:
            last_error = exc
    config["_compression_effective_method"] = "pca_fallback"
    return pca_pair(X_train, X_test, requested_components)


def kernel_pca_pair(X_train, X_test, config):
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    config["_compression_effective_method"] = "kernel_pca_rbf"
    requested_components = max(1, int(config.get("components", 16)))
    n_train = len(X_train)
    if n_train < 3:
        return pca_pair(X_train, X_test, requested_components)
    effective_components = max(1, min(requested_components, n_train - 1))
    sample_count = min(512, n_train)
    sample_idx = np.linspace(0, n_train - 1, sample_count, dtype=int)
    distances = pairwise_distances(X_train[sample_idx], metric="euclidean")
    nonzero = distances[distances > 1e-12]
    median_distance = float(np.median(nonzero)) if len(nonzero) else 1.0
    gamma = 1.0 / max(2.0 * median_distance * median_distance, 1e-12)
    reducer = KernelPCA(
        n_components=effective_components,
        kernel="rbf",
        gamma=gamma,
        fit_inverse_transform=False,
        eigen_solver="randomized",
        remove_zero_eig=True,
        random_state=int(config.get("random_state", RNG_SEED)),
        n_jobs=1,
    )
    try:
        return (
            fixed_width_features(reducer.fit_transform(X_train), requested_components),
            fixed_width_features(reducer.transform(X_test), requested_components),
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        config["_compression_effective_method"] = "pca_fallback"
        return pca_pair(X_train, X_test, requested_components)


def lle_pair(X_train, X_test, config):
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    config["_compression_effective_method"] = "lle"
    config.pop("_manifold_neighbors_effective", None)
    config.pop("_lle_eigen_solver_effective", None)
    config.pop("_lle_reg_effective", None)
    requested_components = max(1, int(config.get("components", 8)))
    n_train = len(X_train)
    if n_train < 4:
        return pca_pair(X_train, X_test, requested_components)
    effective_components = max(1, min(requested_components, n_train - 2, X_train.shape[1]))
    n_neighbors = max(
        effective_components + 1,
        min(int(config.get("manifold_neighbors", 10)), n_train - 1),
    )
    n_neighbors = min(n_neighbors, n_train - 1)
    config["_manifold_neighbors_effective"] = n_neighbors
    base_reg = float(config.get("lle_reg", 1e-3))
    attempts = [("arpack", base_reg), ("arpack", base_reg * 10.0)]
    if n_train <= 512:
        attempts.append(("dense", base_reg * 10.0))
    for eigen_solver, reg in attempts:
        try:
            reducer = LocallyLinearEmbedding(
                n_neighbors=n_neighbors,
                n_components=effective_components,
                reg=reg,
                eigen_solver=eigen_solver,
                method="standard",
                random_state=int(config.get("random_state", RNG_SEED)),
                n_jobs=1,
            )
            train_embedding = reducer.fit_transform(X_train)
            test_embedding = reducer.transform(X_test)
            config["_lle_eigen_solver_effective"] = eigen_solver
            config["_lle_reg_effective"] = reg
            return (
                fixed_width_features(train_embedding, requested_components),
                fixed_width_features(test_embedding, requested_components),
            )
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError):
            continue
    config["_compression_effective_method"] = "pca_fallback"
    return pca_pair(X_train, X_test, requested_components)


def ae_compress_pair(X_train, X_test, config):
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise RuntimeError("torch is required for autoencoder compression experiments.") from exc

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    n_components = max(2, int(config.get("components", 64)))
    input_dim = int(X_train.shape[1])
    hidden_dim = max(n_components * 2, min(256, input_dim))
    epochs = max(1, int(config.get("ae_epochs", 20)))
    batch_size = max(8, int(config.get("ae_batch_size", 64)))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    class EmbeddingAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_components),
            )
            self.decoder = nn.Sequential(
                nn.Linear(n_components, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, batch):
            latent = self.encoder(batch)
            return self.decoder(latent)

    torch.manual_seed(RNG_SEED)
    center = X_train.mean(axis=0, keepdims=True)
    scale = X_train.std(axis=0, keepdims=True) + 1e-6
    train_scaled = ((X_train - center) / scale).astype(np.float32)
    test_scaled = ((X_test - center) / scale).astype(np.float32)
    model = EmbeddingAutoencoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("ae_lr", 1e-3)), weight_decay=1e-4)
    data = torch.from_numpy(train_scaled)
    model.train()
    for epoch in range(epochs):
        generator = torch.Generator().manual_seed(RNG_SEED + epoch)
        indices = torch.randperm(len(data), generator=generator)
        for start in range(0, len(indices), batch_size):
            batch = data[indices[start : start + batch_size]].to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = torch.mean((recon - batch) ** 2)
            loss.backward()
            optimizer.step()

    def encode(array):
        out = []
        model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(array)
            for start in range(0, len(tensor), batch_size):
                batch = tensor[start : start + batch_size].to(device)
                out.append(model.encoder(batch).detach().cpu().numpy().astype(np.float32))
        return np.vstack(out).astype(np.float32)

    return fixed_width_features(encode(train_scaled), n_components), fixed_width_features(encode(test_scaled), n_components)


def compress_feature_pair(X_train, X_test, config):
    method = str(config.get("compression", "pca"))
    n_components = int(config.get("components", config.get("pca", 64)))
    if method == "none":
        return fixed_width_features(X_train, n_components), fixed_width_features(X_test, n_components)
    if method == "pca":
        return pca_pair(X_train, X_test, n_components)
    if method == "svd":
        return svd_pair(X_train, X_test, n_components)
    if method == "gaussian_rp":
        return random_projection_pair(X_train, X_test, n_components, sparse=False)
    if method == "sparse_rp":
        return random_projection_pair(X_train, X_test, n_components, sparse=True)
    if method == "stability_select":
        return stability_select_pair(X_train, X_test, n_components)
    if method == "umap":
        return umap_pair(X_train, X_test, config)
    if method == "isomap":
        return isomap_pair(X_train, X_test, config)
    if method == "kernel_pca_rbf":
        return kernel_pca_pair(X_train, X_test, config)
    if method == "lle":
        return lle_pair(X_train, X_test, config)
    if method == "ae":
        return ae_compress_pair(X_train, X_test, config)
    raise ValueError(f"Unknown feature compression method: {method}")


def scale_feature_pair(train_features, test_features):
    scaler = RobustScaler(quantile_range=(10, 90))
    return (
        scaler.fit_transform(train_features).astype(np.float32),
        scaler.transform(test_features).astype(np.float32),
    )


def finite_score_vector(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    fill = float(np.max(finite)) if len(finite) else 0.0
    return np.nan_to_num(values, nan=fill, posinf=fill, neginf=0.0)


def centroid_fold_scores(X_fit, X_query, X_test, clusters=1, random_state=RNG_SEED):
    clusters = max(1, min(int(clusters), len(X_fit)))
    if clusters == 1:
        centers = np.mean(X_fit, axis=0, keepdims=True)
        query_scores = np.linalg.norm(X_query[:, None, :] - centers[None, :, :], axis=2).min(axis=1)
        test_scores = np.linalg.norm(X_test[:, None, :] - centers[None, :, :], axis=2).min(axis=1)
    else:
        model = KMeans(n_clusters=clusters, n_init="auto", random_state=int(random_state))
        model.fit(X_fit)
        query_scores = model.transform(X_query).min(axis=1)
        test_scores = model.transform(X_test).min(axis=1)
    return finite_score_vector(query_scores), finite_score_vector(test_scores)


def cross_fitted_score_pair(X_train, X_test, fold_scorer, random_state=RNG_SEED):
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    n_train = len(X_train)
    if n_train < 2:
        return centroid_fold_scores(X_train, X_train, X_test, clusters=1, random_state=random_state)

    train_scores = np.zeros(n_train, dtype=np.float64)
    test_fold_scores = []
    splitter = KFold(n_splits=min(5, n_train), shuffle=True, random_state=int(random_state))
    for fold_index, (fit_idx, heldout_idx) in enumerate(splitter.split(X_train)):
        X_fit = X_train[fit_idx]
        X_heldout = X_train[heldout_idx]
        try:
            heldout_scores, test_scores = fold_scorer(X_fit, X_heldout, X_test, fold_index)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            heldout_scores, test_scores = centroid_fold_scores(
                X_fit,
                X_heldout,
                X_test,
                clusters=1,
                random_state=int(random_state) + fold_index,
            )
        train_scores[heldout_idx] = finite_score_vector(heldout_scores)
        test_fold_scores.append(finite_score_vector(test_scores))
    return finite_score_vector(train_scores), finite_score_vector(np.median(np.vstack(test_fold_scores), axis=0))


def kmeans_crossfit_score_pair(X_train, X_test, config):
    clusters = int(config.get("score_clusters", 1))
    seed = int(config.get("random_state", RNG_SEED))

    def score_fold(X_fit, X_query, X_eval, fold_index):
        return centroid_fold_scores(X_fit, X_query, X_eval, clusters=clusters, random_state=seed + fold_index)

    return cross_fitted_score_pair(X_train, X_test, score_fold, random_state=seed)


def gmm_crossfit_score_pair(X_train, X_test, config):
    max_clusters = int(config.get("score_clusters", 3))
    seed = int(config.get("random_state", RNG_SEED))

    def score_fold(X_fit, X_query, X_eval, fold_index):
        if len(X_fit) < 2:
            return centroid_fold_scores(X_fit, X_query, X_eval, clusters=1, random_state=seed + fold_index)
        candidates = []
        for components in range(1, min(max_clusters, len(X_fit)) + 1):
            model = GaussianMixture(
                n_components=components,
                covariance_type="diag",
                reg_covar=1e-4,
                n_init=1,
                random_state=seed + fold_index,
            ).fit(X_fit)
            candidates.append((model.bic(X_fit), model))
        model = min(candidates, key=lambda item: item[0])[1]
        return -model.score_samples(X_query), -model.score_samples(X_eval)

    return cross_fitted_score_pair(X_train, X_test, score_fold, random_state=seed)


def lof_crossfit_score_pair(X_train, X_test, config):
    neighbors = int(config.get("neighbors", 5))
    seed = int(config.get("random_state", RNG_SEED))

    def score_fold(X_fit, X_query, X_eval, fold_index):
        if len(X_fit) < 2:
            return centroid_fold_scores(X_fit, X_query, X_eval, clusters=1, random_state=seed + fold_index)
        model = LocalOutlierFactor(
            n_neighbors=max(1, min(neighbors, len(X_fit) - 1)),
            novelty=True,
            contamination="auto",
            n_jobs=1,
        ).fit(X_fit)
        return -model.score_samples(X_query), -model.score_samples(X_eval)

    return cross_fitted_score_pair(X_train, X_test, score_fold, random_state=seed)


def ocsvm_crossfit_score_pair(X_train, X_test, config):
    seed = int(config.get("random_state", RNG_SEED))
    nu = float(config.get("ocsvm_nu", 0.05))

    def score_fold(X_fit, X_query, X_eval, fold_index):
        if len(X_fit) < 2:
            return centroid_fold_scores(X_fit, X_query, X_eval, clusters=1, random_state=seed + fold_index)
        model = OneClassSVM(kernel="rbf", gamma="scale", nu=nu, cache_size=256).fit(X_fit)
        return -model.score_samples(X_query), -model.score_samples(X_eval)

    return cross_fitted_score_pair(X_train, X_test, score_fold, random_state=seed)


def scale_feature_triplet(train_features, test_features, third_features):
    scaler = RobustScaler(quantile_range=(10, 90))
    return (
        scaler.fit_transform(train_features).astype(np.float32),
        scaler.transform(test_features).astype(np.float32),
        scaler.transform(third_features).astype(np.float32),
    )


def train_robust_scale_pair(X_train_raw, X_test_raw, clip=5.0):
    X_train_raw = np.asarray(X_train_raw, dtype=np.float32)
    X_test_raw = np.asarray(X_test_raw, dtype=np.float32)
    center = np.median(X_train_raw, axis=0, keepdims=True)
    q25 = np.percentile(X_train_raw, 25, axis=0, keepdims=True)
    q75 = np.percentile(X_train_raw, 75, axis=0, keepdims=True)
    scale = q75 - q25
    fallback = np.std(X_train_raw, axis=0, keepdims=True)
    scale = np.where(scale > 1e-6, scale, fallback)
    scale = np.where(scale > 1e-6, scale, 1.0)
    X_train = (X_train_raw - center) / scale
    X_test = (X_test_raw - center) / scale
    X_train = np.clip(X_train, -clip, clip)
    X_test = np.clip(X_test, -clip, clip)
    return X_train.astype(np.float32), X_test.astype(np.float32)


def quantile_scale_pair(X_train_raw, X_test_raw, low_q, high_q, axis):
    X_train_raw = np.asarray(X_train_raw, dtype=np.float32)
    X_test_raw = np.asarray(X_test_raw, dtype=np.float32)
    low = np.percentile(X_train_raw, low_q, axis=axis, keepdims=True)
    high = np.percentile(X_train_raw, high_q, axis=axis, keepdims=True)
    scale = np.where((high - low) > 1e-6, high - low, 1.0)
    X_train = np.clip((X_train_raw - low) / scale, 0.0, 1.0) * 2.0 - 1.0
    X_test = np.clip((X_test_raw - low) / scale, 0.0, 1.0) * 2.0 - 1.0
    return X_train.astype(np.float32), X_test.astype(np.float32)


def minmax_scale_pair(X_train_raw, X_test_raw):
    X_train_raw = np.asarray(X_train_raw, dtype=np.float32)
    X_test_raw = np.asarray(X_test_raw, dtype=np.float32)
    low = np.min(X_train_raw)
    high = np.max(X_train_raw)
    scale = high - low if high - low > 1e-6 else 1.0
    X_train = np.clip((X_train_raw - low) / scale, 0.0, 1.0) * 2.0 - 1.0
    X_test = np.clip((X_test_raw - low) / scale, 0.0, 1.0) * 2.0 - 1.0
    return X_train.astype(np.float32), X_test.astype(np.float32)


def prepare_series_pair_for_scale(mode, X_train_raw, X_test_raw, X_train_z, X_test_z):
    if mode in ("", "per_series_z"):
        return X_train_z, X_test_z
    if mode == "per_series_z_clip3":
        return np.clip(X_train_z, -3.0, 3.0), np.clip(X_test_z, -3.0, 3.0)
    if mode == "train_robust":
        return train_robust_scale_pair(X_train_raw, X_test_raw)
    if mode == "per_series_quantile_05_95":
        train = quantile_scale_pair(X_train_raw, X_train_raw, 5, 95, axis=1)[0]
        test = quantile_scale_pair(X_test_raw, X_test_raw, 5, 95, axis=1)[1]
        return train, test
    if mode == "train_point_quantile_05_95":
        return quantile_scale_pair(X_train_raw, X_test_raw, 5, 95, axis=0)
    if mode == "train_point_quantile_01_99":
        return quantile_scale_pair(X_train_raw, X_test_raw, 1, 99, axis=0)
    if mode == "train_global_quantile_05_95":
        return quantile_scale_pair(X_train_raw, X_test_raw, 5, 95, axis=None)
    if mode == "train_global_quantile_01_99":
        return quantile_scale_pair(X_train_raw, X_test_raw, 1, 99, axis=None)
    if mode == "train_global_minmax_clip":
        return minmax_scale_pair(X_train_raw, X_test_raw)
    raise ValueError(f"Unknown series_scale: {mode}")


def interval_slices(length, count):
    count = max(1, int(count))
    starts = np.linspace(0, length - 1, count, dtype=int)
    min_width = max(4, length // max(4, count))
    spans = []
    for idx, start in enumerate(starts):
        if idx == count - 1:
            end = length
        else:
            end = min(length, max(start + min_width, int(starts[idx + 1])))
        spans.append((int(start), int(max(start + 1, end))))
    spans.append((0, length))
    return spans


def summarize_segment(segment):
    segment = np.asarray(segment, dtype=np.float32)
    if segment.size == 0:
        return np.zeros(9, dtype=np.float32)
    qs = np.percentile(segment, [5, 25, 50, 75, 95]).astype(np.float32)
    x = np.linspace(-1.0, 1.0, len(segment), dtype=np.float32)
    slope = float(np.dot(segment - segment.mean(), x) / (np.dot(x, x) + 1e-9))
    return np.array(
        [
            float(segment.mean()),
            float(segment.std()),
            float(segment.min()),
            float(segment.max()),
            *qs.tolist(),
            slope,
        ],
        dtype=np.float32,
    )


def interval_quantile_features(X, config):
    X = np.asarray(X, dtype=np.float32)
    arrays = [X]
    if config.get("include_diff", True):
        arrays.append(np.diff(X, axis=1, prepend=X[:, :1]).astype(np.float32))
    if config.get("include_periodogram", False):
        periodogram = np.log1p(np.abs(np.fft.rfft(X, axis=1))).astype(np.float32)
        if periodogram.shape[1] > 1:
            periodogram = periodogram[:, 1:]
        arrays.append(periodogram)
    rows = []
    for arr in arrays:
        spans = interval_slices(arr.shape[1], config.get("intervals", 16))
        for row_idx in range(arr.shape[0]):
            if len(rows) <= row_idx:
                rows.append([])
            for start, end in spans:
                rows[row_idx].append(summarize_segment(arr[row_idx, start:end]))
    return np.vstack([np.concatenate(parts) for parts in rows]).astype(np.float32)


def fft_band_features(X, n_bands=24):
    X = np.asarray(X, dtype=np.float32)
    mag = np.abs(np.fft.rfft(X, axis=1)).astype(np.float32)
    if mag.shape[1] > 1:
        mag = mag[:, 1:]
    mag = np.log1p(mag)
    if mag.shape[1] == 0:
        return np.zeros((len(X), int(n_bands)), dtype=np.float32)
    bands = np.array_split(np.arange(mag.shape[1]), max(1, int(n_bands)))
    features = []
    for band in bands:
        if len(band) == 0:
            features.append(np.zeros(len(X), dtype=np.float32))
        else:
            features.append(mag[:, band].mean(axis=1))
    return np.vstack(features).T.astype(np.float32)


def rocket_feature_pair(X_train, X_test, seq_len, num_kernels, seed_offset=0):
    kernels = make_kernels(seq_len, num_kernels=int(num_kernels), seed=RNG_SEED + int(seed_offset))
    return rocket_transform(X_train, kernels), rocket_transform(X_test, kernels)


def aeon_collection(X):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("aeon_collection expects a 2D array")
    return X[:, np.newaxis, :].astype(np.float32, copy=False)


def numpy_feature_matrix(features):
    if hasattr(features, "detach"):
        features = features.detach().cpu().numpy()
    elif hasattr(features, "to_numpy"):
        features = features.to_numpy()
    elif hasattr(features, "toarray"):
        features = features.toarray()
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        features = features.reshape(features.shape[0], -1)
    return np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


def aeon_fit_collection(X_train):
    X_train = np.asarray(X_train, dtype=np.float32)
    if len(X_train) == 1:
        return aeon_collection(np.repeat(X_train, 2, axis=0))
    return aeon_collection(X_train)


def aeon_multirocket_features(X_train, X_test, config):
    from aeon.transformations.collection.convolution_based import MultiRocket

    transformer = MultiRocket(
        n_kernels=int(config.get("num_kernels", 1024)),
        n_jobs=int(config.get("n_jobs", AEON_N_JOBS)),
        random_state=int(config.get("random_state", RNG_SEED)),
        normalise=bool(config.get("normalise", False)),
    )
    transformer.fit(aeon_fit_collection(X_train))
    train_features = transformer.transform(aeon_collection(X_train))
    test_features = transformer.transform(aeon_collection(X_test))
    return numpy_feature_matrix(train_features), numpy_feature_matrix(test_features)


def aeon_hydra_features(X_train, X_test, config):
    from aeon.transformations.collection.convolution_based import HydraTransformer

    transformer = HydraTransformer(
        n_kernels=int(config.get("hydra_kernels", 4)),
        n_groups=int(config.get("hydra_groups", 32)),
        n_jobs=int(config.get("n_jobs", AEON_N_JOBS)),
        random_state=int(config.get("random_state", RNG_SEED)),
        output_type=config.get("output_type", "tensor"),
    )
    transformer.fit(aeon_fit_collection(X_train))
    train_features = transformer.transform(aeon_collection(X_train))
    test_features = transformer.transform(aeon_collection(X_test))
    return numpy_feature_matrix(train_features), numpy_feature_matrix(test_features)


def aeon_feature_pair(X_train, X_test, config, record):
    cache = record.setdefault("_aeon_feature_cache", {})
    key = (
        config["kind"],
        int(config.get("num_kernels", 0)),
        int(config.get("hydra_kernels", 0)),
        int(config.get("hydra_groups", 0)),
        int(config.get("random_state", RNG_SEED)),
        bool(config.get("normalise", False)),
    )
    if key in cache:
        return cache[key]
    if config["kind"] == "aeon_multirocket":
        pair = aeon_multirocket_features(X_train, X_test, config)
    elif config["kind"] == "aeon_hydra":
        pair = aeon_hydra_features(X_train, X_test, config)
    elif config["kind"] == "aeon_multirocket_hydra":
        mr_train, mr_test = aeon_multirocket_features(X_train, X_test, config)
        hydra_train, hydra_test = aeon_hydra_features(X_train, X_test, config)
        pair = (
            np.concatenate([mr_train, hydra_train], axis=1).astype(np.float32),
            np.concatenate([mr_test, hydra_test], axis=1).astype(np.float32),
        )
    else:
        raise ValueError(f"Unknown aeon feature kind: {config['kind']}")
    cache[key] = pair
    return pair


def stable_tail_feature_prune(train_features, test_features, keep):
    train_features = np.asarray(train_features, dtype=np.float32)
    test_features = np.asarray(test_features, dtype=np.float32)
    keep = int(keep)
    if keep <= 0 or train_features.shape[1] <= keep:
        return train_features, test_features
    q25 = np.percentile(train_features, 25, axis=0)
    q50 = np.percentile(train_features, 50, axis=0)
    q75 = np.percentile(train_features, 75, axis=0)
    iqr = q75 - q25
    usable = iqr > 1e-8
    if not np.any(usable):
        return train_features[:, :keep], test_features[:, :keep]
    robust = np.abs((train_features[:, usable] - q50[usable]) / (iqr[usable] + 1e-6))
    p75 = np.percentile(robust, 75, axis=0)
    p95 = np.percentile(robust, 95, axis=0)
    tail_ratio = p95 / (p75 + 1e-6)
    nonzero_rate = np.mean(np.abs(train_features[:, usable]) > 1e-8, axis=0)
    stability_score = tail_ratio + 0.25 * np.abs(nonzero_rate - 0.5)
    usable_indices = np.flatnonzero(usable)
    selected = usable_indices[np.argsort(stability_score)[: min(keep, len(usable_indices))]]
    if len(selected) < keep:
        fallback = np.setdiff1d(np.arange(train_features.shape[1]), selected, assume_unique=False)
        selected = np.concatenate([selected, fallback[: keep - len(selected)]])
    selected = np.sort(selected[:keep])
    return train_features[:, selected], test_features[:, selected]


def maybe_prune_features(train_features, test_features, config):
    if config.get("feature_prune") == "stable_tail":
        return stable_tail_feature_prune(train_features, test_features, config.get("feature_keep", 1024))
    return train_features, test_features


def bandpass_filter_rows(X, low_frac, high_frac):
    X = np.asarray(X, dtype=np.float32)
    spectrum = np.fft.rfft(X, axis=1)
    n = spectrum.shape[1]
    lo = int(np.floor(low_frac * n))
    hi = max(lo + 1, int(np.ceil(high_frac * n)))
    mask = np.zeros(n, dtype=bool)
    mask[lo:hi] = True
    filtered = np.fft.irfft(spectrum * mask, n=X.shape[1], axis=1)
    return filtered.astype(np.float32)


def combine_rank_scores(components):
    train_parts = []
    test_parts = []
    for train_scores, test_scores in components:
        train_rank, test_rank = rank_normalize_scores(train_scores, test_scores)
        train_parts.append(train_rank)
        test_parts.append(test_rank)
    return np.vstack(train_parts).mean(axis=0), np.vstack(test_parts).mean(axis=0)


def choose_shapelets(X_train, num_shapelets, lengths, seed):
    rng = np.random.default_rng(seed)
    X_train = np.asarray(X_train, dtype=np.float32)
    candidates = []
    valid_lengths = [length for length in lengths if 4 <= length <= X_train.shape[1]]
    if not valid_lengths:
        valid_lengths = [min(max(4, X_train.shape[1] // 4), X_train.shape[1])]
    for idx in range(int(num_shapelets)):
        row_idx = int(rng.integers(0, len(X_train)))
        length = int(valid_lengths[idx % len(valid_lengths)])
        start = int(rng.integers(0, max(1, X_train.shape[1] - length + 1)))
        shapelet = X_train[row_idx, start : start + length]
        shapelet = (shapelet - shapelet.mean()) / (shapelet.std() + 1e-6)
        candidates.append(shapelet.astype(np.float32))
    return candidates


def shapelet_distance_features(X, shapelets):
    X = np.asarray(X, dtype=np.float32)
    rows = np.empty((len(X), len(shapelets)), dtype=np.float32)
    for col, shapelet in enumerate(shapelets):
        length = len(shapelet)
        if X.shape[1] < length:
            rows[:, col] = np.inf
            continue
        windows = sliding_window_view(X, window_shape=length, axis=1)
        windows = (windows - windows.mean(axis=2, keepdims=True)) / (windows.std(axis=2, keepdims=True) + 1e-6)
        dist = np.sqrt(((windows - shapelet.reshape(1, 1, -1)) ** 2).mean(axis=2))
        rows[:, col] = dist.min(axis=1)
    rows[~np.isfinite(rows)] = np.nanmax(rows[np.isfinite(rows)]) if np.isfinite(rows).any() else 0.0
    return rows


def inject_anomalies(X, synthetic_per_train=1, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)
    out = []
    for rep in range(max(1, int(synthetic_per_train))):
        for row in X:
            y = row.copy()
            n = len(y)
            kind = int(rng.integers(0, 5))
            width = int(rng.integers(max(4, n // 32), max(5, n // 8 + 1)))
            start = int(rng.integers(0, max(1, n - width + 1)))
            segment = slice(start, start + width)
            if kind == 0:
                y[segment] += rng.choice([-1.0, 1.0]) * (2.5 + rng.random()) * (np.std(row) + 1e-6)
            elif kind == 1:
                y[segment] *= rng.uniform(0.2, 2.8)
            elif kind == 2:
                y[segment] += np.linspace(0.0, rng.choice([-1.0, 1.0]) * 3.0, width, dtype=np.float32)
            elif kind == 3:
                donor = X[int(rng.integers(0, len(X)))]
                donor_start = int(rng.integers(0, max(1, n - width + 1)))
                y[segment] = donor[donor_start : donor_start + width]
            else:
                y[segment] = y[segment][::-1]
            out.append(y.astype(np.float32))
    return np.vstack(out).astype(np.float32)


def interval_frequency_embedding(X):
    interval = interval_quantile_features(
        X,
        {"intervals": 12, "include_diff": True, "include_periodogram": False},
    )
    freq = fft_band_features(X, 24)
    return np.hstack([interval, freq]).astype(np.float32)


def gasf_images(X, size):
    Xs = resample_rows(X, size)
    mins = Xs.min(axis=1, keepdims=True)
    maxs = Xs.max(axis=1, keepdims=True)
    scaled = 2.0 * (Xs - mins) / (maxs - mins + 1e-6) - 1.0
    scaled = np.clip(scaled, -1.0, 1.0)
    phi = np.arccos(scaled)
    return np.cos(phi[:, :, None] + phi[:, None, :]).astype(np.float32)


def gadf_images(X, size):
    Xs = resample_rows(X, size)
    mins = Xs.min(axis=1, keepdims=True)
    maxs = Xs.max(axis=1, keepdims=True)
    scaled = 2.0 * (Xs - mins) / (maxs - mins + 1e-6) - 1.0
    scaled = np.clip(scaled, -1.0, 1.0)
    phi = np.arccos(scaled)
    return np.sin(phi[:, :, None] - phi[:, None, :]).astype(np.float32)


def mtf_images(X, size, bins=8):
    Xs = resample_rows(X, size)
    images = np.empty((len(Xs), size, size), dtype=np.float32)
    for idx, row in enumerate(Xs):
        edges = np.percentile(row, np.linspace(0, 100, bins + 1))
        edges[0] -= 1e-6
        edges[-1] += 1e-6
        states = np.clip(np.digitize(row, edges[1:-1]), 0, bins - 1)
        mat = np.ones((bins, bins), dtype=np.float32) * 1e-3
        for a, b in zip(states[:-1], states[1:]):
            mat[a, b] += 1.0
        mat /= mat.sum(axis=1, keepdims=True)
        images[idx] = mat[states[:, None], states[None, :]]
    return images


def rp_images(X, size):
    Xs = resample_rows(X, size)
    images = np.empty((len(Xs), size, size), dtype=np.float32)
    for idx, row in enumerate(Xs):
        dist = np.abs(row[:, None] - row[None, :])
        scale = np.percentile(dist, 75) + 1e-6
        images[idx] = np.exp(-dist / scale)
    return images


def multiscale_rp_images(X, size):
    Xs = resample_rows(X, size)
    images = np.empty((len(Xs), size, size), dtype=np.float32)
    for idx, row in enumerate(Xs):
        dist = np.abs(row[:, None] - row[None, :])
        scales = [
            np.percentile(dist, 50) + 1e-6,
            np.percentile(dist, 75) + 1e-6,
            np.percentile(dist, 90) + 1e-6,
        ]
        stack = [np.exp(-dist / scale) for scale in scales]
        images[idx] = np.mean(stack, axis=0)
    return images.astype(np.float32)


def spectrogram_images(X, size):
    X = np.asarray(X, dtype=np.float32)
    images = np.empty((len(X), size, size), dtype=np.float32)
    for idx, row in enumerate(X):
        nperseg = max(8, min(len(row), size * 2))
        hop = max(1, nperseg // 2)
        starts = list(range(0, max(1, len(row) - nperseg + 1), hop)) or [0]
        frames = []
        window = np.hanning(nperseg).astype(np.float32)
        for start in starts:
            segment = row[start : start + nperseg]
            if len(segment) < nperseg:
                segment = np.pad(segment, (0, nperseg - len(segment)), mode="edge")
            frames.append(np.log1p(np.abs(np.fft.rfft(segment * window))).astype(np.float32))
        spec = np.vstack(frames).T
        row_axis = np.linspace(0, spec.shape[0] - 1, size)
        col_axis = np.linspace(0, spec.shape[1] - 1, size)
        resized_rows = np.vstack([np.interp(col_axis, np.arange(spec.shape[1]), spec[int(round(r))]) for r in row_axis])
        images[idx] = resized_rows
    images -= images.min(axis=(1, 2), keepdims=True)
    images /= images.max(axis=(1, 2), keepdims=True) + 1e-6
    return images.astype(np.float32)


def scalogram_images(X, size):
    Xs = resample_rows(X, max(size * 2, 96))
    images = np.empty((len(Xs), size, size), dtype=np.float32)
    scales = np.geomspace(2.0, max(4.0, size / 2.0), size).astype(np.float32)
    centered = Xs - Xs.mean(axis=1, keepdims=True)
    col_axis = np.linspace(0, centered.shape[1] - 1, size)
    source_axis = np.arange(centered.shape[1])
    for scale_idx, scale in enumerate(scales):
        radius = int(max(4, min(centered.shape[1] // 2, np.ceil(6 * scale))))
        tau = np.arange(-radius, radius + 1, dtype=np.float32)
        wavelet = np.exp(-(tau**2) / (2 * scale**2)) * np.cos(5.0 * tau / scale)
        wavelet -= wavelet.mean()
        wavelet /= np.sqrt(np.sum(wavelet**2)) + 1e-6
        conv = fftconvolve(centered, wavelet.reshape(1, -1), mode="same", axes=1)
        conv = np.abs(conv).astype(np.float32)
        if conv.shape[1] != centered.shape[1]:
            conv_axis = np.linspace(0, centered.shape[1] - 1, conv.shape[1], dtype=np.float32)
            conv = np.vstack([np.interp(source_axis, conv_axis, row) for row in conv]).astype(np.float32)
        images[:, scale_idx, :] = np.vstack([np.interp(col_axis, source_axis, row) for row in conv])
    images = np.log1p(images)
    images -= images.min(axis=(1, 2), keepdims=True)
    images /= images.max(axis=(1, 2), keepdims=True) + 1e-6
    return images.astype(np.float32)


def image_stack_for_kind(X, image_kind, size):
    if image_kind == "gasf":
        return gasf_images(X, size)
    if image_kind == "gadf":
        return gadf_images(X, size)
    if image_kind == "mtf":
        return mtf_images(X, size)
    if image_kind == "rp":
        return rp_images(X, size)
    if image_kind == "multiscale_rp":
        return multiscale_rp_images(X, size)
    if image_kind == "spectrogram":
        return spectrogram_images(X, size)
    if image_kind == "scalogram":
        return scalogram_images(X, size)
    raise ValueError(f"Unknown image transform: {image_kind}")


def hog_image_features(images, orientations=8, cells=4):
    images = np.asarray(images, dtype=np.float32)
    gx = np.diff(images, axis=2, prepend=images[:, :, :1])
    gy = np.diff(images, axis=1, prepend=images[:, :1, :])
    magnitude = np.sqrt(gx * gx + gy * gy)
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    orientation_bins = np.floor(angle / np.pi * orientations).astype(np.int32)
    orientation_bins = np.clip(orientation_bins, 0, orientations - 1)
    h, w = images.shape[1], images.shape[2]
    y_edges = np.linspace(0, h, cells + 1, dtype=int)
    x_edges = np.linspace(0, w, cells + 1, dtype=int)
    features = np.empty((len(images), cells * cells * orientations), dtype=np.float32)
    col = 0
    for y0, y1 in zip(y_edges[:-1], y_edges[1:]):
        for x0, x1 in zip(x_edges[:-1], x_edges[1:]):
            cell_mag = magnitude[:, y0:y1, x0:x1].reshape(len(images), -1)
            cell_bins = orientation_bins[:, y0:y1, x0:x1].reshape(len(images), -1)
            hist = np.zeros((len(images), orientations), dtype=np.float32)
            for bin_idx in range(orientations):
                hist[:, bin_idx] = np.where(cell_bins == bin_idx, cell_mag, 0.0).sum(axis=1)
            features[:, col : col + orientations] = hist
            col += orientations
    norm = np.linalg.norm(features, axis=1, keepdims=True) + 1e-6
    return (features / norm).astype(np.float32)


def lbp_image_features(images):
    images = np.asarray(images, dtype=np.float32)
    if images.shape[1] < 3 or images.shape[2] < 3:
        return images.reshape(len(images), -1).astype(np.float32)
    center = images[:, 1:-1, 1:-1]
    neighbors = [
        images[:, :-2, :-2],
        images[:, :-2, 1:-1],
        images[:, :-2, 2:],
        images[:, 1:-1, 2:],
        images[:, 2:, 2:],
        images[:, 2:, 1:-1],
        images[:, 2:, :-2],
        images[:, 1:-1, :-2],
    ]
    codes = np.zeros(center.shape, dtype=np.uint8)
    for bit, neighbor in enumerate(neighbors):
        codes |= ((neighbor >= center).astype(np.uint8) << bit)
    flat = codes.reshape(len(images), -1)
    hist = np.zeros((len(images), 256), dtype=np.float32)
    for idx, row in enumerate(flat):
        hist[idx] = np.bincount(row, minlength=256).astype(np.float32)
    hist /= hist.sum(axis=1, keepdims=True) + 1e-6
    return hist


def glcm_image_features(images, levels=8):
    images = np.asarray(images, dtype=np.float32)
    features = []
    offsets = [(0, 1), (1, 0), (1, 1), (1, -1)]
    yy, xx = np.mgrid[0:levels, 0:levels].astype(np.float32)
    for image in images:
        img = image - np.nanmin(image)
        img = img / (np.nanmax(img) + 1e-6)
        quantized = np.clip((img * levels).astype(np.int32), 0, levels - 1)
        row_features = []
        for dy, dx in offsets:
            if dy >= 0:
                a_y = slice(0, quantized.shape[0] - dy)
                b_y = slice(dy, quantized.shape[0])
            else:
                a_y = slice(-dy, quantized.shape[0])
                b_y = slice(0, quantized.shape[0] + dy)
            if dx >= 0:
                a_x = slice(0, quantized.shape[1] - dx)
                b_x = slice(dx, quantized.shape[1])
            else:
                a_x = slice(-dx, quantized.shape[1])
                b_x = slice(0, quantized.shape[1] + dx)
            a = quantized[a_y, a_x].ravel()
            b = quantized[b_y, b_x].ravel()
            matrix = np.zeros((levels, levels), dtype=np.float32)
            np.add.at(matrix, (a, b), 1.0)
            matrix = matrix + matrix.T
            matrix /= matrix.sum() + 1e-6
            contrast = np.sum(((yy - xx) ** 2) * matrix)
            dissimilarity = np.sum(np.abs(yy - xx) * matrix)
            homogeneity = np.sum(matrix / (1.0 + np.abs(yy - xx)))
            asm = np.sum(matrix * matrix)
            energy = np.sqrt(asm)
            entropy = -np.sum(matrix * np.log(matrix + 1e-6))
            mean_i = np.sum(yy * matrix)
            mean_j = np.sum(xx * matrix)
            std_i = np.sqrt(np.sum(((yy - mean_i) ** 2) * matrix)) + 1e-6
            std_j = np.sqrt(np.sum(((xx - mean_j) ** 2) * matrix)) + 1e-6
            correlation = np.sum((yy - mean_i) * (xx - mean_j) * matrix) / (std_i * std_j)
            row_features.extend([contrast, dissimilarity, homogeneity, asm, energy, entropy, correlation])
        features.append(row_features)
    return np.asarray(features, dtype=np.float32)


def image_tensor_for_config(X, config):
    size = int(config.get("size", 32))
    image_kinds = config["image"]
    if isinstance(image_kinds, str):
        image_kinds = [image_kinds]
    channels = [image_stack_for_kind(X, image_kind, size) for image_kind in image_kinds]
    return np.stack(channels, axis=1).astype(np.float32)


def cnn_autoencoder_score_pair(train_images, test_images, config):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        raise RuntimeError("PyTorch is required for imaging_cnn_autoencoder experiments.") from exc

    torch.manual_seed(RNG_SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_images = np.asarray(train_images, dtype=np.float32)
    test_images = np.asarray(test_images, dtype=np.float32)
    max_train = int(config.get("max_train_samples", 512))
    if len(train_images) > max_train:
        rng = np.random.default_rng(RNG_SEED)
        sample_idx = np.sort(rng.choice(len(train_images), size=max_train, replace=False))
        fit_images = train_images[sample_idx]
    else:
        fit_images = train_images

    channels = train_images.shape[1]
    model = nn.Sequential(
        nn.Conv2d(channels, 8, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(8, channels, kernel_size=4, stride=2, padding=1),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("lr", 1e-3)), weight_decay=1e-4)
    batch_size = max(8, int(config.get("batch_size", 128)))
    dataset = TensorDataset(torch.from_numpy(fit_images))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(max(1, int(config.get("epochs", 8)))):
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = torch.mean((recon - batch) ** 2)
            loss.backward()
            optimizer.step()

    def reconstruction_errors(images):
        out = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                batch = torch.from_numpy(images[start : start + batch_size]).to(device)
                recon = model(batch)
                err = torch.mean((recon - batch) ** 2, dim=(1, 2, 3))
                out.append(err.detach().cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    return reconstruction_errors(train_images), reconstruction_errors(test_images)


def pretrained_cnn_features(images, config):
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torchvision.models import ResNet18_Weights, resnet18
    except Exception as exc:
        raise RuntimeError("torchvision is required for imaging_pretrained_cnn_knn experiments.") from exc

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cache_key = ("resnet18", str(device))
    if cache_key not in _PRETRAINED_CNN_CACHE:
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.fc = nn.Identity()
        model.eval().to(device)
        _PRETRAINED_CNN_CACHE[cache_key] = (model, weights.transforms())
    model, _ = _PRETRAINED_CNN_CACHE[cache_key]

    images = np.asarray(images, dtype=np.float32)
    if images.shape[1] == 1:
        images = np.repeat(images, 3, axis=1)
    elif images.shape[1] > 3:
        images = images[:, :3]
    elif images.shape[1] == 2:
        images = np.concatenate([images, images[:, :1]], axis=1)
    batch_size = max(8, int(config.get("batch_size", 96)))
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    features = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = torch.from_numpy(images[start : start + batch_size]).to(device)
            batch = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)
            batch = (batch - mean) / std
            features.append(model(batch).detach().cpu().numpy().astype(np.float32))
    return np.vstack(features).astype(np.float32)


def pretrained_vit_features(images, config):
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torchvision.models import ViT_B_16_Weights, ViT_B_32_Weights, vit_b_16, vit_b_32
    except Exception as exc:
        raise RuntimeError("torchvision is required for imaging_pretrained_vit_knn experiments.") from exc

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    arch = str(config.get("vit_arch", "vit_b_32"))
    cache_key = (arch, str(device))
    if cache_key not in _PRETRAINED_VIT_CACHE:
        if arch == "vit_b_16":
            weights = ViT_B_16_Weights.DEFAULT
            model = vit_b_16(weights=weights)
        elif arch == "vit_b_32":
            weights = ViT_B_32_Weights.DEFAULT
            model = vit_b_32(weights=weights)
        else:
            raise ValueError(f"Unsupported ViT architecture: {arch}")
        model.heads = nn.Identity()
        model.eval().to(device)
        _PRETRAINED_VIT_CACHE[cache_key] = model
    model = _PRETRAINED_VIT_CACHE[cache_key]

    images = np.asarray(images, dtype=np.float32)
    if images.shape[1] == 1:
        images = np.repeat(images, 3, axis=1)
    elif images.shape[1] > 3:
        images = images[:, :3]
    elif images.shape[1] == 2:
        images = np.concatenate([images, images[:, :1]], axis=1)
    batch_size = max(4, int(config.get("batch_size", 32)))
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    features = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = torch.from_numpy(images[start : start + batch_size]).to(device)
            batch = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)
            batch = (batch - mean) / std
            features.append(model(batch).detach().cpu().numpy().astype(np.float32))
    return np.vstack(features).astype(np.float32)


def imaging_features(X, config):
    size = int(config.get("size", 32))
    image_kinds = config["image"]
    if isinstance(image_kinds, str):
        image_kinds = [image_kinds]
    images = np.concatenate([image_stack_for_kind(X, image_kind, size) for image_kind in image_kinds], axis=1)
    extractor = config.get("feature_extractor")
    if extractor == "hog":
        return hog_image_features(images)
    if extractor == "lbp":
        return lbp_image_features(images)
    if extractor == "glcm":
        return glcm_image_features(images)
    return images.reshape(len(images), -1).astype(np.float32)


def score_pair_for_config(X_train, X_test, target_len, config, record, feature_cache=None):
    kind = config["kind"]
    if kind == "rocket_knn":
        train_features, test_features = rocket_feature_pair(
            X_train,
            X_test,
            target_len,
            config.get("num_kernels", 256),
            config.get("seed_offset", 0),
        )
        train_features, test_features = scale_feature_pair(train_features, test_features)
        return knn_score_pair(train_features, test_features, config.get("neighbors", 3))
    if kind == "rocket_density_corrected":
        train_features, test_features = rocket_feature_pair(
            X_train,
            X_test,
            target_len,
            config.get("num_kernels", 256),
            config.get("seed_offset", 501),
        )
        train_features, test_features = scale_feature_pair(train_features, test_features)
        local_train, local_test = density_knn_score_pair(
            train_features,
            test_features,
            config.get("neighbors", 3),
            "local_gap",
        )
        ratio_train, ratio_test = density_knn_score_pair(
            train_features,
            test_features,
            config.get("neighbors", 3),
            "ratio",
        )
        local_train_rank, local_test_rank = rank_normalize_against_train(local_train, local_test)
        ratio_train_rank, ratio_test_rank = rank_normalize_against_train(ratio_train, ratio_test)
        return (local_train_rank + ratio_train_rank) / 2.0, (local_test_rank + ratio_test_rank) / 2.0
    if kind in {"aeon_multirocket", "aeon_hydra", "aeon_multirocket_hydra"}:
        cache_key = (
            kind,
            config.get("num_kernels"),
            config.get("hydra_kernels"),
            config.get("hydra_groups"),
            config.get("feature_prune"),
            config.get("feature_keep"),
            config.get("random_state"),
            config.get("series_scale", "per_series_z"),
        )
        cached = feature_cache.get(cache_key) if feature_cache is not None else None
        if cached is None:
            train_features, test_features = aeon_feature_pair(X_train, X_test, config, record)
            train_features, test_features = maybe_prune_features(train_features, test_features, config)
            train_features, test_features = scale_feature_pair(train_features, test_features)
            if feature_cache is not None:
                feature_cache[cache_key] = (train_features, test_features)
        else:
            train_features, test_features = cached
        score_mode = config.get("score_mode", "local_gap")
        if score_mode == "local_gap":
            return density_knn_score_pair(
                train_features,
                test_features,
                config.get("neighbors", 3),
                "local_gap",
            )
        if score_mode == "ratio":
            return density_knn_score_pair(
                train_features,
                test_features,
                config.get("neighbors", 3),
                "ratio",
            )
        if score_mode == "knn":
            return knn_score_pair(train_features, test_features, config.get("neighbors", 3))
        if score_mode == "kmeans_crossfit":
            return kmeans_crossfit_score_pair(train_features, test_features, config)
        if score_mode == "gmm_diag_crossfit":
            return gmm_crossfit_score_pair(train_features, test_features, config)
        if score_mode == "lof_novelty_crossfit":
            return lof_crossfit_score_pair(train_features, test_features, config)
        if score_mode == "ocsvm_crossfit":
            return ocsvm_crossfit_score_pair(train_features, test_features, config)
        raise ValueError(f"Unknown aeon score_mode: {score_mode}")
    if kind == "interval_quantile_knn":
        train_features = interval_quantile_features(X_train, config)
        test_features = interval_quantile_features(X_test, config)
        train_features, test_features = scale_feature_pair(train_features, test_features)
        return knn_score_pair(train_features, test_features, config.get("neighbors", 3))
    if kind == "frequency_knn":
        train_features = fft_band_features(X_train, config.get("n_bands", 24))
        test_features = fft_band_features(X_test, config.get("n_bands", 24))
        train_features, test_features = scale_feature_pair(train_features, test_features)
        return knn_score_pair(train_features, test_features, config.get("neighbors", 3))
    if kind == "frequency_rocket_rank_ensemble":
        X_diff_train = np.diff(X_train, axis=1, prepend=X_train[:, :1]).astype(np.float32)
        X_diff_test = np.diff(X_test, axis=1, prepend=X_test[:, :1]).astype(np.float32)
        raw_train, raw_test = rocket_feature_pair(X_train, X_test, target_len, config["num_kernels"], 101)
        diff_train, diff_test = rocket_feature_pair(X_diff_train, X_diff_test, target_len, config["num_kernels"], 151)
        fft_train = fft_band_features(X_train, config.get("n_bands", 24))
        fft_test = fft_band_features(X_test, config.get("n_bands", 24))
        components = []
        for tr, te in [(raw_train, raw_test), (diff_train, diff_test), (fft_train, fft_test)]:
            tr, te = scale_feature_pair(tr, te)
            components.append(knn_score_pair(tr, te, config.get("neighbors", 3)))
        return combine_rank_scores(components)
    if kind == "bandpass_rocket_rank_ensemble":
        components = []
        for idx, (lo, hi) in enumerate(config["bands"]):
            bp_train = z_normalize(bandpass_filter_rows(X_train, lo, hi)).astype(np.float32)
            bp_test = z_normalize(bandpass_filter_rows(X_test, lo, hi)).astype(np.float32)
            tr, te = rocket_feature_pair(bp_train, bp_test, target_len, config["num_kernels"], 200 + idx)
            tr, te = scale_feature_pair(tr, te)
            components.append(knn_score_pair(tr, te, config.get("neighbors", 3)))
        return combine_rank_scores(components)
    if kind == "shapelet_prototype_knn":
        max_len = min(int(config.get("max_len", 512)), target_len)
        X_train_s = z_normalize(resample_rows(X_train, max_len)).astype(np.float32)
        X_test_s = z_normalize(resample_rows(X_test, max_len)).astype(np.float32)
        shapelets = choose_shapelets(
            X_train_s,
            config.get("num_shapelets", 24),
            config.get("shapelet_lengths", [16, 32]),
            RNG_SEED + len(record["dataset_name"]),
        )
        train_features = shapelet_distance_features(X_train_s, shapelets)
        test_features = shapelet_distance_features(X_test_s, shapelets)
        train_features, test_features = scale_feature_pair(train_features, test_features)
        return knn_score_pair(train_features, test_features, config.get("neighbors", 3))
    if kind in {"injection_centroid_score", "injection_knn_score"}:
        train_features = interval_frequency_embedding(X_train)
        test_features = interval_frequency_embedding(X_test)
        synthetic = inject_anomalies(
            X_train,
            synthetic_per_train=config.get("synthetic_per_train", 1),
            seed=RNG_SEED + len(record["dataset_name"]),
        )
        synthetic_features = interval_frequency_embedding(synthetic)
        train_features, test_features, synthetic_features = scale_feature_triplet(
            train_features,
            test_features,
            synthetic_features,
        )
        if kind == "injection_knn_score":
            train_scores, test_scores = knn_score_pair(train_features, test_features, config.get("neighbors", 3))
            syn_centroid = synthetic_features.mean(axis=0)
            train_syn = np.linalg.norm(train_features - syn_centroid, axis=1)
            test_syn = np.linalg.norm(test_features - syn_centroid, axis=1)
            train_rank, test_rank = rank_normalize_scores(train_scores, test_scores)
            syn_train_rank, syn_test_rank = rank_normalize_against_train(train_syn, test_syn)
            return train_rank, (test_rank + (1.0 - syn_test_rank)) / 2.0
        normal_centroid = train_features.mean(axis=0)
        synthetic_centroid = synthetic_features.mean(axis=0)
        train_normal = np.linalg.norm(train_features - normal_centroid, axis=1)
        test_normal = np.linalg.norm(test_features - normal_centroid, axis=1)
        train_synthetic = np.linalg.norm(train_features - synthetic_centroid, axis=1)
        test_synthetic = np.linalg.norm(test_features - synthetic_centroid, axis=1)
        train_scores = train_normal - 0.25 * train_synthetic
        test_scores = test_normal - 0.25 * test_synthetic
        return train_scores, test_scores
    if kind == "imaging_knn":
        train_features = imaging_features(X_train, config)
        test_features = imaging_features(X_test, config)
        train_features, test_features = pca_pair(train_features, test_features, config.get("pca", 32))
        train_features, test_features = scale_feature_pair(train_features, test_features)
        return knn_score_pair(train_features, test_features, config.get("neighbors", 3))
    if kind == "imaging_cnn_autoencoder":
        train_images = image_tensor_for_config(X_train, config)
        test_images = image_tensor_for_config(X_test, config)
        return cnn_autoencoder_score_pair(train_images, test_images, config)
    if kind == "imaging_pretrained_cnn_knn":
        train_images = image_tensor_for_config(X_train, config)
        test_images = image_tensor_for_config(X_test, config)
        train_features = pretrained_cnn_features(train_images, config)
        test_features = pretrained_cnn_features(test_images, config)
        train_features, test_features = pca_pair(train_features, test_features, config.get("pca", 64))
        train_features, test_features = scale_feature_pair(train_features, test_features)
        return knn_score_pair(train_features, test_features, config.get("neighbors", 3))
    if kind == "imaging_pretrained_vit_knn":
        image_kinds = config.get("image", "")
        if isinstance(image_kinds, list):
            image_kinds = tuple(image_kinds)
        cache_key = (
            "pretrained_vit",
            config.get("vit_arch", "vit_b_32"),
            image_kinds,
            config.get("size", 32),
            config.get("batch_size", 32),
            config.get("series_scale", "per_series_z"),
        )
        cached = feature_cache.get(cache_key) if feature_cache is not None else None
        if cached is None:
            train_images = image_tensor_for_config(X_train, config)
            test_images = image_tensor_for_config(X_test, config)
            all_images = np.concatenate([train_images, test_images], axis=0)
            all_features = pretrained_vit_features(all_images, config)
            train_features = all_features[: len(train_images)]
            test_features = all_features[len(train_images) :]
            if feature_cache is not None:
                feature_cache[cache_key] = (train_features, test_features)
        else:
            train_features, test_features = cached
        train_features, test_features = compress_feature_pair(train_features, test_features, config)
        train_features, test_features = scale_feature_pair(train_features, test_features)
        return knn_score_pair(train_features, test_features, config.get("neighbors", 3))
    if kind == "train_stability_rank_ensemble":
        components = []
        raw_train, raw_test = rocket_feature_pair(
            X_train,
            X_test,
            target_len,
            config.get("num_kernels", 192),
            701,
        )
        raw_train, raw_test = scale_feature_pair(raw_train, raw_test)
        components.append(knn_score_pair(raw_train, raw_test, config.get("neighbors", 3)))

        fft_train = fft_band_features(X_train, config.get("n_bands", 32))
        fft_test = fft_band_features(X_test, config.get("n_bands", 32))
        fft_train, fft_test = scale_feature_pair(fft_train, fft_test)
        components.append(knn_score_pair(fft_train, fft_test, config.get("neighbors", 3)))

        max_len = min(int(config.get("max_len", 512)), target_len)
        X_train_s = z_normalize(resample_rows(X_train, max_len)).astype(np.float32)
        X_test_s = z_normalize(resample_rows(X_test, max_len)).astype(np.float32)
        shapelets = choose_shapelets(
            X_train_s,
            config.get("num_shapelets", 24),
            config.get("shapelet_lengths", [16, 32]),
            RNG_SEED + 900 + len(record["dataset_name"]),
        )
        shp_train = shapelet_distance_features(X_train_s, shapelets)
        shp_test = shapelet_distance_features(X_test_s, shapelets)
        shp_train, shp_test = scale_feature_pair(shp_train, shp_test)
        components.append(knn_score_pair(shp_train, shp_test, config.get("neighbors", 3)))

        weighted_train = []
        weighted_test = []
        weights = []
        for train_scores, test_scores in components:
            train_rank, test_rank = rank_normalize_against_train(train_scores, test_scores)
            tail = np.percentile(train_rank, 95) - np.percentile(train_rank, 50)
            spread = np.percentile(train_rank, 75) - np.percentile(train_rank, 25)
            stability = 1.0 / (1e-6 + max(float(tail), 0.05) + 0.5 * max(float(spread), 0.05))
            weights.append(stability)
            weighted_train.append(train_rank)
            weighted_test.append(test_rank)
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / max(float(weights.sum()), 1e-6)
        train_scores = np.sum([w * s for w, s in zip(weights, weighted_train)], axis=0)
        test_scores = np.sum([w * s for w, s in zip(weights, weighted_test)], axis=0)
        return train_scores, test_scores
    raise ValueError(f"Unknown config kind: {kind}")


def rank_normalize_against_train(train_scores, test_scores):
    return rank_normalize_scores(train_scores, test_scores)


def append_rows(path, rows, fieldnames):
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_existing_detail_rows(path):
    if not path.exists() or path.stat().st_size == 0:
        return [], None
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def completed_dataset_names(rows, exp_id):
    return {row["dataset_name"] for row in rows if row.get("experiment_id") == exp_id}


def format_indices(indices, sort=True):
    values = sorted(indices) if sort else indices
    return " ".join(str(int(idx)) for idx in values)


def format_float_list(values, ndigits=6):
    return " ".join(f"{float(value):.{ndigits}g}" for value in values)


def threshold_selection_diagnostics(test_scores, threshold, top_n=10):
    scores = np.asarray(test_scores, dtype=np.float64)
    if len(scores) == 0:
        return {
            "selected_indices": "",
            "top_score_indices": "",
            "top_score_values": "",
            "selected_score_max": "",
            "selected_score_min": "",
            "top1_score": "",
            "top2_score": "",
            "top1_top2_margin": "",
            "top1_threshold_margin": "",
        }
    selected = np.flatnonzero(scores > threshold)
    order = np.argsort(scores)[::-1][: min(int(top_n), len(scores))]
    top_values = scores[order]
    top1 = float(top_values[0]) if len(top_values) else float("nan")
    top2 = float(top_values[1]) if len(top_values) > 1 else float("nan")
    selected_scores = scores[selected] if len(selected) else np.asarray([], dtype=np.float64)
    return {
        "selected_indices": format_indices(selected),
        "top_score_indices": format_indices(order, sort=False),
        "top_score_values": format_float_list(top_values),
        "selected_score_max": float(np.max(selected_scores)) if len(selected_scores) else "",
        "selected_score_min": float(np.min(selected_scores)) if len(selected_scores) else "",
        "top1_score": top1 if not np.isnan(top1) else "",
        "top2_score": top2 if not np.isnan(top2) else "",
        "top1_top2_margin": (top1 - top2) if not np.isnan(top1) and not np.isnan(top2) else "",
        "top1_threshold_margin": (top1 - float(threshold)) if not np.isnan(top1) else "",
    }


def assert_detail_dataset_coverage(path, expected_dataset_names, exp_id, logger):
    disk_rows, _ = read_existing_detail_rows(path)
    disk_dataset_names = completed_dataset_names(disk_rows, exp_id)
    missing = [name for name in expected_dataset_names if name not in disk_dataset_names]
    missing_path = path.with_name(f"{path.stem}_missing_datasets.csv")
    if missing:
        with missing_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["experiment_id", "dataset_name"])
            writer.writeheader()
            writer.writerows({"experiment_id": exp_id, "dataset_name": name} for name in missing)
        logger.warning(
            "%s detail CSV is missing %d/%d target datasets after run; wrote %s. Queue will continue.",
            exp_id,
            len(missing),
            len(expected_dataset_names),
            missing_path,
        )
    elif missing_path.exists():
        missing_path.unlink()
    return disk_rows


def repair_missing_datasets(exp_id, db_path, detail_path, target_rows, fieldnames, logger, max_attempts=1):
    detail_rows, existing_fieldnames = read_existing_detail_rows(detail_path)
    fieldnames = fieldnames or existing_fieldnames
    expected_dataset_names = [row["dataset_name"] for row in target_rows]
    by_name = {row["dataset_name"]: row for row in target_rows}
    for attempt in range(1, int(max_attempts) + 1):
        completed = completed_dataset_names(detail_rows, exp_id)
        missing = [name for name in expected_dataset_names if name not in completed]
        if not missing:
            break
        logger.warning(
            "%s repairing %d missing target datasets before queue continues. attempt=%d",
            exp_id,
            len(missing),
            attempt,
        )
        for dataset_name in missing:
            difficulty_row = by_name[dataset_name]
            try:
                rows = run_dataset_task((exp_id, str(db_path), difficulty_row))
            except Exception as exc:
                logger.error("Repair failed for %s: %s", dataset_name, exc, exc_info=True)
                rows = []
            if not rows:
                logger.warning("Repair produced no rows for %s", dataset_name)
                continue
            detail_rows.extend(rows)
            fieldnames = fieldnames or list(rows[0].keys())
            append_rows(detail_path, rows, fieldnames)
    return read_existing_detail_rows(detail_path)[0]


def run_dataset_task(task):
    exp_id, db_path, difficulty_row = task
    spec = get_spec(exp_id)
    record = load_original_record(difficulty_row["dataset_name"], db_path)
    y_test = record["y_test"]
    if len(record["train_series"]) == 0 or len(record["test_series"]) == 0 or len(np.unique(y_test)) < 2:
        return []

    target_len = target_len_for_record(record, "actual_median")
    target_len = min(max(8, target_len), 2048)
    X_train_raw = align_series_lengths(record["train_series"], target_len)
    X_test_raw = align_series_lengths(record["test_series"], target_len)
    X_train_z = z_normalize(X_train_raw).astype(np.float32)
    X_test_z = z_normalize(X_test_raw).astype(np.float32)

    rows = []
    feature_cache = {}
    for config in spec["configs"]:
        X_train, X_test = prepare_series_pair_for_scale(
            config.get("series_scale", "per_series_z"),
            X_train_raw,
            X_test_raw,
            X_train_z,
            X_test_z,
        )
        train_scores, test_scores = score_pair_for_config(
            X_train,
            X_test,
            target_len,
            config,
            record,
            feature_cache=feature_cache,
        )
        metrics = score_metrics(y_test, test_scores)
        for method in spec["thresholds"]:
            rate, threshold_family = rate_for_threshold(method, record)
            threshold, q_effective, cap_target = count_cap_threshold(train_scores, rate)
            train_exceed_count, train_exceed_rate = train_false_positive_stats(train_scores, threshold)
            selection_diagnostics = threshold_selection_diagnostics(test_scores, threshold)
            rows.append(
                {
                    "experiment_id": exp_id,
                    "source_experiment_id": spec["source_experiment_id"],
                    "dataset_name": record["dataset_name"],
                    "family": record["family"],
                    "difficulty_score": difficulty_row["difficulty_score"],
                    "difficulty_reasons": difficulty_row["reasons"],
                    "data_variant": "original_repeated_normal",
                    "target_scope": spec["target"]["scope"],
                    "config_name": config["name"],
                    "score_family": config["kind"],
                    "image_transform": "|".join(config.get("image", []))
                    if isinstance(config.get("image", ""), list)
                    else config.get("image", ""),
                    "feature_extractor": config.get("feature_extractor", ""),
                    "series_scale": config.get("series_scale", "per_series_z"),
                    "compression_method": config.get("compression", ""),
                    "compression_effective_method": config.get(
                        "_compression_effective_method", config.get("compression", "")
                    ),
                    "compression_components_requested": config.get("components", ""),
                    "compression_components_effective": (
                        max(1, min(int(config.get("components", 64)), len(record["train_series"]) - 2))
                        if config.get("compression") == "umap" and len(record["train_series"]) >= 3
                        else config.get("components", "")
                    ),
                    "manifold_neighbors": config.get(
                        "umap_neighbors", config.get("manifold_neighbors", "")
                    ),
                    "manifold_neighbors_effective": config.get(
                        "_manifold_neighbors_effective", ""
                    ),
                    "compression_random_state": config.get("random_state", ""),
                    "lle_eigen_solver_effective": config.get("_lle_eigen_solver_effective", ""),
                    "lle_reg_effective": config.get("_lle_reg_effective", ""),
                    "feature_count": config.get(
                        "feature_keep",
                        config.get(
                            "n_bands",
                            config.get(
                                "num_shapelets",
                                config.get(
                                    "pca",
                                    config.get("components", config.get("hydra_groups", "")),
                                ),
                            ),
                        ),
                    ),
                    "num_kernels": config.get("num_kernels", ""),
                    "knn_neighbors": config.get("neighbors", ""),
                    "score_backend": config.get("score_mode", ""),
                    "score_backend_effective": (
                        "centroid_fallback"
                        if config.get("score_mode")
                        in {"gmm_diag_crossfit", "lof_novelty_crossfit", "ocsvm_crossfit"}
                        and len(record["train_series"]) <= 2
                        else config.get("score_mode", "")
                    ),
                    "score_clusters": config.get("score_clusters", ""),
                    "score_crossfit_folds": (
                        min(5, len(record["train_series"]))
                        if str(config.get("score_mode", "")).endswith("crossfit")
                        else ""
                    ),
                    "threshold_method": method,
                    "threshold_family": threshold_family,
                    "sequence_length": target_len,
                    "metadata_len": record["metadata_len"],
                    "actual_len_median": record["actual_len_median"],
                    "actual_len_max": record["actual_len_max"],
                    "test_actual_len_median": record["test_actual_len_median"],
                    "train_score_count": len(train_scores),
                    "train_count": len(record["train_series"]),
                    "test_size": len(y_test),
                    "anomaly_count": int(np.sum(y_test)),
                    "q_effective": q_effective,
                    "cap_target": cap_target,
                    "threshold": threshold,
                    "train_exceed_count": train_exceed_count,
                    "train_exceed_rate": train_exceed_rate,
                    **selection_diagnostics,
                    **evaluate_threshold(y_test, test_scores, threshold, metrics),
                }
            )
    return rows


def summarize(rows):
    summary = []
    keys = sorted({(row["experiment_id"], row["config_name"], row["threshold_method"]) for row in rows})
    for exp_id, config_name, method in keys:
        subset = [
            row
            for row in rows
            if row["experiment_id"] == exp_id
            and row["config_name"] == config_name
            and row["threshold_method"] == method
        ]
        if not subset:
            continue
        f1s = [float(row["f1"]) for row in subset]
        by_family = defaultdict(list)
        for row in subset:
            by_family[row["family"]].append(float(row["f1"]))
        family_means = [float(np.mean(values)) for values in by_family.values()]
        summary.append(
            {
                "experiment_id": exp_id,
                "data_variant": subset[0]["data_variant"],
                "target_scope": subset[0]["target_scope"],
                "config_name": config_name,
                "score_family": subset[0]["score_family"],
                "threshold_method": method,
                "num_datasets": len(subset),
                "num_families": len(by_family),
                "mean_auc_roc": float(np.mean([float(row["auc_roc"]) for row in subset])),
                "mean_auc_pr": float(np.mean([float(row["auc_pr"]) for row in subset])),
                "mean_f1": float(np.mean(f1s)),
                "median_f1": float(np.median(f1s)),
                "p25_f1": float(np.percentile(f1s, 25)),
                "zero_f1_count": sum(1 for value in f1s if value == 0),
                "ge_0_5_count": sum(1 for value in f1s if value >= 0.5),
                "family_macro_f1": float(np.mean(family_means)) if family_means else 0.0,
                "family_p25_f1": float(np.percentile(family_means, 25)) if family_means else 0.0,
                "mean_predicted_count": float(np.mean([int(row["predicted_count"]) for row in subset])),
                "mean_anomaly_count": float(np.mean([int(row["anomaly_count"]) for row in subset])),
                "mean_tp": float(np.mean([int(row["tp"]) for row in subset])),
                "mean_fp": float(np.mean([int(row["fp"]) for row in subset])),
                "mean_fn": float(np.mean([int(row["fn"]) for row in subset])),
                "mean_train_exceed_rate": float(np.mean([float(row["train_exceed_rate"]) for row in subset])),
                "mean_oracle_f1": float(np.mean([float(row["oracle_f1"]) for row in subset])),
            }
        )
    return sorted(summary, key=lambda row: (row["mean_f1"], row["mean_auc_pr"]), reverse=True)


def make_logger(exp_id, log_path):
    logger = logging.getLogger(exp_id)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(logging.FileHandler(log_path))
    logger.addHandler(logging.StreamHandler())
    for handler in logger.handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    return logger


def run_experiment(exp_id, args):
    spec = get_spec(exp_id)
    paths = spec["paths"]
    logger = make_logger(exp_id, paths["log"])
    if not args.keep_existing:
        for path in [paths["detail"], paths["summary"]]:
            if path.exists():
                path.unlink()

    detail_rows = []
    fieldnames = None
    if args.keep_existing:
        detail_rows, fieldnames = read_existing_detail_rows(paths["detail"])

    targets = target_datasets(spec, args.dataset_limit)
    expected_dataset_names = [row["dataset_name"] for row in targets]
    if args.keep_existing and detail_rows:
        completed = completed_dataset_names(detail_rows, exp_id)
        targets = [row for row in targets if row["dataset_name"] not in completed]
    tasks = [(exp_id, str(args.db_path), row) for row in targets]
    logger.info(
        "Starting %s on %d %s datasets with %d workers. existing_rows=%d",
        exp_id,
        len(tasks),
        spec["target"]["scope"],
        args.workers,
        len(detail_rows),
    )
    if not tasks and detail_rows:
        final_rows = assert_detail_dataset_coverage(paths["detail"], expected_dataset_names, exp_id, logger)
        write_csv(paths["summary"], summarize(final_rows))
        logger.info("%s resume found no remaining datasets.", exp_id)
        return

    completed_count = 0

    def consume_result(difficulty_row, rows):
        nonlocal completed_count, detail_rows, fieldnames
        completed_count += 1
        if not rows:
            logger.warning("No rows produced for %s", difficulty_row["dataset_name"])
        if rows:
            detail_rows.extend(rows)
            fieldnames = fieldnames or list(rows[0].keys())
            append_rows(paths["detail"], rows, fieldnames)
        if completed_count % 5 == 0 or completed_count == len(tasks):
            summary_rows = summarize(detail_rows) if detail_rows else []
            write_csv(paths["summary"], summary_rows)
            best = summary_rows[0] if summary_rows else None
            if best:
                logger.info(
                    "Progress: [%3d/%3d] rows=%d | best=%s/%s meanF1=%.4f medianF1=%.4f aucPR=%.4f zero=%d fp=%.2f",
                    completed_count,
                    len(tasks),
                    len(detail_rows),
                    best["config_name"],
                    best["threshold_method"],
                    best["mean_f1"],
                    best["median_f1"],
                    best["mean_auc_pr"],
                    best["zero_f1_count"],
                    best["mean_fp"],
                )

    if args.workers <= 1:
        for task in tasks:
            _, _, difficulty_row = task
            try:
                rows = run_dataset_task(task)
            except Exception as exc:
                logger.error("Error evaluating %s: %s", difficulty_row["dataset_name"], exc, exc_info=True)
                rows = []
            consume_result(difficulty_row, rows)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_dataset_task, task): task for task in tasks}
            for future in as_completed(futures):
                _, _, difficulty_row = futures[future]
                try:
                    rows = future.result()
                except Exception as exc:
                    logger.error("Error evaluating %s: %s", difficulty_row["dataset_name"], exc, exc_info=True)
                    rows = []
                consume_result(difficulty_row, rows)
    if detail_rows:
        write_csv(paths["detail"], detail_rows)
    final_rows = repair_missing_datasets(exp_id, args.db_path, paths["detail"], target_datasets(spec, args.dataset_limit), fieldnames, logger)
    final_rows = assert_detail_dataset_coverage(paths["detail"], expected_dataset_names, exp_id, logger)
    write_csv(paths["summary"], summarize(final_rows))
    logger.info("%s finished.", exp_id)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run model-hard response research experiments.")
    parser.add_argument("experiment_id", nargs="?", choices=available_experiment_ids())
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--dataset-limit", type=int, default=None)
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args(argv)


def main_for_experiment(exp_id, argv=None):
    run_experiment(exp_id, parse_args(argv))


def main(argv=None):
    args = parse_args(argv)
    if not args.experiment_id:
        raise SystemExit("experiment_id is required")
    run_experiment(args.experiment_id, args)


if __name__ == "__main__":
    main()
