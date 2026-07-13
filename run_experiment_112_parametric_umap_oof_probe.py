import argparse
import csv
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from umap import ParametricUMAP

import run_model_hard_research_experiments as research
from run_balanced_improvement_experiment import count_cap_threshold, evaluate_threshold, score_metrics
from run_experiment_29_train_normal_threshold_calibration import knn_score_pair, train_false_positive_stats
from run_original_improvement_experiment import rate_for_threshold, target_len_for_record
from run_rank_ensemble_calibration import align_series_lengths, z_normalize


EXPERIMENT_ID = "experiment_112_parametric_umap_oof_probe"
DATA_DIR = Path("/Users/minho/Documents/Dataset")
DETAIL_PATH = DATA_DIR / f"{EXPERIMENT_ID}_results.csv"
SUMMARY_PATH = DATA_DIR / f"{EXPERIMENT_ID}_summary.csv"
RNG_SEED = 20260720
THRESHOLDS = ["count_cap_2pct", "count_cap_3pct", "family_guard_v1"]
CONFIGS = [
    {"name": "vit_spectrogram_gaussian_rp64_knn3", "kind": "gaussian_rp", "components": 64},
    {"name": "vit_spectrogram_pca64_knn3", "kind": "pca", "components": 64},
    {"name": "parametric_umap8_direct_knn3", "kind": "pumap_direct", "components": 8},
    {"name": "parametric_umap8_oof3_knn3", "kind": "pumap_oof", "components": 8},
]
TARGET_SPEC = {
    "target": {
        "scope": "full_original",
        "families": research.HARD_SCORE_FAMILIES,
        "train_count_strata": [
            {"min": 200, "max": 1000, "count": 8},
            {"min": 1001, "max": None, "count": 7},
        ],
    }
}


def configure_tensorflow():
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def target_rows(limit=None):
    rows = research.target_datasets(TARGET_SPEC)
    return rows[:limit] if limit is not None else rows


def parametric_embed_many(X_fit, queries, components, seed):
    X_fit = np.asarray(X_fit, dtype=np.float32)
    queries = [np.asarray(query, dtype=np.float32) for query in queries]
    if len(X_fit) < 20:
        train_embedding = None
        query_embeddings = []
        for query in queries:
            fit_embedding, query_embedding = research.pca_pair(X_fit, query, components)
            train_embedding = fit_embedding
            query_embeddings.append(query_embedding)
        return train_embedding, query_embeddings, "pca_fallback_small_train"

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(int(seed))
    model = ParametricUMAP(
        n_components=int(components),
        n_neighbors=min(10, len(X_fit) - 1),
        n_epochs=10,
        batch_size=min(256, max(32, len(X_fit))),
        verbose=False,
        random_state=int(seed),
        keras_fit_kwargs={"verbose": 0},
    )
    # UMAP 0.5.12 trains Keras for loss_report_frequency * n_training_epochs.
    model.n_training_epochs = 1
    model.loss_report_frequency = 1
    model.fit(X_fit)
    fit_embedding = np.asarray(model.transform(X_fit), dtype=np.float32)
    query_embeddings = [np.asarray(model.transform(query), dtype=np.float32) for query in queries]
    tf.keras.backend.clear_session()
    return fit_embedding, query_embeddings, "parametric_umap"


def parametric_embed_pair(X_fit, X_query, components, seed):
    train_embedding, query_embeddings, status = parametric_embed_many(X_fit, [X_query], components, seed)
    return (train_embedding, query_embeddings[0]), status


def query_knn_scores(X_reference, X_query, neighbors=3):
    scaler = research.RobustScaler(quantile_range=(10, 90))
    reference = scaler.fit_transform(X_reference)
    query = scaler.transform(X_query)
    count = max(1, min(int(neighbors), len(reference)))
    model = NearestNeighbors(n_neighbors=count, metric="euclidean").fit(reference)
    distances, _ = model.kneighbors(query)
    return distances.mean(axis=1)


def parametric_oof_score_pair(X_train, X_test, components, seed, folds=3):
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    splitter = KFold(n_splits=min(int(folds), len(X_train)), shuffle=True, random_state=int(seed))
    train_scores = np.zeros(len(X_train), dtype=np.float64)
    test_scores = []
    statuses = []
    for fold_index, (fit_idx, heldout_idx) in enumerate(splitter.split(X_train)):
        fit_embedding, embeddings, status = parametric_embed_many(
            X_train[fit_idx], [X_train[heldout_idx], X_test], components, int(seed) + fold_index
        )
        heldout_embedding, test_embedding = embeddings
        train_scores[heldout_idx] = query_knn_scores(fit_embedding, heldout_embedding)
        test_scores.append(query_knn_scores(fit_embedding, test_embedding))
        statuses.append(status)
    status = "parametric_umap" if set(statuses) == {"parametric_umap"} else "pca_fallback_small_train"
    return train_scores, np.median(np.vstack(test_scores), axis=0), status


def vit_features(record, target_len):
    X_train = z_normalize(align_series_lengths(record["train_series"], target_len)).astype(np.float32)
    X_test = z_normalize(align_series_lengths(record["test_series"], target_len)).astype(np.float32)
    vit_config = {"vit_arch": "vit_b_32", "image": "spectrogram", "size": 32, "batch_size": 32}
    train_images = research.image_tensor_for_config(X_train, vit_config)
    test_images = research.image_tensor_for_config(X_test, vit_config)
    features = research.pretrained_vit_features(np.concatenate([train_images, test_images]), vit_config)
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    return features[: len(train_images)], features[len(train_images) :]


def score_config(train_features, test_features, config):
    if config["kind"] == "gaussian_rp":
        train, test = research.random_projection_pair(train_features, test_features, config["components"], sparse=False)
        train, test = research.scale_feature_pair(train, test)
        scores = knn_score_pair(train, test, 3)
        return *scores, "gaussian_rp"
    if config["kind"] == "pca":
        train, test = research.pca_pair(train_features, test_features, config["components"])
        train, test = research.scale_feature_pair(train, test)
        scores = knn_score_pair(train, test, 3)
        return *scores, "pca"
    if config["kind"] == "pumap_direct":
        (train, test), status = parametric_embed_pair(train_features, test_features, config["components"], RNG_SEED)
        train, test = research.scale_feature_pair(train, test)
        scores = knn_score_pair(train, test, 3)
        return *scores, status
    if config["kind"] == "pumap_oof":
        train_scores, test_scores, status = parametric_oof_score_pair(
            train_features, test_features, config["components"], RNG_SEED, folds=3
        )
        return train_scores, test_scores, status
    raise ValueError(f"Unknown config kind: {config['kind']}")


def selected_indices(scores, threshold):
    return " ".join(str(int(value)) for value in np.flatnonzero(np.asarray(scores) > threshold))


def evaluate_record(difficulty_row):
    record = research.load_original_record(difficulty_row["dataset_name"], str(research.DB_PATH))
    y_test = record["y_test"]
    target_len = min(max(8, target_len_for_record(record, "actual_median")), 2048)
    train_features, test_features = vit_features(record, target_len)
    rows = []
    for config in CONFIGS:
        train_scores, test_scores, effective_method = score_config(train_features, test_features, config)
        metrics = score_metrics(y_test, test_scores)
        for method in THRESHOLDS:
            rate, threshold_family = rate_for_threshold(method, record)
            threshold, q_effective, cap_target = count_cap_threshold(train_scores, rate)
            train_exceed_count, train_exceed_rate = train_false_positive_stats(train_scores, threshold)
            evaluation = evaluate_threshold(y_test, test_scores, threshold, metrics)
            rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "source_experiment_id": "parametric_umap_oof_probe",
                    "dataset_name": record["dataset_name"],
                    "family": record["family"],
                    "difficulty_score": difficulty_row["difficulty_score"],
                    "data_variant": "original_repeated_normal",
                    "target_scope": "large_normal_train_only",
                    "config_name": config["name"],
                    "compression_method": config["kind"],
                    "compression_effective_method": effective_method,
                    "embedding_components": config["components"],
                    "score_calibration": "oof_3fold" if config["kind"] == "pumap_oof" else "direct_train",
                    "tensorflow_gpu_available": int(bool(tf.config.list_physical_devices("GPU"))),
                    "threshold_method": method,
                    "threshold_family": threshold_family,
                    "sequence_length": target_len,
                    "train_count": len(record["train_series"]),
                    "test_size": len(y_test),
                    "anomaly_count": int(np.sum(y_test)),
                    "q_effective": q_effective,
                    "cap_target": cap_target,
                    "threshold": threshold,
                    "train_exceed_count": train_exceed_count,
                    "train_exceed_rate": train_exceed_rate,
                    "selected_indices": selected_indices(test_scores, threshold),
                    **evaluation,
                }
            )
    return rows


def write_rows(path, rows, fieldnames=None):
    if not rows:
        return fieldnames
    exists = path.exists() and path.stat().st_size > 0
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
    return fieldnames


def write_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["config_name"], row["threshold_method"])].append(row)
    summary = []
    for (config_name, threshold_method), values in sorted(grouped.items()):
        f1_values = [float(row["f1"]) for row in values]
        summary.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "config_name": config_name,
                "threshold_method": threshold_method,
                "datasets": len(values),
                "mean_f1": float(np.mean(f1_values)),
                "median_f1": float(np.median(f1_values)),
                "zero_f1": int(sum(value == 0.0 for value in f1_values)),
                "mean_fp": float(np.mean([float(row["fp"]) for row in values])),
                "mean_precision": float(
                    np.mean(
                        [
                            float(row["tp"]) / max(1.0, float(row["tp"]) + float(row["fp"]))
                            for row in values
                        ]
                    )
                ),
                "mean_recall": float(
                    np.mean(
                        [
                            float(row["tp"]) / max(1.0, float(row["tp"]) + float(row["fn"]))
                            for row in values
                        ]
                    )
                ),
            }
        )
    with SUMMARY_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def main():
    parser = argparse.ArgumentParser(description="Run ParametricUMAP normal-only OOF probe.")
    parser.add_argument("--dataset-limit", type=int)
    args = parser.parse_args()
    configure_tensorflow()
    rows = target_rows(args.dataset_limit)
    logger = logging.getLogger(EXPERIMENT_ID)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.info("Starting %s on %d datasets. TensorFlow GPUs=%s", EXPERIMENT_ID, len(rows), tf.config.list_physical_devices("GPU"))
    DETAIL_PATH.unlink(missing_ok=True)
    SUMMARY_PATH.unlink(missing_ok=True)
    all_rows = []
    fieldnames = None
    started = time.time()
    for index, difficulty_row in enumerate(rows, start=1):
        dataset_rows = evaluate_record(difficulty_row)
        fieldnames = write_rows(DETAIL_PATH, dataset_rows, fieldnames)
        all_rows.extend(dataset_rows)
        logger.info("Progress: [%3d/%3d] rows=%d dataset=%s", index, len(rows), len(all_rows), difficulty_row["dataset_name"])
    write_summary(all_rows)
    logger.info("%s finished in %.2f minutes.", EXPERIMENT_ID, (time.time() - started) / 60.0)


if __name__ == "__main__":
    main()
