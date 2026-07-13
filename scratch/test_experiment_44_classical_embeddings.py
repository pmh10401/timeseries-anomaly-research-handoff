import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_original_improvement_experiment import (
    classical_embedding_pair,
    fft_magnitude_features,
    get_spec,
    haar_wavelet_features,
    original_score_pair_for_config,
    pca_embedding_pair,
)


EXP_ID = "experiment_44_classical_embedding_baselines"


def test_experiment_44_spec_compares_classical_embeddings():
    spec = get_spec(EXP_ID)
    names = {config["name"] for config in spec["configs"]}
    embeddings = {config["embedding"] for config in spec["configs"]}

    assert spec["source_experiment_id"] == "paper_time_series_embedding_review"
    assert {"fft", "pca", "haar_wavelet"} <= embeddings
    assert "fft_mag_16_knn3" in names
    assert "pca_16_knn3" in names
    assert "haar_wavelet_32_knn3" in names
    assert "family_guard_v1" in spec["thresholds"]


def test_fft_magnitude_features_are_fixed_width_and_finite():
    X = np.array(
        [
            [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0],
            [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
        ],
        dtype=np.float32,
    )

    features = fft_magnitude_features(X, n_features=6)

    assert features.shape == (2, 6)
    assert np.isfinite(features).all()
    assert features[0].max() > 0


def test_pca_embedding_pair_fits_train_space_and_preserves_rows():
    X_train = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    X_test = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

    train_features, test_features = pca_embedding_pair(X_train, X_test, n_components=2)

    assert train_features.shape == (3, 2)
    assert test_features.shape == (1, 2)
    assert np.isfinite(train_features).all()
    assert np.isfinite(test_features).all()


def test_haar_wavelet_features_are_fixed_width_and_finite():
    X = np.array([[1.0, 2.0, 4.0, 8.0], [8.0, 4.0, 2.0, 1.0]], dtype=np.float32)

    features = haar_wavelet_features(X, n_features=5)

    assert features.shape == (2, 5)
    assert np.isfinite(features).all()
    assert not np.allclose(features[0], features[1])


def test_classical_embedding_score_pair_returns_train_and_test_scores():
    X_train = np.array(
        [
            [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0],
            [0.1, 1.1, 0.1, -0.9, 0.1, 1.1, 0.1, -0.9],
            [1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    X_test = np.array([[3.0, 3.0, -3.0, -3.0, 3.0, 3.0, -3.0, -3.0]], dtype=np.float32)
    config = {"kind": "classical_embedding_knn", "embedding": "fft", "n_features": 4, "neighbors": 2}

    train_features, test_features = classical_embedding_pair(X_train, X_test, config)
    train_scores, test_scores = original_score_pair_for_config(X_train, X_test, 8, config)

    assert train_features.shape == (3, 4)
    assert test_features.shape == (1, 4)
    assert train_scores.shape == (3,)
    assert test_scores.shape == (1,)
    assert np.isfinite(train_scores).all()
    assert np.isfinite(test_scores).all()


if __name__ == "__main__":
    test_experiment_44_spec_compares_classical_embeddings()
    test_fft_magnitude_features_are_fixed_width_and_finite()
    test_pca_embedding_pair_fits_train_space_and_preserves_rows()
    test_haar_wavelet_features_are_fixed_width_and_finite()
    test_classical_embedding_score_pair_returns_train_and_test_scores()
    print("experiment 44 classical embedding tests passed")
