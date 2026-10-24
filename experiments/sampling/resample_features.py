from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from qc_smote_exact import QCSMOTEExact


SETTINGS = {
    "plm": {
        "sampling_strategy": 1.0,
        "k_neighbors": 3,
        "best_of_k": 4,
        "overshoot_factor": 1.0,
        "purity_threshold": 0.3,
        "rho0": 0.9,
        "w_safe": 0.50,
        "w_density": 0.25,
        "w_isolation": 0.25,
        "distance_metric": "minkowski",
        "distance_p": 1.5,
        "candidate_selection_mode": "weighted_sum",
        "purity_mode": "uniform",
        "minority_neighbor_scope": "kmeans_cluster_outlier_fallback",
        "kmeans_n_clusters": 8,
        "kmeans_n_init": 10,
        "cluster_outlier_fallback_quantile": 0.96,
        "random_state": 42,
    },
    "dde": {
        "sampling_strategy": 1.0,
        "k_neighbors": 3,
        "best_of_k": 4,
        "overshoot_factor": 1.0,
        "purity_threshold": 0.3,
        "rho0": 0.9,
        "w_safe": 0.50,
        "w_density": 0.25,
        "w_isolation": 0.25,
        "distance_metric": "minkowski",
        "distance_p": 1.5,
        "candidate_selection_mode": "weighted_sum",
        "purity_mode": "uniform",
        "minority_neighbor_scope": "kmeans_cluster_outlier_fallback",
        "kmeans_n_clusters": 8,
        "kmeans_n_init": 10,
        "cluster_outlier_fallback_quantile": 0.96,
        "random_state": 42,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scale and QC-SMOTE-resample one training partition"
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--branch", choices=sorted(SETTINGS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    features = np.load(args.features, allow_pickle=False)
    labels = np.load(args.labels, allow_pickle=False).astype(int)
    if len(features) != len(labels):
        raise ValueError("Feature and label row counts differ")
    if not np.isfinite(features).all():
        raise ValueError("Feature matrix contains NaN or infinite values")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features).astype(np.float32)
    sampler = QCSMOTEExact(**SETTINGS[args.branch])
    resampled_features, resampled_labels = sampler.fit_resample(scaled, labels)

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "features.npy", np.asarray(resampled_features, dtype=np.float32))
    np.save(args.output / "labels.npy", np.asarray(resampled_labels, dtype=np.int64))
    joblib.dump(scaler, args.output / "scaler.joblib")
    print(
        f"{args.branch}: {features.shape} -> {resampled_features.shape}; "
        f"saved to {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
