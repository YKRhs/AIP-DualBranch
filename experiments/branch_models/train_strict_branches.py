from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "experiments/sampling") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments/sampling"))

from aip_predictor.grande_compat import get_grande_class  # noqa: E402
from qc_smote_exact import QCSMOTEExact  # noqa: E402


RANDOM_STATE = 42
N_SPLITS = 5
THRESHOLD = 0.5
METRICS = ("AUC", "ACC", "MCC", "Sn", "Sp", "Pre", "F1")

QC_PARAMETERS = {
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
    "random_state": RANDOM_STATE,
}

# GRANDE accesses a complete parameter dictionary internally. These defaults
# match the established numeric-data configuration before final overrides.
GRANDE_PARAMETERS = {
    "depth": 3,
    "n_estimators": 128,
    "learning_rate_weights": 0.0005,
    "learning_rate_index": 0.005,
    "learning_rate_values": 0.03,
    "learning_rate_leaf": 0.03,
    "learning_rate_embedding": 0.02,
    "use_category_embeddings": False,
    "embedding_dim_cat": 8,
    "use_numeric_embeddings": False,
    "embedding_dim_num": 8,
    "embedding_threshold": 1,
    "loo_cardinality": 10,
    "dropout": 0.30,
    "selected_variables": 50,
    "data_subset_fraction": 1.0,
    "bootstrap": False,
    "missing_values": False,
    "optimizer": "adam",
    "cosine_decay_restarts": False,
    "reduce_on_plateau_scheduler": True,
    "label_smoothing": 0.0,
    "use_class_weights": False,
    "focal_loss": False,
    "swa": False,
    "es_metric": True,
    "epochs": 80,
    "batch_size": 256,
    "early_stopping_epochs": 15,
    "use_freq_enc": False,
    "use_robust_scale_smoothing": False,
    "problem_type": "binary",
    "random_seed": RANDOM_STATE,
    "verbose": 0,
}

TABICL_PARAMETERS = {
    "n_estimators": 8,
    "softmax_temperature": 0.9,
    "average_logits": True,
    "checkpoint_version": "tabicl-classifier-v2-20260212.ckpt",
    "random_state": RANDOM_STATE,
    "verbose": False,
}


def positive_probability(values) -> np.ndarray:
    array = np.asarray(values)
    return array.astype(float) if array.ndim == 1 else array[:, 1].astype(float)


def calculate_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "AUC": float(roc_auc_score(labels, probabilities)),
        "ACC": float(accuracy_score(labels, predictions)),
        "MCC": float(matthews_corrcoef(labels, predictions)),
        "Sn": float(tp / (tp + fn)),
        "Sp": float(tn / (tn + fp)),
        "Pre": float(tp / (tp + fp)) if tp + fp else 0.0,
        "F1": float(f1_score(labels, predictions, zero_division=0)),
    }


def save_probability_table(
    path: Path,
    labels: np.ndarray,
    probabilities: np.ndarray,
    probability_name: str,
) -> None:
    pd.DataFrame(
        {
            "Sample_Index": np.arange(len(labels)),
            "Label": np.asarray(labels, dtype=int),
            probability_name: np.asarray(probabilities, dtype=float),
        }
    ).to_csv(path, index=False, float_format="%.8f")


def sampler() -> QCSMOTEExact:
    return QCSMOTEExact(**QC_PARAMETERS)


def fit_grande(
    features: np.ndarray,
    labels: np.ndarray,
    evaluation_features: np.ndarray,
):
    # GRANDE early stopping uses 15% of the untouched outer-training rows.
    # The split occurs before scaler fitting and before synthetic sampling.
    X_inner, X_early, y_inner, y_early = train_test_split(
        features,
        labels,
        test_size=0.15,
        stratify=labels,
        random_state=RANDOM_STATE,
    )
    scaler = StandardScaler()
    X_inner_scaled = scaler.fit_transform(X_inner).astype(np.float32)
    X_early_scaled = scaler.transform(X_early).astype(np.float32)
    X_evaluation_scaled = scaler.transform(evaluation_features).astype(np.float32)
    X_fit, y_fit = sampler().fit_resample(X_inner_scaled, y_inner)

    model = get_grande_class()(params=dict(GRANDE_PARAMETERS))
    model.fit(
        X=pd.DataFrame(np.asarray(X_fit, dtype=np.float32)),
        y=pd.Series(np.asarray(y_fit, dtype=int)),
        X_val=pd.DataFrame(X_early_scaled),
        y_val=pd.Series(np.asarray(y_early, dtype=int)),
    )
    probability = positive_probability(
        model.predict_proba(pd.DataFrame(X_evaluation_scaled))
    )
    return probability, scaler, model


def train_plm_branch(feature_directory: Path, output: Path, save_models: bool) -> None:
    labels = np.load(feature_directory / "train_labels.npy", allow_pickle=False).astype(int)
    test_labels = np.load(feature_directory / "test_labels.npy", allow_pickle=False).astype(int)
    oof_probability = np.full(len(labels), np.nan, dtype=float)
    fold_rows = []

    for fold in range(1, N_SPLITS + 1):
        fold_directory = feature_directory / f"fold{fold}"
        X_train = np.load(fold_directory / "X_train_plm.npy", allow_pickle=False)
        y_train = np.load(fold_directory / "y_train.npy", allow_pickle=False).astype(int)
        X_valid = np.load(fold_directory / "X_val_plm.npy", allow_pickle=False)
        y_valid = np.load(fold_directory / "y_val.npy", allow_pickle=False).astype(int)
        valid_index = np.load(fold_directory / "val_idx.npy", allow_pickle=False).astype(int)
        probability, _, model = fit_grande(X_train, y_train, X_valid)
        if not np.array_equal(labels[valid_index], y_valid):
            raise ValueError(f"PLM fold {fold} labels are not aligned")
        oof_probability[valid_index] = probability
        metrics = calculate_metrics(y_valid, probability)
        fold_rows.append({"Branch": "PLM-GRANDE", "Fold": fold, **metrics})
        print(
            f"[PLM-GRANDE] fold {fold}/{N_SPLITS} "
            f"MCC={metrics['MCC']:.4f} AUC={metrics['AUC']:.4f}",
            flush=True,
        )
        del model
        gc.collect()

    if np.isnan(oof_probability).any():
        raise RuntimeError("PLM OOF probability vector is incomplete")

    X_full = np.load(feature_directory / "ensemble_train_plm.npy", allow_pickle=False)
    X_test = np.load(feature_directory / "ensemble_test_plm.npy", allow_pickle=False)
    test_probability, final_scaler, final_model = fit_grande(X_full, labels, X_test)
    save_probability_table(output / "plm_oof.csv", labels, oof_probability, "PLM_GRANDE")
    save_probability_table(output / "plm_test.csv", test_labels, test_probability, "PLM_GRANDE")
    pd.DataFrame(fold_rows).to_csv(output / "plm_fold_metrics.csv", index=False)

    summary = [
        {
            "Branch": "PLM-GRANDE",
            "Evaluation": "Mean strict 5-fold metrics",
            **{
                metric: float(np.mean([row[metric] for row in fold_rows]))
                for metric in METRICS
            },
        },
        {
            "Branch": "PLM-GRANDE",
            "Evaluation": "Pooled training OOF",
            **calculate_metrics(labels, oof_probability),
        },
        {
            "Branch": "PLM-GRANDE",
            "Evaluation": "Independent test",
            **calculate_metrics(test_labels, test_probability),
        },
    ]
    pd.DataFrame(summary).to_csv(output / "plm_summary_metrics.csv", index=False)
    if save_models:
        artifact_directory = output / "artifacts"
        artifact_directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_scaler, artifact_directory / "plm_scaler.joblib", compress=3)
        with (artifact_directory / "plm_grande.pkl").open("wb") as handle:
            pickle.dump(final_model, handle, protocol=5)
    del final_model
    gc.collect()


def build_tabicl(args: argparse.Namespace, fold_name: str):
    from tabicl import TabICLClassifier

    offload_directory = args.output / "tabicl_offload" / fold_name
    offload_directory.mkdir(parents=True, exist_ok=True)
    parameters = {
        **TABICL_PARAMETERS,
        "device": args.device,
        "batch_size": args.tabicl_batch_size,
        "offload_mode": "auto",
        "disk_offload_dir": str(offload_directory),
    }
    if args.tabicl_kv_cache != "none":
        parameters["kv_cache"] = args.tabicl_kv_cache
    return TabICLClassifier(**parameters)


def fit_tabicl(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_evaluation: np.ndarray,
    args: argparse.Namespace,
    fold_name: str,
):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_evaluation_scaled = scaler.transform(X_evaluation).astype(np.float32)
    X_fit, y_fit = sampler().fit_resample(X_train_scaled, y_train)
    model = build_tabicl(args, fold_name)
    model.fit(np.asarray(X_fit, dtype=np.float32), np.asarray(y_fit, dtype=int))
    probability = positive_probability(model.predict_proba(X_evaluation_scaled))
    return probability, scaler, model


def clear_torch_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def train_dde_branch(
    feature_directory: Path,
    output: Path,
    args: argparse.Namespace,
    save_models: bool,
) -> None:
    X_train = np.load(feature_directory / "train_dde.npy", allow_pickle=False)
    X_test = np.load(feature_directory / "test_dde.npy", allow_pickle=False)
    labels = np.load(feature_directory / "train_labels.npy", allow_pickle=False).astype(int)
    test_labels = np.load(feature_directory / "test_labels.npy", allow_pickle=False).astype(int)
    splitter = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    oof_probability = np.full(len(labels), np.nan, dtype=float)
    fold_rows = []

    for fold, (train_index, valid_index) in enumerate(
        splitter.split(X_train, labels), start=1
    ):
        probability, _, model = fit_tabicl(
            X_train[train_index],
            labels[train_index],
            X_train[valid_index],
            args,
            f"fold{fold}",
        )
        oof_probability[valid_index] = probability
        metrics = calculate_metrics(labels[valid_index], probability)
        fold_rows.append({"Branch": "DDE-TabICLv2", "Fold": fold, **metrics})
        print(
            f"[DDE-TabICLv2] fold {fold}/{N_SPLITS} "
            f"MCC={metrics['MCC']:.4f} AUC={metrics['AUC']:.4f}",
            flush=True,
        )
        del model
        clear_torch_cache()

    test_probability, final_scaler, final_model = fit_tabicl(
        X_train, labels, X_test, args, "full_train"
    )
    save_probability_table(
        output / "dde_oof.csv", labels, oof_probability, "DDE_TabICLv2"
    )
    save_probability_table(
        output / "dde_test.csv", test_labels, test_probability, "DDE_TabICLv2"
    )
    pd.DataFrame(fold_rows).to_csv(output / "dde_fold_metrics.csv", index=False)
    summary = [
        {
            "Branch": "DDE-TabICLv2",
            "Evaluation": "Mean strict 5-fold metrics",
            **{
                metric: float(np.mean([row[metric] for row in fold_rows]))
                for metric in METRICS
            },
        },
        {
            "Branch": "DDE-TabICLv2",
            "Evaluation": "Pooled training OOF",
            **calculate_metrics(labels, oof_probability),
        },
        {
            "Branch": "DDE-TabICLv2",
            "Evaluation": "Independent test",
            **calculate_metrics(test_labels, test_probability),
        },
    ]
    pd.DataFrame(summary).to_csv(output / "dde_summary_metrics.csv", index=False)
    if save_models:
        artifact_directory = output / "artifacts"
        artifact_directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_scaler, artifact_directory / "dde_scaler.joblib", compress=3)
        final_model.save(
            artifact_directory / "dde_tabicl.pkl",
            save_model_weights=True,
            save_training_data=True,
            save_kv_cache=args.tabicl_kv_cache != "none",
        )
    del final_model
    clear_torch_cache()


def merge_branch_probabilities(output: Path) -> bool:
    required = (
        output / "plm_oof.csv",
        output / "plm_test.csv",
        output / "dde_oof.csv",
        output / "dde_test.csv",
    )
    if not all(path.exists() for path in required):
        return False

    def merge(left_path: Path, right_path: Path) -> pd.DataFrame:
        left = pd.read_csv(left_path)
        right = pd.read_csv(right_path)
        frame = left.merge(
            right,
            on=["Sample_Index", "Label"],
            how="inner",
            validate="one_to_one",
        )
        if len(frame) != len(left) or len(frame) != len(right):
            raise ValueError("PLM and DDE probability rows are not aligned")
        return frame

    merge(output / "plm_oof.csv", output / "dde_oof.csv").to_csv(
        output / "training_oof.csv", index=False
    )
    merge(output / "plm_test.csv", output / "dde_test.csv").to_csv(
        output / "independent_test.csv", index=False
    )
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the two finalized branches with strict five-fold OOF evaluation"
    )
    parser.add_argument("--branch", choices=["plm", "dde", "both"], default="both")
    parser.add_argument(
        "--plm-features",
        type=Path,
        default=ROOT / "results/features/plm_strict_folds",
    )
    parser.add_argument(
        "--dde-features", type=Path, default=ROOT / "results/features"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/branch_models"
    )
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu")
    parser.add_argument("--tabicl-batch-size", type=int, default=1)
    parser.add_argument(
        "--tabicl-kv-cache", choices=["none", "kv", "repr"], default="repr"
    )
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        if not merge_branch_probabilities(args.output):
            raise FileNotFoundError("Both PLM and DDE branch probability files are required")
        print(f"Merged branch probabilities in: {args.output.resolve()}")
        return

    if args.branch in {"plm", "both"}:
        train_plm_branch(args.plm_features, args.output, args.save_models)
    if args.branch in {"dde", "both"}:
        train_dde_branch(args.dde_features, args.output, args, args.save_models)
    merged = merge_branch_probabilities(args.output)
    manifest = {
        "random_state": RANDOM_STATE,
        "folds": N_SPLITS,
        "threshold": THRESHOLD,
        "qc_parameters": QC_PARAMETERS,
        "grande_parameters": GRANDE_PARAMETERS,
        "tabicl_parameters": TABICL_PARAMETERS,
        "merged_probability_tables_ready": merged,
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Saved branch results to: {args.output.resolve()}")
    if merged:
        print("Combined training_oof.csv and independent_test.csv are ready for fusion.")


if __name__ == "__main__":
    main()
