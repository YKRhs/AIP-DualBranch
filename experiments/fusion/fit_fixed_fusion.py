from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
THRESHOLD = 0.5
MODEL_PARAMETERS = {
    "C": 0.11,
    "class_weight": {0: 1.0, 1: 1.56},
    "penalty": "l2",
    "solver": "liblinear",
    "intercept_scaling": 2.0,
    "max_iter": 5000,
    "random_state": 42,
}
METRICS = ("AUC", "ACC", "MCC", "Sn", "Sp", "Pre", "F1")


def load_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"Sample_Index", "Label", "PLM_GRANDE", "DDE_TabICLv2"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if not np.array_equal(frame["Sample_Index"].to_numpy(), np.arange(len(frame))):
        raise ValueError(f"{path} is not aligned by Sample_Index")
    return frame


def make_features(frame: pd.DataFrame) -> np.ndarray:
    p_plm = frame["PLM_GRANDE"].to_numpy(float)
    p_dde = frame["DDE_TabICLv2"].to_numpy(float)
    return np.column_stack(
        (
            p_plm,
            p_dde,
            np.abs(p_plm - p_dde),
            (p_plm + p_dde) / 2.0,
            p_plm * p_dde,
            np.abs(p_plm - 0.5),
            np.abs(p_dde - 0.5),
        )
    )


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(**MODEL_PARAMETERS))


def calculate_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "AUC": float(roc_auc_score(labels, probabilities)),
        "ACC": float(accuracy_score(labels, predictions)),
        "MCC": float(matthews_corrcoef(labels, predictions)),
        "Sn": float(tp / (tp + fn)),
        "Sp": float(tn / (tn + fp)),
        "Pre": float(tp / (tp + fp)),
        "F1": float(f1_score(labels, predictions)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the fixed fusion stage")
    parser.add_argument(
        "--oof",
        type=Path,
        default=ROOT / "evaluation/base_probabilities/training_oof.csv",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=ROOT / "evaluation/base_probabilities/independent_test.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/fusion_reproduction"
    )
    args = parser.parse_args()

    oof = load_table(args.oof)
    test = load_table(args.test)
    x_train, y_train = make_features(oof), oof["Label"].to_numpy(int)
    x_test, y_test = make_features(test), test["Label"].to_numpy(int)

    fold_rows = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, valid_index) in enumerate(
        splitter.split(x_train, y_train), start=1
    ):
        model = make_model()
        model.fit(x_train[train_index], y_train[train_index])
        probabilities = model.predict_proba(x_train[valid_index])[:, 1]
        fold_rows.append(
            {
                "Evaluation": f"Fold {fold}",
                **calculate_metrics(y_train[valid_index], probabilities),
            }
        )

    summary = {
        "Evaluation": "Mean fixed 5-fold CV",
        **{
            metric: float(np.mean([row[metric] for row in fold_rows]))
            for metric in METRICS
        },
    }

    final_model = make_model()
    final_model.fit(x_train, y_train)
    test_probabilities = final_model.predict_proba(x_test)[:, 1]
    test_predictions = (test_probabilities >= THRESHOLD).astype(int)
    test_metrics = {
        "Evaluation": "Independent test",
        **calculate_metrics(y_test, test_probabilities),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(args.output / "fold_metrics.csv", index=False)
    pd.DataFrame([summary, test_metrics]).to_csv(
        args.output / "summary_metrics.csv", index=False
    )
    pd.DataFrame(
        {
            "Sample_Index": test["Sample_Index"],
            "Label": y_test,
            "Probability": test_probabilities,
            "Prediction": test_predictions,
        }
    ).to_csv(args.output / "independent_test_predictions.csv", index=False)
    joblib.dump(final_model, args.output / "fusion_model.joblib")

    print(pd.DataFrame([summary, test_metrics]).to_string(index=False))
    print(f"Saved fusion reconstruction to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
