from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from aip_predictor.fasta import read_fasta


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_split_counts_and_no_overlap() -> None:
    train = read_fasta(ROOT / "data/splits/AIP_train.fasta")
    test = read_fasta(ROOT / "data/splits/AIP_test.fasta")
    train_labels = np.array([int(row.identifier.split("|")[1]) for row in train])
    test_labels = np.array([int(row.identifier.split("|")[1]) for row in test])

    assert (len(train), int(train_labels.sum())) == (3775, 1498)
    assert (len(test), int(test_labels.sum())) == (419, 180)
    assert set(row.sequence for row in train).isdisjoint(row.sequence for row in test)


def test_archived_predictions_reproduce_reported_metrics() -> None:
    frame = pd.read_csv(ROOT / "evaluation/independent_test_predictions.csv")
    labels = frame["Label"].to_numpy(int)
    probabilities = frame["Final_Probability"].to_numpy(float)
    predictions = frame["Prediction"].to_numpy(int)

    actual = np.array(
        [
            roc_auc_score(labels, probabilities),
            accuracy_score(labels, predictions),
            matthews_corrcoef(labels, predictions),
            recall_score(labels, predictions),
            recall_score(labels, predictions, pos_label=0),
            precision_score(labels, predictions),
            f1_score(labels, predictions),
        ]
    )
    expected = pd.read_csv(ROOT / "evaluation/final_model_metrics.csv").loc[
        0, ["AUC", "ACC", "MCC", "Sn", "Sp", "Pre", "F1"]
    ].to_numpy(dtype=float)

    np.testing.assert_allclose(actual, expected, atol=5e-10, rtol=0)
