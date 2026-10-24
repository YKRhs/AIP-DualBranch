from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import (
    DDE_MODEL_FILE,
    DDE_SCALER_FILE,
    META_MODEL_FILE,
    PLM_MODEL_FILE,
    PLM_SCALER_FILE,
    load_grande,
    load_joblib,
    required_artifacts,
)
from .config import configured_path, configured_paths, load_config
from .dde import extract_dde_batch
from .fasta import SequenceRecord
from .feature_cache import (
    PLMFeatureCache,
    PredictionCache,
    build_plm_signature,
    build_prediction_signature,
)
from .fusion import make_meta_features
from .plm import extract_ensemble_features


def positive_probability(values) -> np.ndarray:
    array = np.asarray(values)
    return array.astype(float) if array.ndim == 1 else array[:, 1].astype(float)


class AIPPredictor:
    def __init__(
        self,
        config_path: str | Path | None = None,
        device: str | None = None,
        progress: Callable[[str], None] | None = None,
    ):
        self.config = load_config(config_path)
        self.device = device
        self.progress = progress or (lambda message: None)
        self.artifacts_dir = configured_path(self.config, "artifacts_dir")
        required = required_artifacts(self.artifacts_dir)
        required.append(configured_path(self.config, "plm_base_model"))
        required.extend(configured_paths(self.config, "plm_fold_weights"))
        missing = [path for path in required if not path.exists()]
        if missing:
            names = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(
                "Required local model files are missing:\n"
                f"{names}\nEdit configs/inference.local.json or download the files."
            )
        self.plm_scaler = load_joblib(self.artifacts_dir / PLM_SCALER_FILE)
        self.plm_model = load_grande(self.artifacts_dir / PLM_MODEL_FILE)
        self.dde_scaler = load_joblib(self.artifacts_dir / DDE_SCALER_FILE)
        self.meta_model = load_joblib(self.artifacts_dir / META_MODEL_FILE)
        self.dde_model = None
        plm_config = self.config["plm"]
        fold_weights = configured_paths(self.config, "plm_fold_weights")
        plm_signature = build_plm_signature(
            configured_path(self.config, "plm_base_model"),
            fold_weights,
            plm_config["tnfa_sequence"],
            int(plm_config["max_length"]),
        )
        self.plm_cache = PLMFeatureCache(
            configured_path(self.config, "cache_dir"),
            plm_signature,
            int(plm_config["feature_dimension"]),
        )
        prediction_signature = build_prediction_signature(
            required_artifacts(self.artifacts_dir),
            plm_signature,
            float(self.config["threshold"]),
        )
        self.prediction_cache = PredictionCache(
            configured_path(self.config, "cache_dir"), prediction_signature
        )

    def _report(self, message: str) -> None:
        self.progress(message)

    def predict(self, records: list[SequenceRecord]) -> pd.DataFrame:
        sequences = [record.sequence for record in records]
        probabilities = [self.prediction_cache.get(sequence) for sequence in sequences]
        missing_prediction_indices = [
            index for index, value in enumerate(probabilities) if value is None
        ]
        self._report(
            "Prediction cache: "
            f"{len(sequences) - len(missing_prediction_indices)}/{len(sequences)} hit"
        )

        if missing_prediction_indices:
            uncached_sequences = [sequences[index] for index in missing_prediction_indices]
            uncached_probabilities = self._predict_uncached(uncached_sequences)
            for index, values in zip(
                missing_prediction_indices, uncached_probabilities, strict=True
            ):
                probabilities[index] = values
                self.prediction_cache.put(sequences[index], values)

        p_plm = np.array([values["p_plm"] for values in probabilities], dtype=float)
        p_dde = np.array([values["p_dde"] for values in probabilities], dtype=float)
        p_final = np.array([values["p_final"] for values in probabilities], dtype=float)
        threshold = float(self.config["threshold"])
        prediction = (p_final >= threshold).astype(int)
        self._report("Prediction complete")

        return pd.DataFrame(
            {
                "ID": [record.identifier for record in records],
                "Sequence": sequences,
                "Length": [len(sequence) for sequence in sequences],
                "Length_Note": [
                    "Outside training length range (>25 aa)"
                    if len(sequence) > 25
                    else ""
                    for sequence in sequences
                ],
                "PLM_Probability": p_plm,
                "DDE_Probability": p_dde,
                "Final_Probability": p_final,
                "Prediction": prediction,
                "Prediction_Label": np.where(prediction == 1, "AIP", "Non-AIP"),
            }
        )

    def _predict_uncached(self, sequences: list[str]) -> list[dict[str, float]]:
        self._report("Extracting 400-dimensional DDE features")
        dde_features = extract_dde_batch(sequences)

        plm_config = self.config["plm"]
        cached = [self.plm_cache.get(sequence) for sequence in sequences]
        missing_indices = [index for index, value in enumerate(cached) if value is None]
        self._report(
            f"PLM cache: {len(sequences) - len(missing_indices)}/{len(sequences)} hit"
        )
        if missing_indices:
            self._report("Extracting five-fold averaged PLM features")
            missing_sequences = [sequences[index] for index in missing_indices]
            extracted = extract_ensemble_features(
                sequences=missing_sequences,
                base_model_path=configured_path(self.config, "plm_base_model"),
                fold_weights=configured_paths(self.config, "plm_fold_weights"),
                tnfa_sequence=plm_config["tnfa_sequence"],
                max_length=int(plm_config["max_length"]),
                batch_size=int(plm_config["batch_size"]),
                device=self.device,
                progress=self.progress,
            )
            for index, feature in zip(missing_indices, extracted, strict=True):
                cached[index] = feature
                self.plm_cache.put(sequences[index], feature)
        plm_features = np.stack(cached).astype(np.float32)

        self._report("Running PLM and DDE classifiers")
        plm_scaled = self.plm_scaler.transform(plm_features).astype(np.float32)
        p_plm = positive_probability(
            self.plm_model.predict_proba(pd.DataFrame(plm_scaled))
        )

        if self.dde_model is None:
            from tabicl import TabICLClassifier

            self.dde_model = TabICLClassifier.load(
                self.artifacts_dir / DDE_MODEL_FILE,
                device=self.device,
            )
            offload_dir = self.artifacts_dir / "tabicl_offload"
            offload_dir.mkdir(parents=True, exist_ok=True)
            self.dde_model.disk_offload_dir = str(offload_dir)
            self.dde_model._build_inference_config()
        dde_scaled = self.dde_scaler.transform(dde_features).astype(np.float32)
        p_dde = positive_probability(self.dde_model.predict_proba(dde_scaled))

        self._report("Combining the two branch probabilities")
        p_final = positive_probability(
            self.meta_model.predict_proba(make_meta_features(p_plm, p_dde))
        )
        return [
            {"p_plm": plm, "p_dde": dde, "p_final": final}
            for plm, dde, final in zip(p_plm, p_dde, p_final, strict=True)
        ]
