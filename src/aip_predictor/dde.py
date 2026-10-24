from __future__ import annotations

from itertools import product

import numpy as np


AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
CODON_USAGE = {
    "A": 4, "C": 2, "D": 2, "E": 2, "F": 2,
    "G": 4, "H": 2, "I": 3, "K": 2, "L": 6,
    "M": 1, "N": 2, "P": 4, "Q": 2, "R": 6,
    "S": 6, "T": 4, "V": 4, "W": 1, "Y": 2,
}
TOTAL_CODONS = 61
DIPEPTIDES = tuple(a + b for a, b in product(AA_ORDER, repeat=2))


def extract_dde(sequence: str) -> np.ndarray:
    length = len(sequence)
    if length <= 1:
        return np.zeros(400, dtype=np.float32)
    denominator = length - 1
    counts = dict.fromkeys(DIPEPTIDES, 0)
    for index in range(denominator):
        counts[sequence[index : index + 2]] += 1

    values = np.empty(400, dtype=np.float32)
    for index, dipeptide in enumerate(DIPEPTIDES):
        observed = counts[dipeptide] / denominator
        expected = (
            CODON_USAGE[dipeptide[0]] / TOTAL_CODONS
        ) * (CODON_USAGE[dipeptide[1]] / TOTAL_CODONS)
        variance = expected * (1.0 - expected) / denominator
        values[index] = (
            (observed - expected) / np.sqrt(variance) if variance > 0 else 0.0
        )
    return values


def extract_dde_batch(sequences: list[str]) -> np.ndarray:
    return np.stack([extract_dde(sequence) for sequence in sequences]).astype(
        np.float32
    )
