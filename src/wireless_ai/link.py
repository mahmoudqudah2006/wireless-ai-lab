from __future__ import annotations

import numpy as np


def log_distance_path_loss(
    distance_m: np.ndarray,
    *,
    reference_loss_db: float = 40.0,
    exponent: float = 2.7,
    reference_distance_m: float = 1.0,
) -> np.ndarray:
    distance = np.asarray(distance_m, dtype=float)
    if np.any(distance < reference_distance_m):
        raise ValueError("distance must be >= reference_distance_m")
    return reference_loss_db + 10.0 * exponent * np.log10(distance / reference_distance_m)


def modulation_class(snr_db: np.ndarray) -> np.ndarray:
    snr = np.asarray(snr_db, dtype=float)
    labels = np.zeros(snr.shape, dtype=np.int64)
    labels[snr >= 4.0] = 1
    labels[snr >= 10.0] = 2
    labels[snr >= 18.0] = 3
    return labels


CLASS_NAMES = ["BPSK", "QPSK", "16-QAM", "64-QAM"]
