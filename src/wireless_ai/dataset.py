from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .link import log_distance_path_loss, modulation_class


FEATURE_NAMES = ["distance_m", "tx_power_dbm", "path_loss_db", "shadowing_db", "snr_db"]


@dataclass(frozen=True, slots=True)
class LinkDataset:
    features: np.ndarray
    labels: np.ndarray
    feature_names: list[str]


def generate_link_dataset(samples: int = 10000, seed: int = 7) -> LinkDataset:
    if samples <= 0:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(seed)
    distance_m = rng.uniform(1.0, 500.0, samples)
    tx_power_dbm = rng.uniform(10.0, 30.0, samples)
    path_loss_db = log_distance_path_loss(distance_m)
    shadowing_db = rng.normal(0.0, 4.0, samples)
    noise_floor_dbm = -100.0
    rx_power_dbm = tx_power_dbm - path_loss_db + shadowing_db
    snr_db = rx_power_dbm - noise_floor_dbm
    features = np.column_stack(
        [distance_m, tx_power_dbm, path_loss_db, shadowing_db, snr_db]
    ).astype(np.float64)
    return LinkDataset(features, modulation_class(snr_db), FEATURE_NAMES.copy())
