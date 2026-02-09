from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def score_sites(features: np.ndarray) -> np.ndarray:
    scaler = MinMaxScaler()
    return scaler.fit_transform(features)
