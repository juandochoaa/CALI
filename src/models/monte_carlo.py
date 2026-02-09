from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from numba import njit


@njit
def simulate_one(mu: float, sigma: float, seed: int) -> float:
    np.random.seed(seed)
    return np.random.normal(mu, sigma)


def run_monte_carlo(mu: float, sigma: float, runs: int, n_jobs: int = -1) -> np.ndarray:
    results = Parallel(n_jobs=n_jobs)(delayed(simulate_one)(mu, sigma, i) for i in range(runs))
    return np.array(results, dtype=float)
