from __future__ import annotations

import numpy as np


def npv(cashflows: np.ndarray, discount_rate: float) -> float:
    return float(np.sum(cashflows / (1 + discount_rate) ** np.arange(len(cashflows))))
