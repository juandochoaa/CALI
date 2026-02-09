from __future__ import annotations

from typing import Any

import numpy as np

from .financials import npv
from .market_sizing import estimate_market
from .monte_carlo import run_monte_carlo


def run_all(
    population: int,
    penetration: float,
    cashflows: np.ndarray,
    discount_rate: float,
) -> dict[str, Any]:
    market = estimate_market(population, penetration)
    value = npv(cashflows, discount_rate)
    sims = run_monte_carlo(0.0, 1.0, runs=1000)
    return {
        'market': market,
        'npv': value,
        'mc_mean': float(sims.mean()),
        'mc_p05': float(np.percentile(sims, 5)),
        'mc_p95': float(np.percentile(sims, 95)),
    }
