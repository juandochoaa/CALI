from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl


def estimate_market(population: int, penetration: float) -> dict[str, float]:
    market = population * penetration
    return {
        'population': float(population),
        'penetration': float(penetration),
        'market': float(market),
    }
