from __future__ import annotations


def estimate_market(population: int, penetration: float) -> dict[str, float]:
    market = population * penetration
    return {
        'population': float(population),
        'penetration': float(penetration),
        'market': float(market),
    }
