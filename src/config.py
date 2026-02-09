from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class Scenario(BaseModel):
    name: str = 'base'
    horizon_years: int = 5
    discount_rate: float = Field(0.12, ge=0.0, le=1.0)
    population_catchment: int = 0
    market_share_target: float = Field(0.05, ge=0.0, le=1.0)
    avg_ticket_cop: float = 0.0
    fixed_costs_cop_year: float = 0.0
    variable_cost_rate: float = Field(0.55, ge=0.0, le=1.0)


class MonteCarlo(BaseModel):
    runs: int = 2000
    seed: int = 42


class Config(BaseModel):
    scenario: Scenario
    monte_carlo: MonteCarlo


def load_config(path: str | Path) -> Config:
    data: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    return Config(**data)
