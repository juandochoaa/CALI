from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.ips_kpis import build_ips_historical_kpis, compute_market_share_by_year


def test_compute_market_share_by_year_sums_to_one_per_year() -> None:
    ing_oper_df = pd.DataFrame(
        {
            "entity": ["IPS A", "IPS B", "IPS A", "IPS B"],
            "year": [2023, 2023, 2024, 2024],
            "REV": [300.0, 700.0, 400.0, 600.0],
        }
    )
    out = compute_market_share_by_year(ing_oper_df)
    by_year = out.groupby("year", as_index=False)["MarketShare"].sum()
    for row in by_year.itertuples(index=False):
        assert np.isclose(float(row.MarketShare), 1.0)


def test_build_ips_historical_kpis_has_expected_columns() -> None:
    ing_oper_df = pd.DataFrame(
        {
            "entity": ["IPS A", "IPS B", "IPS A", "IPS B"],
            "year": [2023, 2023, 2024, 2024],
            "REV": [300.0, 700.0, 400.0, 600.0],
        }
    )
    ratios_df = pd.DataFrame(
        {
            "entity": ["IPS A", "IPS A"],
            "year": [2023, 2024],
            "gross_margin": [0.20, 0.22],
            "net_margin": [0.08, 0.10],
        }
    )

    out = build_ips_historical_kpis(
        ratios_df=ratios_df,
        ing_oper_df=ing_oper_df,
        selected_ips="IPS A",
        years_universe=[2023, 2024],
    )
    assert list(out.columns) == ["Año", "MargenBruto", "MargenNeto", "TamanioMercado"]
    assert len(out) == 2
    assert np.isclose(float(out.loc[out["Año"] == 2023, "TamanioMercado"].iloc[0]), 0.30)
    assert np.isclose(float(out.loc[out["Año"] == 2024, "TamanioMercado"].iloc[0]), 0.40)
