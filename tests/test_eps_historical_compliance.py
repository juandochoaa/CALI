from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.eps_montecarlo import build_eps_historical_compliance


def test_build_eps_historical_compliance_calculates_ratios_and_flags() -> None:
    base_eps = pd.DataFrame(
        {
            "EPS": ["ASMETSALUD EPS", "ASMETSALUD EPS"],
            "AÃ±o": [2024, 2025],
            "TotalIngresoOperativo": [100000.0, 90000.0],
            "Otroscostospornaturaleza": [60000.0, 0.0],
            "Totaldepatrimonio": [20000.0, 15000.0],
            "EfectivooEquivalentes": [10000.0, 2000.0],
            "Activosfinancierosdecortoplazo": [5000.0, 0.0],
            "Provisionesparaotrospasivosygastos": [12000.0, 0.0],
        }
    )

    out = build_eps_historical_compliance(base_eps=base_eps, eps_name="ASMETSALUD EPS")
    assert out.columns[1:].tolist() == [
        "CM_ratio",
        "PA_ratio",
        "RI_ratio",
        "Cumple_CM",
        "Cumple_PA",
        "Cumple_RI",
        "Cumple_3_de_3",
    ]
    assert len(out) == 2

    first = out.iloc[0]
    # ASMETSALUD usa capmin 17.800
    assert np.isclose(float(first["CM_ratio"]), 20000.0 / 17800.0)
    assert np.isclose(float(first["PA_ratio"]), 20000.0 / (0.08 * 100000.0 * 0.6))
    assert np.isclose(float(first["RI_ratio"]), (10000.0 + 5000.0) / 12000.0)
    assert bool(first["Cumple_3_de_3"]) is True

    second = out.iloc[1]
    assert np.isnan(float(second["PA_ratio"]))
    assert np.isnan(float(second["RI_ratio"]))
    assert bool(second["Cumple_PA"]) is False
    assert bool(second["Cumple_RI"]) is False
    assert bool(second["Cumple_3_de_3"]) is False
