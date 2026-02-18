from __future__ import annotations

from pathlib import Path

from src.models.eps_montecarlo import PROBABILITY_COLUMNS


def test_probability_columns_excludes_cashdays_zero() -> None:
    assert PROBABILITY_COLUMNS == [
        "P_avg_CM_ratio_lt_1",
        "P_avg_PA_ratio_lt_1",
        "P_avg_RI_ratio_lt_1",
        "P_avg_CashDays_lt_15",
    ]


def test_clientes_page_does_not_reference_cashdays_zero() -> None:
    page_path = Path("dashboards/pages/1_Clientes.py")
    content = page_path.read_text(encoding="utf-8", errors="ignore")
    assert "P_avg_CashDays_lt_0" not in content
    assert "cash_thresholds=(15, 0)" not in content
