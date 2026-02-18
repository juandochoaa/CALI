from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.eps_montecarlo import build_composite_ranking, compute_reclamos_score


def test_compute_reclamos_score_homologa_eps_aliases() -> None:
    reclamos_df = pd.DataFrame(
        {
            "EPS": ["SURA", "SANITAS", "SALUD TOTAL"],
            "TASA X CADA 10.000 AFILIADOS": [350.0, 500.0, 650.0],
            "RECLAMOS": [100, 200, 300],
        }
    )
    eps_obj = ["EPS SURA", "EPS SANITAS", "SALUD TOTAL EPS"]

    out = compute_reclamos_score(reclamos_df=reclamos_df, eps_obj=eps_obj)
    out = out.sort_values("EPS").reset_index(drop=True)

    assert out["EPS"].tolist() == sorted(eps_obj)
    assert bool(out["reclamos_imputado"].any()) is False

    tasa_map = dict(zip(out["EPS"], out["Tasa_Reclamos_10k"], strict=False))
    assert np.isclose(tasa_map["EPS SURA"], 350.0)
    assert np.isclose(tasa_map["EPS SANITAS"], 500.0)
    assert np.isclose(tasa_map["SALUD TOTAL EPS"], 650.0)


def test_compute_reclamos_score_imputa_mediana_faltantes() -> None:
    reclamos_df = pd.DataFrame(
        {
            "EPS": ["SURA", "SANITAS"],
            "TASA X CADA 10.000 AFILIADOS": [100.0, 300.0],
            "RECLAMOS": [50, 80],
        }
    )
    eps_obj = ["EPS SURA", "EPS SANITAS", "SALUD TOTAL EPS"]

    out = compute_reclamos_score(reclamos_df=reclamos_df, eps_obj=eps_obj)
    row_missing = out[out["EPS"] == "SALUD TOTAL EPS"].iloc[0]

    assert bool(row_missing["reclamos_imputado"]) is True
    assert row_missing["motivo_imputacion_reclamos"] == "sin_tasa_reclamos"
    assert np.isclose(float(row_missing["Tasa_Reclamos_10k"]), 200.0)


def test_build_composite_ranking_aplica_formula_60_20_20() -> None:
    scored_df = pd.DataFrame(
        {
            "EPS": ["EPS A", "EPS B", "EPS C"],
            "Escenario": ["BASE", "BASE", "BASE"],
            "Score_Riesgo": [80.0, 60.0, 40.0],
            "prob_imputada": [False, False, False],
        }
    )
    market_share_df = pd.DataFrame(
        {
            "EPS": ["EPS A", "EPS B", "EPS C"],
            "Afiliados_Valle": [1000.0, 800.0, 500.0],
            "MarketShare_Valle": [0.50, 0.30, 0.20],
        }
    )
    reclamos_score_df = pd.DataFrame(
        {
            "EPS": ["EPS A", "EPS B", "EPS C"],
            "Tasa_Reclamos_10k": [400.0, 300.0, 200.0],
            "Reclamos": [150.0, 120.0, 90.0],
            "Score_Reclamos": [0.0, 50.0, 100.0],
            "reclamos_imputado": [False, False, False],
            "motivo_imputacion_reclamos": ["", "", ""],
        }
    )

    out, ranking_escenario, ranking_global = build_composite_ranking(
        scored_df=scored_df,
        market_share_df=market_share_df,
        reclamos_score_df=reclamos_score_df,
        risk_weight=0.6,
        market_weight=0.2,
        complaints_weight=0.2,
    )

    eps_a = out[out["EPS"] == "EPS A"].iloc[0]
    eps_b = out[out["EPS"] == "EPS B"].iloc[0]
    eps_c = out[out["EPS"] == "EPS C"].iloc[0]

    expected_a = 0.6 * eps_a["Score_Riesgo"] + 0.2 * eps_a["Score_Mercado"] + 0.2 * eps_a["Score_Reclamos"]
    expected_b = 0.6 * eps_b["Score_Riesgo"] + 0.2 * eps_b["Score_Mercado"] + 0.2 * eps_b["Score_Reclamos"]
    expected_c = 0.6 * eps_c["Score_Riesgo"] + 0.2 * eps_c["Score_Mercado"] + 0.2 * eps_c["Score_Reclamos"]

    assert np.isclose(eps_a["Score_Final"], expected_a)
    assert np.isclose(eps_b["Score_Final"], expected_b)
    assert np.isclose(eps_c["Score_Final"], expected_c)

    rank_map = dict(zip(ranking_escenario["EPS"], ranking_escenario["Ranking_Escenario_Final"], strict=False))
    assert rank_map["EPS A"] < rank_map["EPS B"] < rank_map["EPS C"]

    assert "Score_Reclamos" in ranking_global.columns
    assert "Tasa_Reclamos_10k" in ranking_global.columns
    assert "Ranking_Global_Final" in ranking_global.columns

