from __future__ import annotations

import re
import inspect
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dashboards.ui import (
    append_total_row,
    apply_theme,
    chart_container,
    divider,
    explain_box,
    page_header,
    section_header,
    style_chart,
)
from src.models.eps_scoring import RATIO_SPECS, build_blocks_long, compute_ratios

try:
    from src.models import eps_montecarlo as eps_montecarlo_model
except Exception as exc:  # pragma: no cover - safety guard for deployment mismatches
    st.error(f"No se pudo cargar el modulo de modelo EPS Monte Carlo. Detalle: {exc}")
    st.stop()

EPS_OBJ_DEFAULT = list(getattr(eps_montecarlo_model, "EPS_OBJ_DEFAULT", []))
PROBABILITY_COLUMNS = list(getattr(eps_montecarlo_model, "PROBABILITY_COLUMNS", []))

run_eps_montecarlo = eps_montecarlo_model.run_eps_montecarlo
compute_market_share_valle = eps_montecarlo_model.compute_market_share_valle
compute_reclamos_score = eps_montecarlo_model.compute_reclamos_score
impute_missing_probabilities = eps_montecarlo_model.impute_missing_probabilities
score_risk_percentiles = eps_montecarlo_model.score_risk_percentiles
build_composite_ranking = eps_montecarlo_model.build_composite_ranking
build_income_statement_view = eps_montecarlo_model.build_income_statement_view


def _fallback_compute_cxp_revenue_score(
    eps_eeff_df: pd.DataFrame,
    eps_obj: list[str] | None = None,
) -> pd.DataFrame:
    universe = list(eps_obj or EPS_OBJ_DEFAULT)
    out = pd.DataFrame({"EPS": universe})
    out["CxP_Comercial"] = np.nan
    out["Ingresos"] = np.nan
    out["CxP_over_REV"] = np.nan
    out["Score_CxP_REV"] = 0.0
    out["cxp_rev_imputado"] = True
    out["motivo_imputacion_cxp_rev"] = "funcion_no_disponible_en_modelo"
    return out


def _fallback_build_eps_historical_compliance(
    base_eps: pd.DataFrame,
    eps_name: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Año",
            "CM_ratio",
            "PA_ratio",
            "RI_ratio",
            "Cumple_CM",
            "Cumple_PA",
            "Cumple_RI",
            "Cumple_3_de_3",
        ]
    )


def _fallback_run_eps_montecarlo_backtesting(
    upc_df: pd.DataFrame,
    eps_eeff_df: pd.DataFrame,
    eps_edad_df: pd.DataFrame,
    eps_afiliados_hist_df: pd.DataFrame,
    eps_obj: list[str] | None = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    _ = (upc_df, eps_eeff_df, eps_edad_df, eps_afiliados_hist_df, eps_obj, kwargs)
    detail = pd.DataFrame(
        columns=[
            "EPS",
            "AnchorYear",
            "TargetYear",
            "Metric",
            "PredictedProb",
            "ObservedBreach",
            "AbsError",
            "Brier",
        ]
    )
    summary = pd.DataFrame(
        columns=["Metric", "N", "PredictedMean", "ObservedRate", "MAE", "Brier"]
    )
    diagnostics = {"reason": "funcion_backtesting_no_disponible_en_modelo"}
    return detail, summary, diagnostics


def _fallback_run_eps_seed_stability(
    upc_df: pd.DataFrame,
    eps_eeff_df: pd.DataFrame,
    eps_edad_df: pd.DataFrame,
    eps_afiliados_hist_df: pd.DataFrame,
    market_share_df: pd.DataFrame,
    **kwargs: Any,
) -> Dict[str, pd.DataFrame]:
    _ = (upc_df, eps_eeff_df, eps_edad_df, eps_afiliados_hist_df, market_share_df, kwargs)
    return {
        "per_seed_global": pd.DataFrame(),
        "per_seed_escenario": pd.DataFrame(),
        "pairwise_global": pd.DataFrame(),
        "pairwise_escenario": pd.DataFrame(),
        "summary_global": pd.DataFrame(),
        "summary_escenario": pd.DataFrame(),
    }


compute_cxp_revenue_score = getattr(
    eps_montecarlo_model,
    "compute_cxp_revenue_score",
    _fallback_compute_cxp_revenue_score,
)
build_eps_historical_compliance = getattr(
    eps_montecarlo_model,
    "build_eps_historical_compliance",
    _fallback_build_eps_historical_compliance,
)
run_eps_montecarlo_backtesting = getattr(
    eps_montecarlo_model,
    "run_eps_montecarlo_backtesting",
    _fallback_run_eps_montecarlo_backtesting,
)
run_eps_seed_stability = getattr(
    eps_montecarlo_model,
    "run_eps_seed_stability",
    _fallback_run_eps_seed_stability,
)

st.set_page_config(page_title="Clientes", layout="wide")
apply_theme()

page_header(
    "Clientes",
    "Ranking EPS por riesgo Monte Carlo, mercado Valle y tasa de reclamos.",
    "EPS Monte Carlo",
)


def normalize_account(text: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return " ".join(normalized.lower().split())


def get_eps_universe() -> list[str]:
    return [eps for eps in EPS_OBJ_DEFAULT if "excepcion" not in normalize_account(eps)]


def normalize_sheet_name(text: object) -> str:
    return re.sub(r"[^a-z0-9]", "", normalize_account(text))


def parse_seed_values(text: str) -> List[int]:
    raw_parts = re.split(r"[,\s;]+", str(text).strip())
    seeds: List[int] = []
    for part in raw_parts:
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            continue
        seeds.append(value)
    unique = sorted(set(seeds))
    return unique


def parse_age_group(text: object) -> str | None:
    if not isinstance(text, str):
        return None
    normalized = normalize_account(text)
    if "80" in normalized and ("mas" in normalized or "+" in normalized):
        return "80+"
    nums = re.findall(r"\d+", normalized)
    if len(nums) >= 2:
        return f"{int(nums[0])}-{int(nums[1])}"
    return None


def find_col(columns: List[str], includes: List[str]) -> str | None:
    for col in columns:
        norm = normalize_account(col)
        if any(token in norm for token in includes):
            return col
    return None


def validate_reclamos_sheet(df: pd.DataFrame) -> str | None:
    if df.empty:
        return "La hoja RECLAMOS esta vacia."

    cols = [str(c) for c in df.columns]
    eps_col = find_col(cols, ["eps"])
    tasa_col = find_col(
        cols,
        [
            "tasa x cada 10.000 afiliados",
            "tasa x cada 10000 afiliados",
            "tasa",
        ],
    )
    if eps_col is None:
        return "La hoja RECLAMOS no tiene columna EPS."
    if tasa_col is None:
        return "La hoja RECLAMOS no tiene columna de TASA X CADA 10.000 AFILIADOS."
    return None


def risk_bucket(score: float | None) -> str:
    if score is None or pd.isna(score):
        return "Sin dato"
    if score >= 70:
        return "Bajo riesgo"
    if score >= 50:
        return "Riesgo medio"
    return "Alto riesgo"


def percentile_score(values: pd.Series, low_is_better: bool) -> pd.Series:
    data = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=values.index, dtype=float)
    valid = data.dropna()
    n = len(valid)
    if n == 0:
        return out
    if n == 1:
        out.loc[valid.index] = 100.0
        return out
    rank = valid.rank(method="average", ascending=True) - 1.0
    if low_is_better:
        score = (1.0 - (rank / (n - 1.0))) * 100.0
    else:
        score = (rank / (n - 1.0)) * 100.0
    out.loc[valid.index] = score
    return out


EPS_RATIO_LABELS: Dict[str, str] = {
    "current_ratio": "Razon corriente",
    "cash_ratio": "Razon de caja",
    "wc_to_rev": "Capital de trabajo neto / REV",
    "days_cash": "Dias de caja (proxy)",
    "current_assets_ratio": "Activos corrientes / Activos totales",
    "debt_to_assets": "Deuda total / Activos",
    "equity_ratio": "Patrimonio / Activos",
    "assets_to_liabilities": "Activos / Pasivos totales",
    "current_liab_share": "Peso del corto plazo",
    "net_debt_to_ebitda": "Deuda neta CP / EBITDA",
    "gross_margin": "Margen bruto",
    "ebitda_margin": "Margen EBITDA",
    "ebit_margin": "Margen operativo (EBIT)",
    "net_margin": "Margen neto",
    "roa": "ROA",
    "asset_turnover": "Rotacion de activos",
    "dso": "DSO (dias de cartera)",
    "dpo": "DPO (dias de proveedores)",
    "opex_cash_ratio": "Indice OPEX en efectivo",
    "da_intensity": "Intensidad dep/amort",
}

EPS_RATIO_KINDS: Dict[str, str] = {
    "current_ratio": "x",
    "cash_ratio": "x",
    "wc_to_rev": "percent",
    "days_cash": "days",
    "current_assets_ratio": "percent",
    "debt_to_assets": "percent",
    "equity_ratio": "percent",
    "assets_to_liabilities": "x",
    "current_liab_share": "percent",
    "net_debt_to_ebitda": "x",
    "gross_margin": "percent",
    "ebitda_margin": "percent",
    "ebit_margin": "percent",
    "net_margin": "percent",
    "roa": "percent",
    "asset_turnover": "x",
    "dso": "days",
    "dpo": "days",
    "opex_cash_ratio": "percent",
    "da_intensity": "percent",
}

EPS_RATIO_GROUPS: Dict[str, List[str]] = {
    "Liquidez": ["current_ratio", "cash_ratio", "wc_to_rev", "days_cash", "current_assets_ratio"],
    "Solvencia": ["debt_to_assets", "equity_ratio", "assets_to_liabilities", "current_liab_share", "net_debt_to_ebitda"],
    "Rentabilidad": ["gross_margin", "ebitda_margin", "ebit_margin", "net_margin", "roa"],
    "Eficiencia": ["asset_turnover", "dso", "dpo", "opex_cash_ratio", "da_intensity"],
}


def format_ratio_value(value: float | None, kind: str) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if kind == "percent":
        return f"{value * 100:.2f}%"
    if kind == "days":
        return f"{value:,.1f} dias"
    if kind == "x":
        return f"{value:.2f}x"
    return f"{value:,.2f}"


def normalize_eps_key(text: object) -> str:
    normalized = normalize_account(text)
    tokens = [tok for tok in normalized.split() if tok != "eps"]
    return " ".join(tokens).strip()


def canonicalize_eps_name(text: object, eps_universe: List[str]) -> str | None:
    raw_norm = normalize_account(text)
    raw_key = normalize_eps_key(text)
    exact_map = {normalize_account(eps): eps for eps in eps_universe}
    key_map = {normalize_eps_key(eps): eps for eps in eps_universe}

    if raw_norm in exact_map:
        return exact_map[raw_norm]
    if raw_key in key_map:
        return key_map[raw_key]

    if raw_key:
        for key, eps in key_map.items():
            if raw_key and (raw_key in key or key in raw_key):
                return eps
    return None


def build_eps_ratio_dataset(
    eps_eeff_df: pd.DataFrame,
    eps_universe: List[str],
) -> tuple[pd.DataFrame, List[int]]:
    if eps_eeff_df.empty:
        return pd.DataFrame(), []

    cols = [str(c) for c in eps_eeff_df.columns]
    eps_col = find_col(cols, ["eps"])
    cuenta_col = find_col(cols, ["cuenta"])
    year_cols = sorted(
        [c for c in eps_eeff_df.columns if str(c).strip().isdigit()],
        key=lambda c: int(str(c).strip()),
    )

    if eps_col is None or cuenta_col is None or not year_cols:
        return pd.DataFrame(), []

    work = eps_eeff_df[[eps_col, cuenta_col] + year_cols].copy()
    work["EPS_clean"] = work[eps_col].map(lambda x: canonicalize_eps_name(x, eps_universe))
    work = work[work["EPS_clean"].notna()].copy()
    if work.empty:
        return pd.DataFrame(), []

    work["CUENTA"] = work[cuenta_col].astype(str)
    blocks_df = build_blocks_long(
        df=work,
        year_cols=year_cols,
        entity_col="EPS_clean",
        account_col="CUENTA",
    )
    ratios_df = compute_ratios(blocks_df)
    if ratios_df.empty:
        return pd.DataFrame(), []

    ratios_df["entity"] = ratios_df["entity"].astype(str).str.strip()
    ratios_df = ratios_df[ratios_df["entity"].isin(eps_universe)].copy()
    ratios_df["year"] = pd.to_numeric(ratios_df["year"], errors="coerce")
    ratios_df = ratios_df[ratios_df["year"].notna()].copy()
    ratios_df["year"] = ratios_df["year"].astype(int)
    ratios_df = ratios_df.sort_values(["entity", "year"]).reset_index(drop=True)
    years = sorted(ratios_df["year"].unique().tolist())
    return ratios_df, years


def build_eps_ratio_table(
    ratios_df: pd.DataFrame,
    selected_eps: str,
    ratio_list: List[str],
    years: List[int],
) -> pd.DataFrame:
    if ratios_df.empty:
        return pd.DataFrame(columns=["Indicador"] + [str(y) for y in years] + ["Promedio"])

    subset = ratios_df[ratios_df["entity"] == selected_eps]
    rows: List[Dict[str, Any]] = []
    for ratio in ratio_list:
        if ratio not in ratios_df.columns:
            continue
        kind = EPS_RATIO_KINDS.get(ratio, "ratio")
        row: Dict[str, Any] = {"Indicador": EPS_RATIO_LABELS.get(ratio, ratio)}
        values: List[float] = []
        for year in years:
            year_values = pd.to_numeric(
                subset.loc[subset["year"] == int(year), ratio],
                errors="coerce",
            ).dropna()
            value = float(year_values.mean()) if not year_values.empty else np.nan
            row[str(year)] = format_ratio_value(value, kind)
            if pd.notna(value):
                values.append(value)
        mean_value = float(np.mean(values)) if values else np.nan
        row["Promedio"] = format_ratio_value(mean_value, kind)
        rows.append(row)
    return pd.DataFrame(rows)


def build_eps_indicator_comparison(
    ratios_df: pd.DataFrame,
    indicator: str,
    period: str,
) -> pd.DataFrame:
    if ratios_df.empty or indicator not in ratios_df.columns:
        return pd.DataFrame(columns=["EPS", "Valor", "Periodo"])

    base = ratios_df[["entity", "year", indicator]].copy()
    base[indicator] = pd.to_numeric(base[indicator], errors="coerce")
    base = base.dropna(subset=[indicator])
    if base.empty:
        return pd.DataFrame(columns=["EPS", "Valor", "Periodo"])

    if period == "Promedio":
        comp = (
            base.groupby("entity", as_index=False)[indicator]
            .mean()
            .rename(columns={"entity": "EPS", indicator: "Valor"})
        )
        comp["Periodo"] = "Promedio"
        return comp

    year = int(period)
    comp = (
        base[base["year"] == year]
        .groupby("entity", as_index=False)[indicator]
        .mean()
        .rename(columns={"entity": "EPS", indicator: "Valor"})
    )
    comp["Periodo"] = str(year)
    return comp


def build_composite_ranking_local(
    base_df: pd.DataFrame,
    market_share_df: pd.DataFrame,
    reclamos_score_df: pd.DataFrame,
    cxp_score_df: pd.DataFrame,
    risk_weight: float,
    market_weight: float,
    complaints_weight: float,
    cxp_rev_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = base_df.copy()

    drop_cols = [
        "MarketShare_Valle",
        "Afiliados_Valle",
        "Score_Mercado",
        "Tasa_Reclamos_10k",
        "Reclamos",
        "Score_Reclamos",
        "reclamos_imputado",
        "motivo_imputacion_reclamos",
        "CxP_Comercial",
        "Ingresos",
        "CxP_over_REV",
        "Score_CxP_REV",
        "cxp_rev_imputado",
        "motivo_imputacion_cxp_rev",
        "Score_Final",
        "Ranking_Escenario_Final",
    ]
    existing_drop = [c for c in drop_cols if c in out.columns]
    if existing_drop:
        out = out.drop(columns=existing_drop)

    market_cols = ["EPS", "MarketShare_Valle", "Afiliados_Valle"]
    market_base = market_share_df[[c for c in market_cols if c in market_share_df.columns]].copy()
    market_base = market_base.drop_duplicates(subset=["EPS"], keep="last")
    if "MarketShare_Valle" not in market_base.columns:
        market_base["MarketShare_Valle"] = 0.0
    if "Afiliados_Valle" not in market_base.columns:
        market_base["Afiliados_Valle"] = 0.0
    market_base["Score_Mercado"] = percentile_score(
        market_base["MarketShare_Valle"], low_is_better=False
    )

    reclamos_cols = [
        "EPS",
        "Tasa_Reclamos_10k",
        "Reclamos",
        "Score_Reclamos",
        "reclamos_imputado",
        "motivo_imputacion_reclamos",
    ]
    reclamos_base = reclamos_score_df[
        [c for c in reclamos_cols if c in reclamos_score_df.columns]
    ].drop_duplicates(subset=["EPS"], keep="last")
    for col in reclamos_cols:
        if col not in reclamos_base.columns:
            if col in {"Score_Reclamos", "Tasa_Reclamos_10k", "Reclamos"}:
                reclamos_base[col] = np.nan
            elif col == "reclamos_imputado":
                reclamos_base[col] = True
            else:
                reclamos_base[col] = ""

    cxp_cols = [
        "EPS",
        "CxP_Comercial",
        "Ingresos",
        "CxP_over_REV",
        "Score_CxP_REV",
        "cxp_rev_imputado",
        "motivo_imputacion_cxp_rev",
    ]
    cxp_base = cxp_score_df[
        [c for c in cxp_cols if c in cxp_score_df.columns]
    ].drop_duplicates(subset=["EPS"], keep="last")
    for col in cxp_cols:
        if col not in cxp_base.columns:
            if col in {"CxP_Comercial", "Ingresos", "CxP_over_REV", "Score_CxP_REV"}:
                cxp_base[col] = np.nan
            elif col == "cxp_rev_imputado":
                cxp_base[col] = True
            else:
                cxp_base[col] = ""

    out = out.merge(market_base, on="EPS", how="left")
    out = out.merge(reclamos_base, on="EPS", how="left")
    out = out.merge(cxp_base, on="EPS", how="left")

    if "Score_Riesgo" not in out.columns:
        out["Score_Riesgo"] = np.nan
    if "prob_imputada" not in out.columns:
        out["prob_imputada"] = False
    if "motivo_imputacion" not in out.columns:
        out["motivo_imputacion"] = ""

    out["Score_Riesgo"] = pd.to_numeric(out["Score_Riesgo"], errors="coerce").fillna(0.0)
    out["Score_Mercado"] = pd.to_numeric(out["Score_Mercado"], errors="coerce").fillna(0.0)
    out["Score_Reclamos"] = pd.to_numeric(out["Score_Reclamos"], errors="coerce").fillna(0.0)
    out["Score_CxP_REV"] = pd.to_numeric(out["Score_CxP_REV"], errors="coerce").fillna(0.0)
    out["MarketShare_Valle"] = pd.to_numeric(out["MarketShare_Valle"], errors="coerce").fillna(0.0)
    out["Afiliados_Valle"] = pd.to_numeric(out["Afiliados_Valle"], errors="coerce").fillna(0.0)
    out["reclamos_imputado"] = out["reclamos_imputado"].fillna(True)
    out["cxp_rev_imputado"] = out["cxp_rev_imputado"].fillna(True)
    out["motivo_imputacion_reclamos"] = out["motivo_imputacion_reclamos"].fillna("")
    out["motivo_imputacion_cxp_rev"] = out["motivo_imputacion_cxp_rev"].fillna("")

    out["Score_Final"] = (
        risk_weight * out["Score_Riesgo"]
        + market_weight * out["Score_Mercado"]
        + complaints_weight * out["Score_Reclamos"]
        + cxp_rev_weight * out["Score_CxP_REV"]
    )
    out["Ranking_Escenario_Final"] = (
        out.groupby("Escenario")["Score_Final"].rank(method="dense", ascending=False).astype(int)
    )
    ranking_escenario = out.sort_values(
        ["Escenario", "Ranking_Escenario_Final", "EPS"]
    ).reset_index(drop=True)

    ranking_global = (
        out.groupby("EPS", as_index=False)
        .agg(
            Score_Final=("Score_Final", "mean"),
            Score_Riesgo=("Score_Riesgo", "mean"),
            Score_Mercado=("Score_Mercado", "mean"),
            Score_Reclamos=("Score_Reclamos", "mean"),
            Score_CxP_REV=("Score_CxP_REV", "mean"),
            MarketShare_Valle=("MarketShare_Valle", "mean"),
            Afiliados_Valle=("Afiliados_Valle", "mean"),
            Tasa_Reclamos_10k=("Tasa_Reclamos_10k", "mean"),
            Reclamos=("Reclamos", "mean"),
            CxP_over_REV=("CxP_over_REV", "mean"),
            CxP_Comercial=("CxP_Comercial", "mean"),
            prob_imputada=("prob_imputada", "max"),
            reclamos_imputado=("reclamos_imputado", "max"),
            cxp_rev_imputado=("cxp_rev_imputado", "max"),
        )
        .sort_values("Score_Final", ascending=False)
        .reset_index(drop=True)
    )
    ranking_global["Ranking_Global_Final"] = (
        ranking_global["Score_Final"].rank(method="dense", ascending=False).astype(int)
    )
    ranking_global = ranking_global.sort_values(
        ["Ranking_Global_Final", "EPS"]
    ).reset_index(drop=True)
    return out, ranking_escenario, ranking_global


def render_score_methodology() -> None:
    def formula_gap(px: int = 16) -> None:
        st.markdown(f"<div style='height:{px}px'></div>", unsafe_allow_html=True)

    section_header("Metodologia del Score EPS", "Modelo unificado con riesgo, mercado y reclamos")
    st.markdown("**1) Riesgo Monte Carlo (55%)**")
    st.latex(
        r"P_{avg,m}=\frac{1}{N_{sim}}\sum_{s=1}^{N_{sim}}\left(\frac{1}{T}\sum_{t=1}^{T}\mathbf{1}[incumple_{m,s,t}]\right)"
    )
    formula_gap(20)
    st.latex(r"Score_{Riesgo}=\frac{1}{M}\sum_{m=1}^{M}Percentil_{inv}(P_{avg,m})")

    divider()
    st.markdown("**2) Mercado Valle (5%)**")
    st.latex(r"MarketShare_{EPS}=\frac{Afiliados_{EPS,Valle}}{\sum_j Afiliados_{j,Valle}}")
    formula_gap(20)
    st.latex(r"Score_{Mercado}=Percentil_{dir}(MarketShare_{EPS})")

    divider()
    st.markdown("**3) Reclamos x 10.000 (20%)**")
    st.latex(
        r"Tasa_{10k,EPS}=\text{valor de la hoja RECLAMOS en columna }TASA\ X\ CADA\ 10.000\ AFILIADOS"
    )
    formula_gap(20)
    st.latex(r"Score_{Reclamos}=Percentil_{inv}(Tasa_{10k,EPS})")

    divider()
    st.markdown("**4) CxP comerciales / Ingresos (20%)**")
    st.latex(
        r"CxP\_over\_REV_{EPS}=\frac{CuentasPorPagarComerciales_{EPS}}{IngresosTotales_{EPS}}"
    )
    formula_gap(20)
    st.latex(r"Score_{CxP/REV}=Percentil_{inv}(CxP\_over\_REV_{EPS})")

    divider()
    st.markdown("**Score final ponderado**")
    st.latex(
        r"Score_{Final}=0.55\cdot Score_{Riesgo}+0.05\cdot Score_{Mercado}+0.20\cdot Score_{Reclamos}+0.20\cdot Score_{CxP/REV}"
    )

    formulas_df = pd.DataFrame(
        [
            {"Variable": "P_avg_*", "Definicion": "Probabilidad promedio de incumplimiento por metrica en el horizonte."},
            {"Variable": "Score_Riesgo", "Definicion": "Promedio de percentiles invertidos de P_avg_* (menor probabilidad = mejor)."},
            {"Variable": "Score_Mercado", "Definicion": "Percentil directo de participacion de afiliados en Valle."},
            {"Variable": "Tasa_Reclamos_10k", "Definicion": "Tasa de reclamos por cada 10.000 afiliados (hoja RECLAMOS)."},
            {"Variable": "Score_Reclamos", "Definicion": "Percentil invertido de la tasa de reclamos (menor tasa = mejor)."},
            {"Variable": "CxP_over_REV", "Definicion": "Cuentas por pagar comerciales sobre ingresos totales (menor = mejor)."},
            {"Variable": "Score_CxP_REV", "Definicion": "Percentil invertido de CxP_over_REV."},
            {"Variable": "Score_Final", "Definicion": "Combinacion ponderada 55/5/20/20 de riesgo, mercado, reclamos y CxP/REV."},
        ]
    )
    st.dataframe(formulas_df, width="stretch", hide_index=True)

    divider()
    section_header("Detalle tecnico Monte Carlo", "Supuestos, variables estocasticas y ecuaciones del modelo")
    st.markdown(
        (
            "El modelo simula trayectorias anuales por EPS desde el ano posterior al ancla financiera "
            "hasta el horizonte definido. En cada trayectoria se generan drivers aleatorios, se "
            "actualizan ingresos y estructura financiera, y luego se evalua incumplimiento en cada metrica."
        )
    )

    st.markdown("**1) Variables estocasticas por ano y simulacion**")
    st.markdown(
        (
            "Se simulan 5 drivers financieros: crecimiento de afiliados `g`, loss ratio `LR`, "
            "admin ratio `AR`, otros gastos operativos `OER` y ratio de reservas `rho_res`."
        )
    )
    st.latex(r"X_{k,s,t}=\mathrm{clip}\left(\mathcal{N}\left(\mu_k+\Delta_k,\sigma_k\right),L_k,U_k\right)")
    st.markdown("Donde `k in {g, LR, AR, OER, rho_res}`.")
    st.markdown(
        "Para la metrica de dias de pago se simula adicionalmente `CxP_ratio` (cuentas por pagar/ingresos)."
    )
    st.latex(r"\Delta_g=g_{shift}^{escenario},\qquad \Delta_{LR}=LR_{shift}^{escenario}")
    st.markdown(
        (
            "Los parametros `mu` y `sigma` se estiman por EPS con historico 2019..ano ancla. "
            "Los escenarios estresan `g` y `LR`; los demas drivers mantienen su distribucion base."
        )
    )

    divider()
    st.markdown("**2) Como se usa la UPC (proyeccion por segmento y mezcla EPS)**")
    st.markdown(
        (
            "Primero se proyecta UPC por segmento (`Regimen x GrupoEdad_UPC`) con crecimiento anual "
            "ajustado por optimism factor."
        )
    )
    st.latex(r"g^{base}_{seg}=\left(\frac{UPC_{seg,y_1}}{UPC_{seg,y_0}}\right)^{\frac{1}{y_1-y_0}}-1")
    st.latex(r"g^{adj}_{seg}=g^{base}_{seg}\cdot f_{optimism},\qquad UPC_{seg,t}=UPC_{seg,t-1}\cdot (1+g^{adj}_{seg})")
    st.markdown(
        (
            "Luego se calcula `UPC_mix` por EPS usando pesos etarios/regimen fijos de `EPS_Edad` "
            "(mix observado, sin cambio de composicion en el horizonte)."
        )
    )
    st.latex(r"UPC\_mix_{e,t}=\sum_{r,a} w_{e,r,a}\cdot UPC_{r,a,t},\qquad \sum_{r,a} w_{e,r,a}=1")

    divider()
    st.markdown("**3) Dinamica anual del estado financiero simulado**")
    st.latex(r"Afiliados_{s,t}=Afiliados_{s,t-1}\cdot(1+g_{s,t})")
    st.latex(r"Ingresos_{s,t}=Afiliados_{s,t}\cdot UPC\_mix_{e,t}\cdot k_e")
    st.latex(r"OPEX\_cash_{s,t}=Ingresos_{s,t}\cdot\left(LR_{s,t}+AR_{s,t}+OER_{s,t}\right)")
    st.latex(r"Margen_{s,t}=Ingresos_{s,t}-OPEX\_cash_{s,t}")
    st.latex(r"Cash_{s,t}=\max\left(0,\;Cash_{s,t-1}+\alpha\cdot Margen_{s,t}\right),\ \alpha=1")
    st.latex(r"Equity_{s,t}=Equity_{s,t-1}+\beta\cdot Margen_{s,t},\ \beta=1")
    st.latex(r"Reservas_{s,t}=Ingresos_{s,t}\cdot \rho\_{res,s,t}")
    st.latex(r"Inversiones_{s,t}=Ingresos_{s,t}\cdot ratio\_{inv,0}")

    st.markdown("**4) Ratios de cumplimiento evaluados cada ano**")
    st.latex(r"CashDays_{s,t}=365\cdot\frac{Cash_{s,t}}{OPEX\_cash_{s,t}}")
    st.latex(r"CxP\_ratio_{s,t}\sim \mathrm{clip}\left(\mathcal{N}\left(\mu_{CxP/REV},\sigma_{CxP/REV}\right),0,3\right)")
    st.latex(r"CxP_{s,t}=Ingresos_{s,t}\cdot CxP\_ratio_{s,t}")
    st.latex(r"PayDays_{s,t}=365\cdot\frac{CxP_{s,t}}{OPEX\_cash_{s,t}}")
    st.latex(r"incumple_{PayDays,s,t}=\mathbf{1}\left[PayDays_{s,t}<20\ \vee\ PayDays_{s,t}>70\right]")
    st.latex(r"CM\_ratio_{s,t}=\frac{Equity_{s,t}}{CapMinReq_e}")
    st.latex(r"PA\_ratio_{s,t}=\frac{Equity_{s,t}}{0.08\cdot Ingresos_{s,t}\cdot LR_{s,t}}")
    st.latex(r"RI\_ratio_{s,t}=\frac{Cash_{s,t}+Inversiones_{s,t}}{Reservas_{s,t}}")

    divider()
    st.markdown("**5) Probabilidad promedio de incumplimiento (enfoque PROMEDIO)**")
    st.latex(
        r"P_{avg,m}=\frac{1}{N_{sim}}\sum_{s=1}^{N_{sim}}\left(\frac{1}{T}\sum_{t=1}^{T}\mathbf{1}\left[incumple_{m,s,t}\right]\right)"
    )
    st.markdown(
        (
            "Cada metrica `m` produce una probabilidad promedio `P_avg`. Luego estas probabilidades "
            "se convierten a score por percentiles invertidos (menor probabilidad = mayor score)."
        )
    )

    divider()
    st.markdown("**6) Robustez del modelo: backtesting y estabilidad por semilla**")
    st.latex(r"Brier_m=\frac{1}{n}\sum_{i=1}^{n}\left(\hat p_{i,m}-y_{i,m}\right)^2")
    st.latex(r"MAE_m=\frac{1}{n}\sum_{i=1}^{n}\left|\hat p_{i,m}-y_{i,m}\right|")
    st.latex(r"\rho_{Spearman}=corr\left(rank(Ranking^{seed_a}),rank(Ranking^{seed_b})\right)")
    st.markdown(
        (
            "El backtesting se hace 1-step ahead: se ancla en anos historicos y se compara la probabilidad "
            "predicha contra el incumplimiento observado en el ano siguiente. "
            "La estabilidad por semilla compara rankings entre corridas con semillas distintas usando Spearman."
        )
    )


def resolve_cali_excel_path() -> Path | None:
    path = ROOT_DIR / "data" / "raw" / "Cali ANALISIS.xlsx"
    if path.exists():
        return path
    alt = ROOT_DIR / "Cali ANALISIS.xlsx"
    if alt.exists():
        return alt
    return None


def file_signature(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_mtime_ns}-{stat.st_size}"


@st.cache_data(show_spinner=False)
def _cached_sheet_names(path_str: str, sig: str) -> List[str]:
    _ = sig
    return pd.ExcelFile(path_str).sheet_names


@st.cache_data(show_spinner=False)
def _cached_read_sheet(path_str: str, sheet_name: str, sig: str) -> pd.DataFrame:
    _ = sig
    df = pd.read_excel(path_str, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_cali_sheet(aliases: List[str]) -> tuple[pd.DataFrame, str]:
    path = resolve_cali_excel_path()
    if path is None:
        return pd.DataFrame(), "Archivo no encontrado: Cali ANALISIS.xlsx"

    try:
        sig = file_signature(path)
        sheet_names = _cached_sheet_names(str(path), sig)
    except Exception as exc:
        return pd.DataFrame(), f"No se pudo abrir Cali ANALISIS.xlsx: {exc}"

    normalized_sheet_map = {normalize_sheet_name(s): s for s in sheet_names}
    selected_sheet = None
    for alias in aliases:
        key = normalize_sheet_name(alias)
        if key in normalized_sheet_map:
            selected_sheet = normalized_sheet_map[key]
            break
    if selected_sheet is None:
        return pd.DataFrame(), f"No se encontro hoja para aliases {aliases}. Disponibles: {sheet_names}"

    try:
        df = _cached_read_sheet(str(path), selected_sheet, sig)
        return df, f"Excel: {path.name} (hoja {selected_sheet})"
    except Exception as exc:
        return pd.DataFrame(), f"Error leyendo hoja {selected_sheet}: {exc}"


def load_cali_sheet_by_column_tokens(
    required_column_tokens: List[List[str]],
) -> tuple[pd.DataFrame, str]:
    path = resolve_cali_excel_path()
    if path is None:
        return pd.DataFrame(), "Archivo no encontrado: Cali ANALISIS.xlsx"

    try:
        sig = file_signature(path)
        sheet_names = _cached_sheet_names(str(path), sig)
    except Exception as exc:
        return pd.DataFrame(), f"No se pudo abrir Cali ANALISIS.xlsx: {exc}"

    def has_required_columns(columns: List[str]) -> bool:
        normalized_cols = [normalize_account(c) for c in columns]
        for token_set in required_column_tokens:
            token_set_norm = [normalize_account(t) for t in token_set]
            ok = any(all(tok in col for tok in token_set_norm) for col in normalized_cols)
            if not ok:
                return False
        return True

    for sheet_name in sheet_names:
        try:
            preview = pd.read_excel(path, sheet_name=sheet_name, nrows=5)
        except Exception:
            continue
        preview_cols = [str(c).strip() for c in preview.columns]
        if has_required_columns(preview_cols):
            try:
                df = _cached_read_sheet(str(path), sheet_name, sig)
                return (
                    df,
                    (
                        f"Excel: {path.name} (hoja {sheet_name}, autodetectada por columnas: "
                        f"{required_column_tokens})"
                    ),
                )
            except Exception as exc:
                return pd.DataFrame(), f"Error leyendo hoja autodetectada {sheet_name}: {exc}"

    return (
        pd.DataFrame(),
        f"No se encontro hoja con columnas requeridas {required_column_tokens}. Disponibles: {sheet_names}",
    )


def affiliates_55_table(age_df: pd.DataFrame) -> pd.DataFrame | None:
    if age_df.empty:
        return None

    cols = [str(c) for c in age_df.columns]
    dept_col = find_col(cols, ["departamento"])
    eps_col = find_col(cols, ["eps"])
    age_col = find_col(cols, ["quinquenio", "edad"])
    fem_col = find_col(cols, ["femenino"])
    masc_col = find_col(cols, ["masculino"])
    total_col = find_col(cols, ["total afiliados", "total"])
    if not all([dept_col, eps_col, age_col, fem_col, masc_col, total_col]):
        return None

    work = age_df[[dept_col, eps_col, age_col, fem_col, masc_col, total_col]].copy()
    work[dept_col] = work[dept_col].astype(str)
    work = work[work[dept_col].str.contains("valle del cauca", case=False, na=False)]
    if work.empty:
        return None

    work["EPS"] = work[eps_col].astype(str).str.strip().str.upper()
    work["age_group"] = work[age_col].map(parse_age_group)
    work = work.dropna(subset=["age_group"])

    def is_55_plus(group: str) -> bool:
        if group == "80+":
            return True
        if "-" in group:
            try:
                return int(group.split("-")[0]) >= 55
            except ValueError:
                return False
        return False

    work = work[work["age_group"].map(is_55_plus)]
    if work.empty:
        return None

    for col in [fem_col, masc_col, total_col]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    out = (
        work.groupby("EPS")[[fem_col, masc_col, total_col]]
        .sum(min_count=1)
        .reset_index()
        .rename(
            columns={
                fem_col: "Femenino",
                masc_col: "Masculino",
                total_col: "Total afiliados",
            }
        )
        .sort_values("Total afiliados", ascending=False)
        .reset_index(drop=True)
    )
    out = append_total_row(out, "EPS", ["Femenino", "Masculino", "Total afiliados"])
    return out


def regimen_valle_table(mun_df: pd.DataFrame) -> pd.DataFrame | None:
    if mun_df.empty:
        return None

    cols = [str(c) for c in mun_df.columns]
    dep_col = find_col(cols, ["departamento"])
    eps_col = find_col(cols, ["eps"])
    contrib_col = find_col(cols, ["afiliados contributivo"])
    subs_col = find_col(cols, ["afiliados subsidiado"])
    esp_col = find_col(cols, ["afiliados especiales", "afiliados excepcion"])
    total_col = find_col(cols, ["total afiliados"])
    if not all([dep_col, eps_col, contrib_col, subs_col, total_col]):
        return None

    work = mun_df[[dep_col, eps_col, contrib_col, subs_col, total_col] + ([esp_col] if esp_col else [])].copy()
    work[dep_col] = work[dep_col].astype(str)
    work = work[work[dep_col].str.contains("valle del cauca", case=False, na=False)]
    if work.empty:
        return None

    work["EPS"] = work[eps_col].astype(str).str.strip().str.upper()
    for col in [contrib_col, subs_col, total_col] + ([esp_col] if esp_col else []):
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

    agg_cols = [contrib_col, subs_col, total_col] + ([esp_col] if esp_col else [])
    out = work.groupby("EPS", as_index=False)[agg_cols].sum()

    rename_map = {
        contrib_col: "Contributivo",
        subs_col: "Subsidiado",
        total_col: "Total afiliados",
    }
    if esp_col:
        rename_map[esp_col] = "Especiales/Excepcion"
    out = out.rename(columns=rename_map)

    for col in ["Contributivo", "Subsidiado", "Especiales/Excepcion"]:
        if col not in out.columns:
            out[col] = 0.0

    out = out.sort_values("Total afiliados", ascending=False).reset_index(drop=True)
    out = append_total_row(
        out,
        "EPS",
        ["Contributivo", "Subsidiado", "Especiales/Excepcion", "Total afiliados"],
    )
    return out


@st.cache_data(show_spinner=False)
def run_clientes_pipeline(
    eps_eeff_df: pd.DataFrame,
    upc_df: pd.DataFrame,
    eps_edad_df: pd.DataFrame,
    eps_anos_df: pd.DataFrame,
    eps_afiliados_df: pd.DataFrame,
    reclamos_df: pd.DataFrame,
    n_sim: int,
    horizon_end: int,
    upc_optimism_factor: float,
    upc_growth_start_year: int,
    upc_growth_end_year: int,
    base_lr_shift: float,
    base_g_shift: float,
    stress_lr_shift: float,
    stress_lr_g_shift: float,
    stress_mix_lr_shift: float,
    stress_mix_g_shift: float,
) -> tuple[
    Dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, Dict[str, float]],
]:
    eps_universe = get_eps_universe()
    scenarios = {
        "BASE": {"LR_shift": base_lr_shift, "g_shift": base_g_shift},
        "STRESS_LR": {"LR_shift": stress_lr_shift, "g_shift": stress_lr_g_shift},
        "STRESS_MIX": {"LR_shift": stress_mix_lr_shift, "g_shift": stress_mix_g_shift},
    }

    results_df, diagnostics = run_eps_montecarlo(
        upc_df=upc_df,
        eps_eeff_df=eps_eeff_df,
        eps_edad_df=eps_edad_df,
        eps_afiliados_hist_df=eps_anos_df,
        eps_obj=eps_universe,
        n_sim=n_sim,
        horizon_end=horizon_end,
        cash_thresholds=(15,),
        upc_optimism_factor=upc_optimism_factor,
        upc_growth_start_year=upc_growth_start_year,
        upc_growth_end_year=upc_growth_end_year,
        scenarios=scenarios,
        random_seed=42,
    )
    market_share_df = compute_market_share_valle(
        eps_afiliados_df=eps_afiliados_df,
        eps_obj=eps_universe,
    )
    results_imputed = impute_missing_probabilities(
        results_df=results_df,
        eps_obj=eps_universe,
        scenarios=list(scenarios.keys()),
        probability_columns=PROBABILITY_COLUMNS,
    )
    results_scored = score_risk_percentiles(
        df=results_imputed,
        probability_columns=PROBABILITY_COLUMNS,
    )
    reclamos_scored_df = compute_reclamos_score(
        reclamos_df=reclamos_df,
        eps_obj=eps_universe,
    )
    cxp_scored_df = compute_cxp_revenue_score(
        eps_eeff_df=eps_eeff_df,
        eps_obj=eps_universe,
    )
    composite_kwargs = {
        "scored_df": results_scored,
        "market_share_df": market_share_df,
        "reclamos_score_df": reclamos_scored_df,
        "cxp_score_df": cxp_scored_df,
        "risk_weight": 0.55,
        "market_weight": 0.05,
        "complaints_weight": 0.2,
        "cxp_rev_weight": 0.2,
    }
    composite_base = results_scored.copy()
    try:
        sig = inspect.signature(build_composite_ranking)
        supported_kwargs = {
            key: value for key, value in composite_kwargs.items() if key in sig.parameters
        }
        built_out = build_composite_ranking(**supported_kwargs)
        if isinstance(built_out, tuple) and len(built_out) >= 1 and isinstance(built_out[0], pd.DataFrame):
            composite_base = built_out[0].copy()
    except Exception:
        composite_base = results_scored.copy()

    results_ranked, ranking_escenario, ranking_global = build_composite_ranking_local(
        base_df=composite_base,
        market_share_df=market_share_df,
        reclamos_score_df=reclamos_scored_df,
        cxp_score_df=cxp_scored_df,
        risk_weight=0.55,
        market_weight=0.05,
        complaints_weight=0.2,
        cxp_rev_weight=0.2,
    )
    return (
        diagnostics,
        market_share_df,
        reclamos_scored_df,
        cxp_scored_df,
        results_ranked,
        ranking_escenario,
        ranking_global,
        scenarios,
    )


with st.sidebar:
    st.header("Monte Carlo EPS")
    n_sim = int(
        st.number_input(
            "Numero de simulaciones",
            min_value=1_000,
            max_value=50_000,
            value=10_000,
            step=1_000,
        )
    )
    horizon_end = int(
        st.number_input(
            "Ano fin de simulacion",
            min_value=2026,
            max_value=2040,
            value=2030,
            step=1,
        )
    )
    upc_optimism_factor = float(
        st.slider(
            "UPC optimism factor",
            min_value=0.5,
            max_value=2.0,
            value=1.2,
            step=0.05,
        )
    )
    upc_growth_start_year = int(
        st.number_input("Ano inicio crecimiento UPC", min_value=2010, max_value=2030, value=2019, step=1)
    )
    upc_growth_end_year = int(
        st.number_input("Ano fin crecimiento UPC", min_value=2010, max_value=2035, value=2026, step=1)
    )
    if upc_growth_end_year < upc_growth_start_year:
        st.warning("Ano fin de crecimiento UPC menor al inicio; se intercambian para el calculo.")
        upc_growth_start_year, upc_growth_end_year = upc_growth_end_year, upc_growth_start_year

    st.subheader("Shocks por escenario")
    with st.expander("BASE", expanded=False):
        base_lr_shift = float(st.number_input("BASE LR shift", value=0.00, step=0.01, format="%.2f"))
        base_g_shift = float(st.number_input("BASE g shift", value=0.00, step=0.01, format="%.2f"))
    with st.expander("STRESS_LR", expanded=False):
        stress_lr_shift = float(st.number_input("STRESS_LR LR shift", value=0.05, step=0.01, format="%.2f"))
        stress_lr_g_shift = float(st.number_input("STRESS_LR g shift", value=0.00, step=0.01, format="%.2f"))
    with st.expander("STRESS_MIX", expanded=False):
        stress_mix_lr_shift = float(st.number_input("STRESS_MIX LR shift", value=0.05, step=0.01, format="%.2f"))
        stress_mix_g_shift = float(st.number_input("STRESS_MIX g shift", value=-0.03, step=0.01, format="%.2f"))

    st.subheader("Robustez del modelo")
    robustness_n_sim = int(
        st.number_input(
            "N sim robustez (backtesting/semillas)",
            min_value=500,
            max_value=20_000,
            value=3_000,
            step=500,
        )
    )
    backtest_start_year = int(
        st.number_input(
            "Ano inicio backtesting",
            min_value=2018,
            max_value=2030,
            value=2021,
            step=1,
        )
    )
    seeds_input = st.text_input(
        "Semillas (coma separadas)",
        value="7, 42, 77, 123, 2026",
    )
    run_robustness = st.button("Ejecutar robustez")

    st.caption(
        "Score final: 55% riesgo Monte Carlo + 5% mercado Valle + 20% reclamos + 20% CxP/Ingresos."
    )

render_score_methodology()

section_header("Datos fuente", "Modelo EPS")
explain_box(
    "Como se calcula",
    [
        "Se usa Cali ANALISIS.xlsx (hojas: EPS_EEFF, UPC, EPS_Edad, EPS_Anos, EPS_Afiliados, RECLAMOS).",
        "Las probabilidades se calculan con Monte Carlo en enfoque PROMEDIO.",
        "El % de mercado se calcula sobre todo Valle del Cauca (denominador total Valle).",
        "La tasa de reclamos por 10.000 afiliados se transforma a score por percentiles invertidos.",
        "CxP/Ingresos se calcula con cuentas por pagar comerciales sobre ingresos totales (menor = mejor).",
    ],
)

eps_eeff_df, eps_eeff_src = load_cali_sheet(["EPS_EEFF", "EPS EEFF"])
upc_df, upc_src = load_cali_sheet(["UPC"])
eps_edad_df, eps_edad_src = load_cali_sheet(["EPS_Edad", "EPS Edad"])
eps_anos_df, eps_anos_src = load_cali_sheet(["EPS_Años", "EPS_Anos", "EPS Años", "EPS Anos"])
eps_afiliados_df, eps_afiliados_src = load_cali_sheet(["EPS_Afiliados", "EPS Afiliados"])
reclamos_df, reclamos_src = load_cali_sheet(["RECLAMOS", "Reclamos"])
if reclamos_df.empty:
    reclamos_df, reclamos_src = load_cali_sheet_by_column_tokens(
        required_column_tokens=[
            ["eps"],
            ["tasa", "10"],
        ]
    )

sources_ok = [
    ("EPS_EEFF", eps_eeff_df, eps_eeff_src),
    ("UPC", upc_df, upc_src),
    ("EPS_Edad", eps_edad_df, eps_edad_src),
    ("EPS_Anos", eps_anos_df, eps_anos_src),
    ("EPS_Afiliados", eps_afiliados_df, eps_afiliados_src),
    ("RECLAMOS", reclamos_df, reclamos_src),
]
for label, df_src, detail in sources_ok:
    if df_src.empty:
        st.error(f"No se pudo cargar {label}. Detalle: {detail}")
        st.stop()

reclamos_validation_error = validate_reclamos_sheet(reclamos_df)
if reclamos_validation_error:
    st.error(f"No se pudo usar RECLAMOS. Detalle: {reclamos_validation_error}")
    st.stop()

with st.spinner("Ejecutando simulacion Monte Carlo..."):
    (
        diagnostics,
        market_share_df,
        reclamos_scored_df,
        cxp_scored_df,
        results_ranked,
        ranking_escenario,
        ranking_global,
        scenarios,
) = run_clientes_pipeline(
        eps_eeff_df=eps_eeff_df,
        upc_df=upc_df,
        eps_edad_df=eps_edad_df,
        eps_anos_df=eps_anos_df,
        eps_afiliados_df=eps_afiliados_df,
        reclamos_df=reclamos_df,
        n_sim=n_sim,
        horizon_end=horizon_end,
        upc_optimism_factor=upc_optimism_factor,
        upc_growth_start_year=upc_growth_start_year,
        upc_growth_end_year=upc_growth_end_year,
        base_lr_shift=base_lr_shift,
        base_g_shift=base_g_shift,
        stress_lr_shift=stress_lr_shift,
        stress_lr_g_shift=stress_lr_g_shift,
        stress_mix_lr_shift=stress_mix_lr_shift,
        stress_mix_g_shift=stress_mix_g_shift,
    )

eps_universe = get_eps_universe()
results_ranked = results_ranked[results_ranked["EPS"].isin(eps_universe)].copy()
ranking_escenario = ranking_escenario[ranking_escenario["EPS"].isin(eps_universe)].copy()
ranking_global = ranking_global[ranking_global["EPS"].isin(eps_universe)].copy()
market_share_df = market_share_df[market_share_df["EPS"].isin(eps_universe)].copy()
reclamos_scored_df = reclamos_scored_df[reclamos_scored_df["EPS"].isin(eps_universe)].copy()
cxp_scored_df = cxp_scored_df[cxp_scored_df["EPS"].isin(eps_universe)].copy()

ranking_global_exec = ranking_global.copy()
ranking_global_exec["Riesgo"] = ranking_global_exec["Score_Final"].map(risk_bucket)
ranking_global_exec["Imputada"] = ranking_global_exec["prob_imputada"].map({True: "Si", False: "No"})
ranking_global_exec["Imputada_Reclamos"] = ranking_global_exec["reclamos_imputado"].map(
    {True: "Si", False: "No"}
)
ranking_global_exec["Imputada_CxP"] = ranking_global_exec["cxp_rev_imputado"].map(
    {True: "Si", False: "No"}
)

robustness_seeds = parse_seed_values(seeds_input)
if len(robustness_seeds) < 2:
    robustness_seeds = [7, 42]

robustness_state_key = "clientes_robustness_v1"
if run_robustness:
    with st.spinner("Ejecutando backtesting y estabilidad por semilla..."):
        backtest_detail_df, backtest_summary_df, backtest_diag = run_eps_montecarlo_backtesting(
            upc_df=upc_df,
            eps_eeff_df=eps_eeff_df,
            eps_edad_df=eps_edad_df,
            eps_afiliados_hist_df=eps_anos_df,
            eps_obj=eps_universe,
            n_sim=robustness_n_sim,
            cash_thresholds=(15,),
            paydays_range=(20.0, 70.0),
            upc_optimism_factor=upc_optimism_factor,
            upc_growth_start_year=upc_growth_start_year,
            upc_growth_end_year=upc_growth_end_year,
            start_anchor_year=backtest_start_year,
            random_seed=42,
        )
        stability_out = run_eps_seed_stability(
            upc_df=upc_df,
            eps_eeff_df=eps_eeff_df,
            eps_edad_df=eps_edad_df,
            eps_afiliados_hist_df=eps_anos_df,
            market_share_df=market_share_df,
            reclamos_score_df=reclamos_scored_df,
            cxp_score_df=cxp_scored_df,
            eps_obj=eps_universe,
            seeds=robustness_seeds,
            n_sim=robustness_n_sim,
            horizon_end=horizon_end,
            cash_thresholds=(15,),
            paydays_range=(20.0, 70.0),
            upc_optimism_factor=upc_optimism_factor,
            upc_growth_start_year=upc_growth_start_year,
            upc_growth_end_year=upc_growth_end_year,
            scenarios=scenarios,
            risk_weight=0.55,
            market_weight=0.05,
            complaints_weight=0.2,
            cxp_rev_weight=0.2,
        )
        st.session_state[robustness_state_key] = {
            "backtest_detail": backtest_detail_df,
            "backtest_summary": backtest_summary_df,
            "backtest_diag": backtest_diag,
            "stability_out": stability_out,
            "params": {
                "n_sim": robustness_n_sim,
                "start_anchor_year": backtest_start_year,
                "seeds": robustness_seeds,
            },
        }

robustness_payload = st.session_state.get(robustness_state_key)

eps_ratios_df, eps_ratio_years = build_eps_ratio_dataset(
    eps_eeff_df=eps_eeff_df,
    eps_universe=eps_universe,
)
available_ratio_indicators = [
    ratio
    for ratio in EPS_RATIO_LABELS.keys()
    if ratio in eps_ratios_df.columns and not pd.to_numeric(eps_ratios_df[ratio], errors="coerce").dropna().empty
]

tab_analisis, tab_datos, tab_eps, tab_comp = st.tabs(
    ["Analisis", "Datos", "Analisis EPS", "Comparacion"]
)

with tab_analisis:
    section_header("Resumen ejecutivo", "Ranking Monte Carlo + mercado Valle + reclamos + CxP/Ingresos")
    scen_choice = st.selectbox("Escenario", list(scenarios.keys()), index=0)

    ranking_exec = ranking_escenario[ranking_escenario["Escenario"] == scen_choice].copy()
    ranking_exec["Riesgo"] = ranking_exec["Score_Final"].map(risk_bucket)
    ranking_exec["Imputada"] = ranking_exec["prob_imputada"].map({True: "Si", False: "No"})
    ranking_exec["Imputada_Reclamos"] = ranking_exec["reclamos_imputado"].map({True: "Si", False: "No"})
    ranking_exec["Imputada_CxP"] = ranking_exec["cxp_rev_imputado"].map({True: "Si", False: "No"})

    top_eps = ranking_exec.sort_values("Ranking_Escenario_Final").head(1)
    top_name = top_eps["EPS"].iloc[0] if not top_eps.empty else "NA"
    top_score = float(top_eps["Score_Final"].iloc[0]) if not top_eps.empty else np.nan
    mean_score = float(ranking_exec["Score_Final"].mean()) if not ranking_exec.empty else np.nan
    top3_share = (
        float(ranking_exec.sort_values("MarketShare_Valle", ascending=False).head(3)["MarketShare_Valle"].sum())
        if not ranking_exec.empty
        else np.nan
    )
    imputadas_n = int(ranking_exec["prob_imputada"].sum()) if not ranking_exec.empty else 0
    resumen_exec = pd.DataFrame(
        [
            {"Indicador": "EPS evaluadas", "Valor": f"{len(ranking_exec):,}"},
            {"Indicador": "Lider escenario", "Valor": str(top_name).upper()},
            {"Indicador": "Score lider", "Valor": f"{top_score:.1f}" if pd.notna(top_score) else "NA"},
            {"Indicador": "Score promedio", "Valor": f"{mean_score:.1f}" if pd.notna(mean_score) else "NA"},
            {"Indicador": "Mercado Top 3", "Valor": f"{top3_share:.1%}" if pd.notna(top3_share) else "NA"},
            {"Indicador": "EPS con imputacion EEFF", "Valor": f"{imputadas_n}"},
        ]
    )
    st.dataframe(resumen_exec, width="stretch", hide_index=True)

    divider()
    section_header("Tablero de decision", "Top y alertas por escenario seleccionado")
    c_top, c_alert = st.columns(2)

    with c_top:
        st.caption("Top 5 recomendado")
        top5 = ranking_exec.sort_values("Ranking_Escenario_Final").head(5).copy()
        st.dataframe(
            top5[
                [
                    "Ranking_Escenario_Final",
                    "EPS",
                    "Riesgo",
                    "Score_Final",
                    "Score_Riesgo",
                    "Score_Mercado",
                    "Score_Reclamos",
                    "Score_CxP_REV",
                    "Tasa_Reclamos_10k",
                    "CxP_over_REV",
                    "MarketShare_Valle",
                    "Imputada",
                    "Imputada_Reclamos",
                    "Imputada_CxP",
                ]
            ].style.format(
                {
                    "Score_Final": "{:.1f}",
                    "Score_Riesgo": "{:.1f}",
                    "Score_Mercado": "{:.1f}",
                    "Score_Reclamos": "{:.1f}",
                    "Score_CxP_REV": "{:.1f}",
                    "Tasa_Reclamos_10k": "{:.2f}",
                    "CxP_over_REV": "{:.2%}",
                    "MarketShare_Valle": "{:.2%}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    with c_alert:
        st.caption("Alertas (ultimas 5)")
        last5 = ranking_exec.sort_values("Ranking_Escenario_Final", ascending=False).head(5).copy()
        st.dataframe(
            last5[
                [
                    "Ranking_Escenario_Final",
                    "EPS",
                    "Riesgo",
                    "Score_Final",
                    "Score_Riesgo",
                    "Score_Reclamos",
                    "Score_CxP_REV",
                    "Tasa_Reclamos_10k",
                    "CxP_over_REV",
                    "MarketShare_Valle",
                    "Imputada",
                    "Imputada_Reclamos",
                    "Imputada_CxP",
                ]
            ].style.format(
                {
                    "Score_Final": "{:.1f}",
                    "Score_Riesgo": "{:.1f}",
                    "Score_Reclamos": "{:.1f}",
                    "Score_CxP_REV": "{:.1f}",
                    "Tasa_Reclamos_10k": "{:.2f}",
                    "CxP_over_REV": "{:.2%}",
                    "MarketShare_Valle": "{:.2%}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    divider()
    section_header(
        "Ranking por escenario",
        "Score final = 55% riesgo + 5% mercado + 20% reclamos + 20% CxP/Ingresos",
    )
    st.dataframe(
        ranking_exec[
            [
                "Escenario",
                "Ranking_Escenario_Final",
                "EPS",
                "Riesgo",
                "Score_Final",
                "Score_Riesgo",
                "Score_Mercado",
                "Score_Reclamos",
                "Score_CxP_REV",
                "Tasa_Reclamos_10k",
                "CxP_over_REV",
                "MarketShare_Valle",
                "Imputada",
                "Imputada_Reclamos",
                "Imputada_CxP",
            ]
        ].style.format(
            {
                "Score_Final": "{:.1f}",
                "Score_Riesgo": "{:.1f}",
                "Score_Mercado": "{:.1f}",
                "Score_Reclamos": "{:.1f}",
                "Score_CxP_REV": "{:.1f}",
                "Tasa_Reclamos_10k": "{:.2f}",
                "CxP_over_REV": "{:.2%}",
                "MarketShare_Valle": "{:.2%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    scen_plot = ranking_exec.sort_values("Ranking_Escenario_Final").head(10)
    if not scen_plot.empty:
        fig = px.bar(
            scen_plot.sort_values("Score_Final", ascending=True),
            x="Score_Final",
            y="EPS",
            orientation="h",
            color="Riesgo",
            title=f"Top 10 EPS por Score Final - {scen_choice}",
            labels={"Score_Final": "Score Final", "EPS": "EPS"},
            color_discrete_map={
                "Bajo riesgo": "#0f6a62",
                "Riesgo medio": "#e9a03b",
                "Alto riesgo": "#c25416",
                "Sin dato": "#7d8b8b",
            },
        )
        fig = style_chart(fig)
        chart_container(fig)

    divider()
    section_header("Ranking global", "Promedio de escenarios")
    st.dataframe(
        ranking_global_exec[
            [
                "Ranking_Global_Final",
                "EPS",
                "Riesgo",
                "Score_Final",
                "Score_Riesgo",
                "Score_Mercado",
                "Score_Reclamos",
                "Score_CxP_REV",
                "Tasa_Reclamos_10k",
                "CxP_over_REV",
                "MarketShare_Valle",
                "Afiliados_Valle",
                "Imputada",
                "Imputada_Reclamos",
                "Imputada_CxP",
            ]
        ].style.format(
            {
                "Score_Final": "{:.1f}",
                "Score_Riesgo": "{:.1f}",
                "Score_Mercado": "{:.1f}",
                "Score_Reclamos": "{:.1f}",
                "Score_CxP_REV": "{:.1f}",
                "Tasa_Reclamos_10k": "{:.2f}",
                "CxP_over_REV": "{:.2%}",
                "MarketShare_Valle": "{:.2%}",
                "Afiliados_Valle": "{:,.0f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    divider()
    section_header(
        "Robustez del modelo",
        "Backtesting historico + estabilidad por semilla (Spearman de rankings)",
    )
    if robustness_payload is None:
        st.info(
            "Ejecuta 'Robustez del modelo' en el sidebar para calcular backtesting y estabilidad por semilla."
        )
    else:
        params_view = pd.DataFrame(
            [
                {
                    "Parametro": "N sim robustez",
                    "Valor": robustness_payload.get("params", {}).get("n_sim"),
                },
                {
                    "Parametro": "Ano inicio backtesting",
                    "Valor": robustness_payload.get("params", {}).get("start_anchor_year"),
                },
                {
                    "Parametro": "Semillas",
                    "Valor": ", ".join(str(x) for x in robustness_payload.get("params", {}).get("seeds", [])),
                },
            ]
        )
        st.dataframe(params_view, width="stretch", hide_index=True)

        backtest_summary_df = robustness_payload.get("backtest_summary", pd.DataFrame()).copy()
        backtest_detail_df = robustness_payload.get("backtest_detail", pd.DataFrame()).copy()
        backtest_diag = robustness_payload.get("backtest_diag", {})

        st.markdown("**Backtesting (1-step ahead)**")
        if backtest_summary_df.empty:
            st.info("No se pudo construir backtesting con la historia disponible.")
        else:
            st.dataframe(
                backtest_summary_df.style.format(
                    {
                        "PredictedMean": "{:.2%}",
                        "ObservedRate": "{:.2%}",
                        "MAE": "{:.2%}",
                        "Brier": "{:.4f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            backtest_resume = pd.DataFrame(
                [
                    {
                        "Indicador": "Ventanas anchor usadas",
                        "Valor": len(backtest_diag.get("anchors_used", [])),
                    },
                    {
                        "Indicador": "Observaciones evaluadas",
                        "Valor": int(len(backtest_detail_df)),
                    },
                    {
                        "Indicador": "MAE promedio",
                        "Valor": f"{backtest_summary_df['MAE'].mean():.2%}",
                    },
                    {
                        "Indicador": "Brier promedio",
                        "Valor": f"{backtest_summary_df['Brier'].mean():.4f}",
                    },
                ]
            )
            st.dataframe(backtest_resume, width="stretch", hide_index=True)

        stability_out = robustness_payload.get("stability_out", {})
        global_summary_df = stability_out.get("summary_global", pd.DataFrame()).copy()
        pairwise_global_df = stability_out.get("pairwise_global", pd.DataFrame()).copy()
        escenario_summary_df = stability_out.get("summary_escenario", pd.DataFrame()).copy()

        divider()
        st.markdown("**Estabilidad por semilla (Spearman)**")
        if global_summary_df.empty:
            st.info("No se pudo calcular estabilidad por semilla.")
        else:
            st.dataframe(
                global_summary_df.style.format(
                    {
                        "Mean": "{:.3f}",
                        "Min": "{:.3f}",
                        "Max": "{:.3f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            if not pairwise_global_df.empty:
                st.caption("Detalle pairwise global entre semillas")
                st.dataframe(
                    pairwise_global_df.style.format(
                        {
                            "Spearman_Score_Final": "{:.3f}",
                            "Spearman_Ranking_Global": "{:.3f}",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )
            if not escenario_summary_df.empty:
                st.caption("Resumen Spearman por escenario")
                st.dataframe(
                    escenario_summary_df.style.format(
                        {
                            "Mean": "{:.3f}",
                            "Min": "{:.3f}",
                            "Max": "{:.3f}",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )

    with st.expander("Ver detalle tecnico de probabilidades"):
        section_header("Probabilidades base", "PROMEDIO por escenario")
        explain_box(
            "Como se calcula",
            [
                "P_avg_* = proporcion promedio de anos de incumplimiento en la simulacion.",
                "Se imputan probabilidades faltantes por promedio del escenario.",
                "prob_imputada identifica EPS con faltantes historicos en EEFF.",
                "Score_Reclamos se obtiene por percentil invertido de Tasa_Reclamos_10k.",
                "Score_CxP_REV se obtiene por percentil invertido de CxP_over_REV.",
            ],
        )
        prob_cols = (
            ["Escenario", "EPS"]
            + PROBABILITY_COLUMNS
            + [
                "Tasa_Reclamos_10k",
                "Score_Reclamos",
                "CxP_over_REV",
                "Score_CxP_REV",
                "reclamos_imputado",
                "motivo_imputacion_reclamos",
                "cxp_rev_imputado",
                "motivo_imputacion_cxp_rev",
                "prob_imputada",
                "motivo_imputacion",
            ]
        )
        prob_view = results_ranked[prob_cols].copy().sort_values(["Escenario", "EPS"])
        prob_view["prob_imputada"] = prob_view["prob_imputada"].map({True: "Si", False: "No"})
        prob_view["reclamos_imputado"] = prob_view["reclamos_imputado"].map({True: "Si", False: "No"})
        st.dataframe(
            prob_view.style.format(
                {col: "{:.2%}" for col in PROBABILITY_COLUMNS}
                | {
                    "Score_Reclamos": "{:.1f}",
                    "Tasa_Reclamos_10k": "{:.2f}",
                    "CxP_over_REV": "{:.2%}",
                    "Score_CxP_REV": "{:.1f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

with tab_datos:
    section_header("% de mercado sobre todo Valle", "Denominador total de afiliados del departamento")
    explain_box(
        "Como se calcula",
        [
            "Se usa EPS_Afiliados filtrando Departamento = Valle del Cauca.",
            "Numerador: afiliados de cada EPS objetivo.",
            "Denominador: afiliados totales de todas las EPS en Valle.",
        ],
    )

    share_view = market_share_df.copy().sort_values("MarketShare_Valle", ascending=False)
    share_view = share_view.rename(columns={"MarketShare_Valle": "% Mercado Valle"})
    st.dataframe(
        share_view.style.format({"Afiliados_Valle": "{:,.0f}", "% Mercado Valle": "{:.2%}"}),
        width="stretch",
        hide_index=True,
    )

    divider()
    section_header("Reclamos por EPS", "Tasa x cada 10.000 afiliados (menor = mejor)")
    st.dataframe(
        reclamos_scored_df.sort_values("Tasa_Reclamos_10k", ascending=True).style.format(
            {
                "Tasa_Reclamos_10k": "{:.2f}",
                "Reclamos": "{:,.0f}",
                "Score_Reclamos": "{:.1f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    divider()
    section_header("CxP comerciales sobre ingresos", "Factor adicional de score (menor = mejor)")
    st.dataframe(
        cxp_scored_df.sort_values("CxP_over_REV", ascending=True).style.format(
            {
                "CxP_Comercial": "{:,.0f}",
                "Ingresos": "{:,.0f}",
                "CxP_over_REV": "{:.2%}",
                "Score_CxP_REV": "{:.1f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    divider()
    section_header("Afiliados mayores de 55 anos", "Valle del Cauca")
    age55_df = affiliates_55_table(eps_edad_df)
    if age55_df is None or age55_df.empty:
        st.info("No fue posible construir la tabla de afiliados mayores de 55 anos.")
    else:
        st.dataframe(
            age55_df.style.format(
                {
                    "Femenino": "{:,.0f}",
                    "Masculino": "{:,.0f}",
                    "Total afiliados": "{:,.0f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    divider()
    section_header("Composicion por regimen", "Valle del Cauca")
    reg_df = regimen_valle_table(eps_afiliados_df)
    if reg_df is None or reg_df.empty:
        st.info("No fue posible construir la composicion por regimen.")
    else:
        st.dataframe(
            reg_df.style.format(
                {
                    "Contributivo": "{:,.0f}",
                    "Subsidiado": "{:,.0f}",
                    "Especiales/Excepcion": "{:,.0f}",
                    "Total afiliados": "{:,.0f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

with tab_eps:
    section_header("Analisis EPS", "Vista unica: estado de resultados + probabilidades")
    selected_eps = st.selectbox(
        "EPS",
        eps_universe,
        index=0,
        format_func=lambda x: str(x).upper(),
    )

    eps_global = ranking_global_exec[ranking_global_exec["EPS"] == selected_eps].copy()
    rank_global = int(eps_global["Ranking_Global_Final"].iloc[0]) if not eps_global.empty else np.nan
    score_global = float(eps_global["Score_Final"].iloc[0]) if not eps_global.empty else np.nan
    riesgo_global = str(eps_global["Riesgo"].iloc[0]) if not eps_global.empty else "Sin dato"
    share_valle = float(eps_global["MarketShare_Valle"].iloc[0]) if not eps_global.empty else np.nan
    score_reclamos_global = float(eps_global["Score_Reclamos"].iloc[0]) if not eps_global.empty else np.nan
    tasa_reclamos_global = float(eps_global["Tasa_Reclamos_10k"].iloc[0]) if not eps_global.empty else np.nan
    score_cxp_global = float(eps_global["Score_CxP_REV"].iloc[0]) if not eps_global.empty else np.nan
    cxp_over_rev_global = float(eps_global["CxP_over_REV"].iloc[0]) if not eps_global.empty else np.nan
    resumen_eps = pd.DataFrame(
        [
            {"Indicador": "Ranking global", "Valor": f"#{rank_global}" if pd.notna(rank_global) else "NA"},
            {"Indicador": "Score global", "Valor": f"{score_global:.1f}" if pd.notna(score_global) else "NA"},
            {"Indicador": "Riesgo", "Valor": riesgo_global},
            {"Indicador": "% mercado Valle", "Valor": f"{share_valle:.2%}" if pd.notna(share_valle) else "NA"},
            {
                "Indicador": "Score reclamos",
                "Valor": f"{score_reclamos_global:.1f}" if pd.notna(score_reclamos_global) else "NA",
            },
            {
                "Indicador": "Tasa reclamos 10k",
                "Valor": f"{tasa_reclamos_global:.2f}" if pd.notna(tasa_reclamos_global) else "NA",
            },
            {
                "Indicador": "Score CxP/Ingresos",
                "Valor": f"{score_cxp_global:.1f}" if pd.notna(score_cxp_global) else "NA",
            },
            {
                "Indicador": "CxP/Ingresos",
                "Valor": f"{cxp_over_rev_global:.2%}" if pd.notna(cxp_over_rev_global) else "NA",
            },
        ]
    )
    st.dataframe(resumen_eps, width="stretch", hide_index=True)

    divider()
    section_header(
        "Indicadores financieros historicos (EPS)",
        "Misma metodologia de ratios usada en Analisis IPS",
    )
    if eps_ratios_df.empty or not eps_ratio_years:
        st.info("No hay informacion suficiente en EPS_EEFF para calcular indicadores financieros EPS.")
    else:
        ratio_tabs = st.tabs(["Liquidez", "Solvencia", "Rentabilidad", "Eficiencia"])
        ratio_tab_order = ["Liquidez", "Solvencia", "Rentabilidad", "Eficiencia"]
        for ratio_tab, group_name in zip(ratio_tabs, ratio_tab_order):
            with ratio_tab:
                table = build_eps_ratio_table(
                    ratios_df=eps_ratios_df,
                    selected_eps=selected_eps,
                    ratio_list=EPS_RATIO_GROUPS[group_name],
                    years=eps_ratio_years,
                )
                if table.empty:
                    st.info(f"No hay datos para {group_name.lower()} en la EPS seleccionada.")
                else:
                    st.dataframe(table, width="stretch", hide_index=True)

    divider()
    section_header("Estado de resultados (anual)")
    income_df = build_income_statement_view(diagnostics["base_eps"], selected_eps)
    if income_df.empty:
        st.info("No hay estado de resultados para la EPS seleccionada.")
    else:
        st.dataframe(
            income_df.style.format(
                {
                    "Ingresos": "{:,.0f}",
                    "OPEX_caja": "{:,.0f}",
                    "EBITDA": "{:,.0f}",
                    "EBIT": "{:,.0f}",
                    "Utilidad_Neta": "{:,.0f}",
                    "Margen_EBITDA": "{:.2%}",
                    "Margen_EBIT": "{:.2%}",
                    "Margen_Neto": "{:.2%}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    divider()
    section_header("Cumplimiento historico CM/PA/RI", "Solo EPS - calculado con historico contable")
    compliance_df = build_eps_historical_compliance(diagnostics["base_eps"], selected_eps)
    if compliance_df.empty:
        st.info("No hay datos historicos suficientes para calcular cumplimiento CM/PA/RI.")
    else:
        compliance_show = compliance_df.copy()
        for col in ["Cumple_CM", "Cumple_PA", "Cumple_RI", "Cumple_3_de_3"]:
            compliance_show[col] = compliance_show[col].map({True: "Si", False: "No"})
        st.dataframe(
            compliance_show.style.format(
                {
                    "CM_ratio": "{:.2f}",
                    "PA_ratio": "{:.2f}",
                    "RI_ratio": "{:.2f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    divider()
    section_header("Probabilidades y scoring de la EPS", "Resultados por escenario")
    eps_prob_cols = [
        "Escenario",
        "Ranking_Escenario_Final",
        "EPS",
        "Score_Final",
        "Score_Riesgo",
        "Score_Mercado",
        "Score_Reclamos",
        "Score_CxP_REV",
        "Tasa_Reclamos_10k",
        "Reclamos",
        "CxP_over_REV",
        "MarketShare_Valle",
    ] + PROBABILITY_COLUMNS + [
        "prob_imputada",
        "motivo_imputacion",
        "reclamos_imputado",
        "motivo_imputacion_reclamos",
        "cxp_rev_imputado",
        "motivo_imputacion_cxp_rev",
    ]
    eps_probs = (
        ranking_escenario[ranking_escenario["EPS"] == selected_eps][eps_prob_cols]
        .sort_values("Escenario")
        .reset_index(drop=True)
    )

    if eps_probs.empty:
        st.info("No hay probabilidades calculadas para la EPS seleccionada.")
    else:
        eps_probs["Riesgo"] = eps_probs["Score_Final"].map(risk_bucket)
        eps_probs["prob_imputada"] = eps_probs["prob_imputada"].map({True: "Si", False: "No"})
        eps_probs["reclamos_imputado"] = eps_probs["reclamos_imputado"].map({True: "Si", False: "No"})
        eps_probs["cxp_rev_imputado"] = eps_probs["cxp_rev_imputado"].map({True: "Si", False: "No"})
        eps_fmt = {
            "Score_Final": "{:.1f}",
            "Score_Riesgo": "{:.1f}",
            "Score_Mercado": "{:.1f}",
            "Score_Reclamos": "{:.1f}",
            "Score_CxP_REV": "{:.1f}",
            "Tasa_Reclamos_10k": "{:.2f}",
            "Reclamos": "{:,.0f}",
            "CxP_over_REV": "{:.2%}",
            "MarketShare_Valle": "{:.2%}",
        }
        eps_fmt.update({col: "{:.2%}" for col in PROBABILITY_COLUMNS})
        st.dataframe(
            eps_probs.style.format(eps_fmt),
            width="stretch",
            hide_index=True,
        )

    divider()
    section_header("Fuentes cargadas")
    st.caption(eps_eeff_src)
    st.caption(upc_src)
    st.caption(eps_edad_src)
    st.caption(eps_anos_src)
    st.caption(eps_afiliados_src)
    st.caption(reclamos_src)

with tab_comp:
    section_header(
        "Comparacion de indicadores EPS",
        "Selector dinamico por indicador para comparar todas las EPS",
    )
    if eps_ratios_df.empty or not available_ratio_indicators:
        st.info("No hay informacion suficiente para construir la comparacion de indicadores EPS.")
    else:
        c_ind, c_per = st.columns([2, 1])
        selected_indicator = c_ind.selectbox(
            "Indicador financiero",
            available_ratio_indicators,
            format_func=lambda key: EPS_RATIO_LABELS.get(key, key),
        )
        period_options = ["Promedio"] + [str(year) for year in eps_ratio_years]
        selected_period = c_per.selectbox("Periodo", period_options, index=0)

        compare_df = build_eps_indicator_comparison(
            ratios_df=eps_ratios_df,
            indicator=selected_indicator,
            period=selected_period,
        )
        if compare_df.empty:
            st.info("No hay datos para ese indicador en el periodo seleccionado.")
        else:
            direction = RATIO_SPECS.get(selected_indicator, {}).get("direction", "higher")
            if direction == "lower":
                compare_df["Orden"] = compare_df["Valor"]
                compare_df = compare_df.sort_values(["Orden", "EPS"], ascending=[True, True]).reset_index(drop=True)
                direction_msg = "Sentido del indicador: menor valor es mejor."
            elif direction == "range":
                target_mid = 40.0
                compare_df["Orden"] = (compare_df["Valor"] - target_mid).abs()
                compare_df = compare_df.sort_values(["Orden", "EPS"], ascending=[True, True]).reset_index(drop=True)
                direction_msg = "Sentido del indicador: mejor cuando esta cerca del rango objetivo de dias."
            else:
                compare_df["Orden"] = compare_df["Valor"]
                compare_df = compare_df.sort_values(["Orden", "EPS"], ascending=[False, True]).reset_index(drop=True)
                direction_msg = "Sentido del indicador: mayor valor es mejor."

            compare_df["Ranking"] = np.arange(1, len(compare_df) + 1)
            ratio_kind = EPS_RATIO_KINDS.get(selected_indicator, "ratio")
            compare_table = compare_df.copy()
            compare_table["Valor"] = compare_table["Valor"].map(
                lambda val: format_ratio_value(val, ratio_kind)
            )
            st.caption(direction_msg)
            st.dataframe(
                compare_table[["Ranking", "EPS", "Valor", "Periodo"]],
                width="stretch",
                hide_index=True,
            )

            plot_df = compare_df.sort_values("Ranking", ascending=False)
            fig = px.bar(
                plot_df,
                x="Valor",
                y="EPS",
                orientation="h",
                title=f"{EPS_RATIO_LABELS.get(selected_indicator, selected_indicator)} - {selected_period}",
                labels={"Valor": EPS_RATIO_LABELS.get(selected_indicator, selected_indicator), "EPS": "EPS"},
            )
            if ratio_kind == "percent":
                fig.update_xaxes(tickformat=".1%")
            elif ratio_kind == "days":
                fig.update_xaxes(tickformat=",.0f")
            elif ratio_kind == "x":
                fig.update_xaxes(tickformat=".2f")
            fig = style_chart(fig)
            chart_container(fig)
