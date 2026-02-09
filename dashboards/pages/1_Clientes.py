from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboards.data_loader import load_cifras_eps, load_eps_financials
from dashboards.ui import apply_theme, chart_container, divider, page_header, section_header, style_chart
from src.models.eps_scoring import (
    BLOCK_DEFS,
    RATIO_SPECS,
    apply_size_factors,
    aggregate_scores,
    build_blocks_long,
    compute_ratios,
    compute_net_income_factor,
    compute_revenue_factor,
    score_ratios,
    winsorize_ratios,
)


st.set_page_config(page_title="Clientes", layout="wide")
apply_theme()

page_header(
    "Clientes",
    "Indicadores financieros y scoring basados en los estados de las EPS.",
    "EPS financials",
)

PAGE_ID = "clientes"
DEFAULT_WEIGHTS = {
    "liquidity": 30,
    "solvency": 30,
    "profitability": 20,
    "efficiency": 20,
}
if st.session_state.get("active_page") != PAGE_ID:
    st.session_state["active_page"] = PAGE_ID
    st.session_state["clientes_weight_liquidity"] = DEFAULT_WEIGHTS["liquidity"]
    st.session_state["clientes_weight_solvency"] = DEFAULT_WEIGHTS["solvency"]
    st.session_state["clientes_weight_profitability"] = DEFAULT_WEIGHTS["profitability"]
    st.session_state["clientes_weight_efficiency"] = DEFAULT_WEIGHTS["efficiency"]

with st.sidebar:
    st.header("Filters")
    st.selectbox("Escenario", ["Base", "Conservador", "Agresivo"], index=0)
    st.slider("Horizonte (anos)", 3, 10, 5)
    st.slider("Market share objetivo", 0.02, 0.15, 0.08)
    st.caption("Archivo EPS: Cali ANALISIS.xlsx (hoja EPS EEFF)")
    k_age = st.number_input("k (FactorEdad)", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
    st.caption(
        "Nota: k controla cuánto pesan los grupos de mayor edad; a mayor k, más peso para edades altas."
    )
    st.subheader("Pesos del Score Financiero")
    w_liquidity = st.slider(
        "Liquidez (%)",
        0,
        100,
        key="clientes_weight_liquidity",
    )
    w_solvency = st.slider(
        "Endeudamiento (%)",
        0,
        100,
        key="clientes_weight_solvency",
    )
    w_profitability = st.slider(
        "Rentabilidad (%)",
        0,
        100,
        key="clientes_weight_profitability",
    )
    w_efficiency = st.slider(
        "Eficiencia (%)",
        0,
        100,
        key="clientes_weight_efficiency",
    )

section_header("Datos fuente", "EPS")

eps_df, eps_source = load_eps_financials("EPS EEFF")
age_df, age_source = load_cifras_eps("EPS_Edad")
mun_df, mun_source = load_cifras_eps("EPS_Afiliados")


def normalize_account(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = " ".join(normalized.lower().split())
    return normalized


def normalize_eps_name(text: str) -> str:
    cleaned = normalize_account(text)
    cleaned = cleaned.replace(".xlsx", "").strip()
    cleaned = " ".join(cleaned.split())
    return cleaned


def parse_age_group(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    normalized = normalize_account(text)
    if "80" in normalized and ("mas" in normalized or "+" in normalized):
        return "80+"
    nums = re.findall(r"\d+", normalized)
    if len(nums) >= 2:
        start = int(nums[0])
        end = int(nums[1])
        return f"{start}-{end}"
    return None


def find_col(columns: List[str], includes: List[str]) -> str | None:
    for col in columns:
        norm = normalize_account(col)
        if any(token in norm for token in includes):
            return col
    return None


def valle_eps_keys_from_mun(df: pd.DataFrame) -> set[str] | None:
    if df.empty:
        return None
    cols = [str(c) for c in df.columns]
    dept_col = find_col(cols, ["departamento"])
    eps_col = find_col(cols, ["eps"])
    if not dept_col or not eps_col:
        return None
    work = df[[dept_col, eps_col]].copy()
    work[dept_col] = work[dept_col].astype(str)
    work = work[work[dept_col].str.contains("valle del cauca", case=False, na=False)]
    if work.empty:
        return None
    eps_keys = work[eps_col].astype(str).str.strip().map(normalize_eps_name).dropna().unique().tolist()
    return set(eps_keys)


def fmt_ratio(value: float | None, kind: str) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "NA"
    if kind == "percent":
        return f"{value * 100:.1f}%"
    if kind == "days":
        return f"{value:,.0f} dias"
    if kind == "x":
        return f"{value:.2f}x"
    return f"{value:,.2f}"


def fmt_currency_millions(value: float | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "NA"
    return f"${value:,.0f}"


def compute_delta(curr: float | None, prev: float | None) -> float | None:
    if curr is None or prev is None:
        return None
    if pd.isna(curr) or pd.isna(prev) or prev == 0:
        return None
    delta = (curr - prev) / abs(prev)
    if pd.isna(delta):
        return None
    return float(delta)


def distance_to_range(value: float | None, low: float = 20, high: float = 60) -> float | None:
    if value is None or pd.isna(value):
        return None
    if low <= value <= high:
        return 0.0
    if value < low:
        return float(low - value)
    return float(value - high)


def improvement_flag(
    curr: float | None,
    prev: float | None,
    direction: str,
    low: float = 20,
    high: float = 60,
) -> float | None:
    if curr is None or prev is None:
        return None
    if pd.isna(curr) or pd.isna(prev) or prev == 0:
        return None
    if direction == "higher":
        if curr > prev:
            return 1.0
        if curr < prev:
            return -1.0
        return 0.0
    if direction == "lower":
        if curr < prev:
            return 1.0
        if curr > prev:
            return -1.0
        return 0.0
    if direction == "range":
        dist_curr = distance_to_range(curr, low, high)
        dist_prev = distance_to_range(prev, low, high)
        if dist_curr is None or dist_prev is None:
            return None
        if dist_curr < dist_prev:
            return 1.0
        if dist_curr > dist_prev:
            return -1.0
        return 0.0
    return None


def style_ratio_table(
    df: pd.DataFrame, flags_df: pd.DataFrame, delta_cols: List[str]
) -> pd.io.formats.style.Styler:
    def fmt_delta(val: float | None) -> str:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ""
        if val > 0:
            return f"▲ {val * 100:.1f}%"
        if val < 0:
            return f"▼ {abs(val) * 100:.1f}%"
        return "0.0%"

    def _style_row(row: pd.Series) -> List[str]:
        styles = []
        for col in row.index:
            if col in delta_cols:
                flag = None
                if col in flags_df.columns and row.name in flags_df.index:
                    flag = flags_df.loc[row.name, col]
                if flag is None or (isinstance(flag, float) and pd.isna(flag)) or flag == 0:
                    styles.append("")
                elif flag > 0:
                    styles.append("color: #0f6a62; font-weight: 600;")
                else:
                    styles.append("color: #c92a2a; font-weight: 600;")
            else:
                styles.append("")
        return styles

    styled = df.style.format({col: fmt_delta for col in delta_cols})
    styled = styled.apply(_style_row, axis=1)
    return styled


INTERVENED_KEYWORDS = [
    "nueva eps",
    "sanitas",
    "sos",
    "famisanar",
    "asmetsalud",
    "emssanar",
    "coosalud",
    "eps familiar",
]

WARM_COLORS = ["#c92a2a", "#e8590c", "#f08c00", "#d9480f", "#ff922b", "#fa5252"]
COOL_COLORS = ["#0f6a62", "#1c7ed6", "#2f4858", "#228be6", "#0b7285", "#4c6ef5"]

EXCLUDED_EPS = [
    "ferronales",
    "ferronales - eas",
    "asmetsalud",
    "asmet salud",
    "comfenalco valle",
    "comfenalco valle eps",
    "eps familiar",
    "eps familiar de colombia",
]
EXCLUDED_EPS_KEYS = {normalize_eps_name(name) for name in EXCLUDED_EPS}


def is_intervened_eps(name: str) -> bool:
    norm = normalize_eps_name(name)
    compact = re.sub(r"[^a-z0-9]+", "", norm)
    for key in INTERVENED_KEYWORDS:
        if key in norm:
            return True
        key_compact = re.sub(r"[^a-z0-9]+", "", key)
        if key_compact and key_compact in compact:
            return True
    return False


def is_excluded_eps(name: str) -> bool:
    norm = normalize_eps_name(name)
    if norm in EXCLUDED_EPS_KEYS:
        return True
    compact = re.sub(r"[^a-z0-9]+", "", norm)
    for key in EXCLUDED_EPS_KEYS:
        if key in norm:
            return True
        key_compact = re.sub(r"[^a-z0-9]+", "", key)
        if key_compact and key_compact in compact:
            return True
    return False


def build_color_map(names: List[str]) -> Dict[str, str]:
    color_map: Dict[str, str] = {}
    warm_i = 0
    cool_i = 0
    for name in names:
        if is_intervened_eps(name):
            color_map[name] = WARM_COLORS[warm_i % len(WARM_COLORS)]
            warm_i += 1
        else:
            color_map[name] = COOL_COLORS[cool_i % len(COOL_COLORS)]
            cool_i += 1
    return color_map


def apply_line_chart_style(fig: go.Figure, legend_title: str = "EPS") -> go.Figure:
    fig = style_chart(fig)
    fig.update_layout(
        showlegend=True,
        legend_title_text=legend_title,
        legend=dict(
            orientation="h",
            x=0.01,
            xanchor="left",
            y=0.01,
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(11,31,31,0.25)",
            borderwidth=1,
            font=dict(color="#000000"),
        ),
        font=dict(color="#000000"),
        title_font=dict(size=16, family="Newsreader", color="#000000"),
        title_x=0.5,
        title_xanchor="center",
        margin=dict(l=18, r=18, t=36, b=24),
    )
    fig.update_xaxes(tickfont=dict(color="#000000"), title_font=dict(color="#000000"))
    fig.update_yaxes(tickfont=dict(color="#000000"), title_font=dict(color="#000000"))
    return fig


if eps_df.empty:
    st.warning("No se encontro el archivo de estados financieros EPS o no se pudo leer.")
    st.caption(f"Detalle: {eps_source}")
    st.stop()

valle_eps_keys = valle_eps_keys_from_mun(mun_df)

year_cols = [c for c in eps_df.columns if str(c).strip().isdigit()]
year_cols = sorted(year_cols, key=lambda x: int(x))
if not year_cols:
    st.warning("No se encontraron columnas de anos en la hoja EPS.")
    st.stop()

if "EPS_clean" not in eps_df.columns and "EPS" in eps_df.columns:
    eps_df["EPS_clean"] = eps_df["EPS"].astype(str).str.replace(".xlsx", "", regex=False).str.strip()

if "CUENTA" in eps_df.columns:
    eps_df["CUENTA_norm"] = eps_df["CUENTA"].astype(str).map(normalize_account)

if "EPS_clean" in eps_df.columns:
    eps_df = eps_df[~eps_df["EPS_clean"].map(is_excluded_eps)]
    if valle_eps_keys:
        eps_df["eps_key"] = eps_df["EPS_clean"].map(normalize_eps_name)
        eps_df = eps_df[eps_df["eps_key"].isin(valle_eps_keys)]


eps_list = sorted(eps_df["EPS_clean"].dropna().unique().tolist())

blocks_df = build_blocks_long(eps_df, year_cols)
blocks_df["net_income_factor"] = compute_net_income_factor(blocks_df)
blocks_df["revenue_factor"] = compute_revenue_factor(blocks_df)
ratios_df = compute_ratios(blocks_df)
ratio_cols = list(RATIO_SPECS.keys())
wins_df = winsorize_ratios(ratios_df, ratio_cols, lower=0.02, upper=0.98)
scored_df = score_ratios(wins_df, RATIO_SPECS, dpo_range=(20, 60), dpo_zero=(0, 120))
weights = {
    "liquidity": w_liquidity / 100,
    "solvency": w_solvency / 100,
    "profitability": w_profitability / 100,
    "efficiency": w_efficiency / 100,
}
weight_sum = sum(weights.values())
if weight_sum > 0:
    weights = {k: v / weight_sum for k, v in weights.items()}
scored_df = aggregate_scores(scored_df, RATIO_SPECS, weights)
factor_cols = ["net_income_factor", "revenue_factor"]
if not all(col in scored_df.columns for col in factor_cols):
    scored_df = scored_df.merge(
        blocks_df[["entity", "year"] + factor_cols],
        on=["entity", "year"],
        how="left",
    )
scored_df = apply_size_factors(
    scored_df, weights, factor_cols=["net_income_factor", "revenue_factor"]
)
scored_df["eps_key"] = scored_df["entity"].map(normalize_eps_name)
scored_df = scored_df[~scored_df["eps_key"].map(is_excluded_eps)]


def accounts_used(df: pd.DataFrame, accounts: List[str]) -> str:
    if "CUENTA_norm" not in df.columns:
        return "No disponible"
    account_set = set(df["CUENTA_norm"].dropna())
    used = [acc for acc in accounts if normalize_account(acc) in account_set]
    return " + ".join(used) if used else "No disponible"


def blocks_table(selected: str) -> pd.DataFrame:
    df_eps = eps_df[eps_df["EPS_clean"] == selected]
    rows = []
    for code, accounts, _, _ in BLOCK_DEFS:
        row = {"Bloque": code, "Cuenta usada": accounts_used(df_eps, accounts)}
        subset = blocks_df[(blocks_df["entity"] == selected) & (blocks_df["year"].isin([int(y) for y in year_cols]))]
        for year in year_cols:
            value = subset.loc[subset["year"] == int(year), code]
            row[str(year)] = fmt_currency_millions(value.iloc[0] if not value.empty else None)
        rows.append(row)
    return pd.DataFrame(rows)


def ratio_table(
    selected: str,
    ratio_list: List[str],
    labels: Dict[str, str],
    kinds: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    subset = ratios_df[ratios_df["entity"] == selected]
    rows = []
    flags_rows = []
    delta_cols: List[str] = []
    ordered_cols = ["Indicador"]
    for idx, year in enumerate(year_cols):
        ordered_cols.append(str(year))
        if idx > 0:
            delta_col = f"Δ {year}"
            ordered_cols.append(delta_col)
            delta_cols.append(delta_col)

    for ratio in ratio_list:
        row = {"Indicador": labels.get(ratio, ratio)}
        flags_row: Dict[str, float | None] = {}
        direction = RATIO_SPECS.get(ratio, {}).get("direction", "higher")
        for idx, year in enumerate(year_cols):
            value = subset.loc[subset["year"] == int(year), ratio]
            curr = value.iloc[0] if not value.empty else None
            prev = None
            if idx > 0:
                prev_year = year_cols[idx - 1]
                prev_val = subset.loc[subset["year"] == int(prev_year), ratio]
                prev = prev_val.iloc[0] if not prev_val.empty else None
            base = fmt_ratio(curr, kinds.get(ratio, "ratio"))
            row[str(year)] = base
            if idx > 0:
                delta = compute_delta(curr, prev)
                row[f"Δ {year}"] = delta
                flags_row[f"Δ {year}"] = improvement_flag(curr, prev, direction)
        rows.append(row)
        flags_rows.append(flags_row)

    data_df = pd.DataFrame(rows).reindex(columns=ordered_cols)
    flags_df = pd.DataFrame(flags_rows).reindex(columns=delta_cols)
    return data_df, flags_df, delta_cols


def affiliates_55_table(df: pd.DataFrame) -> pd.DataFrame | None:
    if df.empty:
        return None
    cols = [str(c) for c in df.columns]
    dept_col = find_col(cols, ["departamento"])
    eps_col = find_col(cols, ["eps"])
    age_col = find_col(cols, ["quinquenio", "edad"])
    fem_col = find_col(cols, ["femenino"])
    masc_col = find_col(cols, ["masculino"])
    total_col = find_col(cols, ["total afiliados", "total"])
    if not all([dept_col, eps_col, age_col, fem_col, masc_col, total_col]):
        return None

    work = df[[dept_col, eps_col, age_col, fem_col, masc_col, total_col]].copy()
    work[dept_col] = work[dept_col].astype(str)
    work = work[work[dept_col].str.contains("valle del cauca", case=False, na=False)]
    work["EPS_display"] = work[eps_col].astype(str).str.strip().str.upper()
    work["eps_key"] = work["EPS_display"].map(normalize_eps_name)
    work = work[~work["eps_key"].map(is_excluded_eps)]
    work["age_group"] = work[age_col].map(parse_age_group)
    work = work.dropna(subset=["age_group"])

    def is_55_plus(group: str) -> bool:
        if group == "80+":
            return True
        if "-" in group:
            try:
                start = int(group.split("-")[0])
                return start >= 55
            except ValueError:
                return False
        return False

    work = work[work["age_group"].map(is_55_plus)]
    for col in [fem_col, masc_col, total_col]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    summed = (
        work.groupby("eps_key")[[fem_col, masc_col, total_col]]
        .sum(min_count=1)
        .reset_index()
    )
    display = (
        work.groupby("eps_key")["EPS_display"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
        .reset_index()
    )
    merged = summed.merge(display, on="eps_key", how="left")
    merged = merged.rename(
        columns={
            "EPS_display": "EPS",
            fem_col: "Femenino",
            masc_col: "Masculino",
            total_col: "Total afiliados",
        }
    )
    merged = merged[["EPS", "Femenino", "Masculino", "Total afiliados"]]
    merged = merged.sort_values("Total afiliados", ascending=False)
    return merged


def population_pyramid(df: pd.DataFrame) -> pd.DataFrame | None:
    if df.empty:
        return None
    cols = [str(c) for c in df.columns]
    dept_col = find_col(cols, ["departamento"])
    age_col = find_col(cols, ["quinquenio", "edad"])
    fem_col = find_col(cols, ["femenino"])
    masc_col = find_col(cols, ["masculino"])
    if not all([dept_col, age_col, fem_col, masc_col]):
        return None

    work = df[[dept_col, age_col, fem_col, masc_col]].copy()
    work[dept_col] = work[dept_col].astype(str)
    work = work[work[dept_col].str.contains("valle del cauca", case=False, na=False)]
    work["age_group"] = work[age_col].map(parse_age_group)
    work = work.dropna(subset=["age_group"])
    work[fem_col] = pd.to_numeric(work[fem_col], errors="coerce")
    work[masc_col] = pd.to_numeric(work[masc_col], errors="coerce")
    grouped = (
        work.groupby("age_group")[[fem_col, masc_col]]
        .sum(min_count=1)
        .reset_index()
    )

    def age_order(val: str) -> int:
        if val == "80+":
            return 80
        if "-" in val:
            return int(val.split("-")[0])
        return 0

    grouped["order"] = grouped["age_group"].map(age_order)
    grouped = grouped.sort_values("order", ascending=True)
    grouped = grouped.rename(columns={fem_col: "Femenino", masc_col: "Masculino"})
    return grouped[["age_group", "Femenino", "Masculino"]]


def population_pyramid_by_eps(df: pd.DataFrame, eps_key: str) -> pd.DataFrame | None:
    if df.empty:
        return None
    cols = [str(c) for c in df.columns]
    dept_col = find_col(cols, ["departamento"])
    eps_col = find_col(cols, ["eps"])
    age_col = find_col(cols, ["quinquenio", "edad"])
    fem_col = find_col(cols, ["femenino"])
    masc_col = find_col(cols, ["masculino"])
    if not all([dept_col, eps_col, age_col, fem_col, masc_col]):
        return None

    work = df[[dept_col, eps_col, age_col, fem_col, masc_col]].copy()
    work[dept_col] = work[dept_col].astype(str)
    work = work[work[dept_col].str.contains("valle del cauca", case=False, na=False)]
    work["eps_key"] = work[eps_col].astype(str).map(normalize_eps_name)
    work = work[work["eps_key"] == eps_key]
    if work.empty:
        return None
    work["age_group"] = work[age_col].map(parse_age_group)
    work = work.dropna(subset=["age_group"])
    work[fem_col] = pd.to_numeric(work[fem_col], errors="coerce")
    work[masc_col] = pd.to_numeric(work[masc_col], errors="coerce")
    grouped = (
        work.groupby("age_group")[[fem_col, masc_col]]
        .sum(min_count=1)
        .reset_index()
    )

    def age_order(val: str) -> int:
        if val == "80+":
            return 80
        if "-" in val:
            return int(val.split("-")[0])
        return 0

    grouped["order"] = grouped["age_group"].map(age_order)
    grouped = grouped.sort_values("order", ascending=True)
    grouped = grouped.rename(columns={fem_col: "Femenino", masc_col: "Masculino"})
    return grouped[["age_group", "Femenino", "Masculino"]]

def get_regimen_pivot(
    df: pd.DataFrame, eps_keys: List[str] | None = None
) -> Tuple[pd.DataFrame | None, str | None]:
    if df.empty:
        return None, "No hay datos disponibles en EPS_Afiliados."

    cols = [str(c) for c in df.columns]
    eps_col = find_col(cols, ["eps"])
    mun_col = find_col(cols, ["municipio"])
    if not eps_col or not mun_col:
        return None, "No se encontraron columnas EPS/Municipio en EPS_Afiliados."

    def find_col_contains(tokens: List[str]) -> str | None:
        for col in cols:
            norm = normalize_account(col)
            if all(tok in norm for tok in tokens):
                return col
        return None

    contrib_col = find_col_contains(["afiliados", "contributivo"])
    subs_col = find_col_contains(["afiliados", "subsidiado"])
    esp_col = (
        find_col_contains(["afiliados", "especial"])
        or find_col_contains(["afiliados", "excepcion"])
        or find_col_contains(["afiliados", "excepción"])
    )
    reg_col = find_col(cols, ["regimen", "régimen"])
    aff_col = find_col(cols, ["afiliados", "total"])

    work = df.copy()
    work[mun_col] = work[mun_col].astype(str)
    work = work[work[mun_col].str.contains("cali", case=False, na=False)]
    work["EPS_display"] = work[eps_col].astype(str).str.strip().str.upper()
    work["eps_key"] = work["EPS_display"].map(normalize_eps_name)
    work = work[~work["eps_key"].map(is_excluded_eps)]
    if eps_keys:
        work = work[work["eps_key"].isin(eps_keys)]

    if contrib_col and subs_col:
        cols_keep = [eps_col, "EPS_display", "eps_key", contrib_col, subs_col]
        if esp_col:
            cols_keep.append(esp_col)
        subset = work[cols_keep].copy()
        subset[contrib_col] = pd.to_numeric(subset[contrib_col], errors="coerce")
        subset[subs_col] = pd.to_numeric(subset[subs_col], errors="coerce")
        if esp_col:
            subset[esp_col] = pd.to_numeric(subset[esp_col], errors="coerce")
        grouped = (
            subset.groupby(["eps_key", "EPS_display"])[[contrib_col, subs_col] + ([esp_col] if esp_col else [])]
            .sum(min_count=1)
            .reset_index()
        )
        pivot = grouped.rename(
            columns={
                contrib_col: "Contributivo",
                subs_col: "Subsidiado",
                esp_col: "Especiales/Excepción" if esp_col else esp_col,
            }
        )
    elif reg_col and aff_col:
        subset = work[[eps_col, "EPS_display", "eps_key", reg_col, aff_col]].copy()

        def normalize_regimen(value: object) -> str | None:
            norm = normalize_account(value)
            if "subsidiado" in norm:
                return "Subsidiado"
            if "contributivo" in norm:
                return "Contributivo"
            if "especial" in norm or "excepcion" in norm or "excepción" in norm:
                return "Especiales/Excepción"
            return None

        subset["Regimen"] = subset[reg_col].map(normalize_regimen)
        subset = subset.dropna(subset=["Regimen"])
        subset["afiliados"] = pd.to_numeric(subset[aff_col], errors="coerce")
        subset = subset.dropna(subset=["afiliados"])
        grouped = (
            subset.groupby(["eps_key", "EPS_display", "Regimen"])["afiliados"]
            .sum()
            .reset_index()
        )
        pivot = (
            grouped.pivot_table(
                index=["eps_key", "EPS_display"],
                columns="Regimen",
                values="afiliados",
                aggfunc="sum",
                fill_value=0,
            )
            .reset_index()
        )
    else:
        return None, "No se encontraron columnas necesarias de régimen en EPS_Afiliados."

    for col in ["Subsidiado", "Contributivo", "Especiales/Excepción"]:
        if col not in pivot.columns:
            pivot[col] = 0
    return pivot, None


def score_table(selected: str) -> pd.DataFrame:
    subset = scored_df[scored_df["entity"] == selected]
    rows = []
    mapping = [
        ("Liquidez", "liquidity_score"),
        ("Endeudamiento", "solvency_score"),
        ("Rentabilidad", "profitability_score"),
        ("Eficiencia", "efficiency_score"),
        ("Score Financiero", "score_financiero"),
    ]
    for label, col in mapping:
        row = {"Score": label}
        for year in year_cols:
            value = subset.loc[subset["year"] == int(year), col]
            row[str(year)] = round(value.iloc[0], 1) if not value.empty and pd.notna(value.iloc[0]) else None
        rows.append(row)
    return pd.DataFrame(rows)


ratio_labels = {
    "current_ratio": "Razón corriente",
    "cash_ratio": "Razón de caja",
    "wc_to_rev": "Capital de trabajo neto / REV",
    "days_cash": "Días de caja (proxy)",
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
    "asset_turnover": "Rotación de activos",
    "dso": "DSO (días de cartera)",
    "dpo": "DPO (días de proveedores)",
    "opex_cash_ratio": "Índice OPEX en efectivo",
    "da_intensity": "Intensidad dep/amort",
}

ratio_kinds = {
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


liquidity_ratios = ["current_ratio", "cash_ratio", "wc_to_rev", "days_cash", "current_assets_ratio"]
solvency_ratios = ["debt_to_assets", "equity_ratio", "assets_to_liabilities", "current_liab_share", "net_debt_to_ebitda"]
profitability_ratios = ["gross_margin", "ebitda_margin", "ebit_margin", "net_margin", "roa"]
efficiency_ratios = ["asset_turnover", "dso", "dpo", "opex_cash_ratio", "da_intensity"]


tab_analisis, tab_datos, tab_eps = st.tabs(["Análisis", "Datos", "Análisis EPS"])

with tab_analisis:

    # Prepare FactorEdad + PotencialNormalizado (Cali) for ScoreFinal
    factor_pot = None
    score_final_df = None

    if not age_df.empty and not mun_df.empty:
        age_cols = [str(c) for c in age_df.columns]
        eps_col = find_col(age_cols, ["eps"])
        age_col = find_col(age_cols, ["quinquenio", "edad"])
        total_col = find_col(age_cols, ["total afiliados", "total", "afiliados"])

        mun_cols = [str(c) for c in mun_df.columns]
        mun_eps_col = find_col(mun_cols, ["eps"])
        mun_mun_col = find_col(mun_cols, ["municipio"])
        def find_col_contains(tokens: List[str]) -> str | None:
            for col in mun_cols:
                norm = normalize_account(col)
                if all(tok in norm for tok in tokens):
                    return col
            return None

        mun_aff_col = find_col_contains(["total", "afiliados"]) or find_col(mun_cols, ["afiliados", "total"])

        if eps_col and age_col and total_col and mun_eps_col and mun_mun_col and mun_aff_col:
            age_work = age_df[[eps_col, age_col, total_col]].copy()
            age_work = age_work.dropna(subset=[eps_col, age_col, total_col])
            age_work[eps_col] = age_work[eps_col].astype(str).map(normalize_eps_name)
            age_work["age_group"] = age_work[age_col].map(parse_age_group)
            age_work["total"] = pd.to_numeric(age_work[total_col], errors="coerce")
            age_work = age_work.dropna(subset=["age_group", "total"])

            target_groups = [
                "35-39",
                "40-44",
                "45-49",
                "50-54",
                "55-59",
                "60-64",
                "65-69",
                "70-74",
                "75-79",
                "80+",
            ]
            weights_map = {g: float(np.exp(k_age * i)) for i, g in enumerate(target_groups)}

            totals = age_work.groupby(eps_col)["total"].sum()
            age_sum = age_work.groupby([eps_col, "age_group"])["total"].sum().reset_index()
            age_sum["p_g"] = age_sum.apply(lambda r: r["total"] / totals.get(r[eps_col], np.nan), axis=1)
            age_sum["weight"] = age_sum["age_group"].map(weights_map).fillna(0.0)
            age_sum["weighted"] = age_sum["p_g"] * age_sum["weight"]

            age_index = age_sum.groupby(eps_col)["weighted"].sum().reset_index()
            age_index.columns = ["EPS", "AgeMixIndex"]
            age_index["eps_key"] = age_index["EPS"].map(normalize_eps_name)
            age_index = age_index[~age_index["eps_key"].map(is_excluded_eps)]
            if valle_eps_keys:
                age_index = age_index[age_index["eps_key"].isin(valle_eps_keys)]
            median_index = age_index["AgeMixIndex"].median()
            if median_index == 0 or pd.isna(median_index):
                age_index["FactorEdad"] = 1.0
            else:
                age_index["FactorEdad"] = age_index["AgeMixIndex"] / median_index

            mun_work = mun_df[[mun_eps_col, mun_mun_col, mun_aff_col]].copy()
            mun_work[mun_mun_col] = mun_work[mun_mun_col].astype(str)
            mun_work = mun_work[mun_work[mun_mun_col].str.contains("cali", case=False, na=False)]
            mun_work[mun_eps_col] = mun_work[mun_eps_col].astype(str).map(normalize_eps_name)
            mun_work["afiliados"] = pd.to_numeric(mun_work[mun_aff_col], errors="coerce")
            mun_work = mun_work.dropna(subset=["afiliados"])

            eps_aff = mun_work.groupby(mun_eps_col)["afiliados"].sum().reset_index()
            total_aff = eps_aff["afiliados"].sum()
            eps_aff["market_share"] = eps_aff["afiliados"] / total_aff if total_aff else np.nan

            min_val = eps_aff["market_share"].min()
            max_val = eps_aff["market_share"].max()
            if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
                eps_aff["PotencialNormalizado"] = 1.0
            else:
                eps_aff["PotencialNormalizado"] = 0.5 + (
                    (eps_aff["market_share"] - min_val) / (max_val - min_val)
                )

            eps_aff = eps_aff.rename(columns={mun_eps_col: "EPS"})
            eps_aff["eps_key"] = eps_aff["EPS"].map(normalize_eps_name)
            eps_aff = eps_aff[~eps_aff["eps_key"].map(is_excluded_eps)]
            if valle_eps_keys:
                eps_aff = eps_aff[eps_aff["eps_key"].isin(valle_eps_keys)]

            factor_pot = age_index.merge(
                eps_aff[["EPS", "eps_key", "PotencialNormalizado", "market_share"]],
                on="eps_key",
                how="inner",
                suffixes=("_age", ""),
            )
            factor_pot["EPS"] = factor_pot["EPS"].astype(str).str.upper()

            score_final_df = scored_df.copy()
            score_final_df["eps_key"] = score_final_df["entity"].map(normalize_eps_name)
            score_final_df = score_final_df.merge(
                factor_pot[["eps_key", "EPS", "FactorEdad", "PotencialNormalizado", "market_share"]],
                on="eps_key",
                how="inner",
            )
            score_final_df["ScoreFinal"] = (
                (score_final_df["score_financiero"] / 100)
                * score_final_df["PotencialNormalizado"]
                * score_final_df["FactorEdad"]
            )

    last_year = max(int(y) for y in year_cols)

    section_header("Filtros de Análisis", "Define qué EPS comparar en las gráficas")
    filter_options = ["Todas", "Solo intervenidas", "Solo no intervenidas"]
    if score_final_df is not None and not score_final_df.empty:
        filter_options.extend(["Top 5 (Score Final)", "Top 10 (Score Final)"])
    filter_choice = st.selectbox("EPS en análisis", filter_options, index=0)

    if score_final_df is not None and not score_final_df.empty:
        analysis_base = score_final_df.copy()
        if "eps_key" not in analysis_base.columns:
            analysis_base["eps_key"] = analysis_base["EPS"].map(normalize_eps_name)
        eps_lookup = analysis_base[["EPS", "eps_key"]].drop_duplicates()
    else:
        analysis_base = scored_df.copy()
        analysis_base["eps_key"] = analysis_base["entity"].map(normalize_eps_name)
        eps_lookup = analysis_base[["entity", "eps_key"]].rename(columns={"entity": "EPS"}).drop_duplicates()

    if filter_choice.startswith("Top") and score_final_df is not None and not score_final_df.empty:
        top_n = 5 if "Top 5" in filter_choice else 10
        top_df = score_final_df[score_final_df["year"] == last_year].dropna(subset=["ScoreFinal"])
        analysis_keys = (
            top_df.sort_values("ScoreFinal", ascending=False).head(top_n)["eps_key"].dropna().unique().tolist()
        )
    elif filter_choice == "Solo intervenidas":
        analysis_keys = eps_lookup[eps_lookup["EPS"].map(is_intervened_eps)]["eps_key"].tolist()
    elif filter_choice == "Solo no intervenidas":
        analysis_keys = eps_lookup[~eps_lookup["EPS"].map(is_intervened_eps)]["eps_key"].tolist()
    else:
        analysis_keys = eps_lookup["eps_key"].tolist()

    divider()

    section_header("Evolución de Scores", "EPS según filtro seleccionado")
    st.caption(
        "Colores: tonos cálidos (rojo/naranja) = EPS intervenidas; tonos fríos (azul/verde) = demás EPS."
    )
    col_scores_left, col_scores_right = st.columns(2)

    # Line chart: Top 5 EPS by last year score (Score Financiero)
    scored_plot = scored_df.copy()
    scored_plot["eps_key"] = scored_plot["entity"].map(normalize_eps_name)
    scored_plot = scored_plot[scored_plot["eps_key"].isin(analysis_keys)]
    scored_plot["entity_display"] = scored_plot["entity"].map(lambda x: str(x).upper())
    line_df = scored_plot
    color_map_fin = build_color_map(line_df["entity_display"].dropna().unique().tolist())

    with col_scores_left:
        if not line_df.empty:
            fig = px.line(
                line_df,
                x="year",
                y="score_financiero",
                color="entity_display",
                markers=True,
                title="Score Financiero",
                labels={"year": "Año", "score_financiero": "Score Financiero"},
                color_discrete_map=color_map_fin,
            )
            fig.update_yaxes(range=[0, 100])
            fig = apply_line_chart_style(fig)
            chart_container(fig)
        else:
            st.info("No hay EPS para mostrar con el filtro actual.")

    # Line chart: Top 5 EPS by last year score (Score Final)
    if score_final_df is not None and not score_final_df.empty:
        score_final_plot = score_final_df.copy()
        if "eps_key" not in score_final_plot.columns:
            score_final_plot["eps_key"] = score_final_plot["EPS"].map(normalize_eps_name)
        score_final_plot = score_final_plot[score_final_plot["eps_key"].isin(analysis_keys)]
        score_final_plot["EPS_display"] = score_final_plot["EPS"].map(lambda x: str(x).upper())
        line_final = score_final_plot
        with col_scores_right:
            if not line_final.empty:
                color_map_final = build_color_map(line_final["EPS_display"].dropna().unique().tolist())
                fig = px.line(
                    line_final,
                    x="year",
                    y="ScoreFinal",
                    color="EPS_display",
                    markers=True,
                    title="Score Final",
                    labels={"year": "Año", "ScoreFinal": "Score Final"},
                    color_discrete_map=color_map_final,
                )
                fig = apply_line_chart_style(fig)
                chart_container(fig)
            else:
                st.info("No hay EPS con Score Final para el filtro actual.")

    divider()

    section_header("Ranking por Año", "Ranking (1 = mejor)")
    st.caption(
        "Colores: tonos cálidos (rojo/naranja) = EPS intervenidas; tonos fríos (azul/verde) = demás EPS."
    )
    col_rank_left, col_rank_right = st.columns(2)

    rank_fin = scored_plot.copy()
    rank_fin["rank"] = rank_fin.groupby("year")["score_financiero"].rank(ascending=False, method="dense")
    with col_rank_left:
        if not rank_fin.empty:
            fig = px.line(
                rank_fin,
                x="year",
                y="rank",
                color="entity_display",
                markers=True,
                title="Ranking Score Financiero",
                labels={"year": "Año", "rank": "Ranking (1 = mejor)"},
                color_discrete_map=color_map_fin,
            )
            fig.update_yaxes(autorange="reversed", dtick=1)
            fig = apply_line_chart_style(fig)
            chart_container(fig)
        else:
            st.info("No hay EPS para mostrar con el filtro actual.")

    if score_final_df is not None and not score_final_df.empty:
        rank_final = score_final_plot.copy()
        rank_final["rank"] = rank_final.groupby("year")["ScoreFinal"].rank(ascending=False, method="dense")
        with col_rank_right:
            if not rank_final.empty:
                color_map_final = build_color_map(rank_final["EPS_display"].dropna().unique().tolist())
                fig = px.line(
                    rank_final,
                    x="year",
                    y="rank",
                    color="EPS_display",
                    markers=True,
                    title="Ranking Score Final",
                    labels={"year": "Año", "rank": "Ranking (1 = mejor)"},
                    color_discrete_map=color_map_final,
                )
                fig.update_yaxes(autorange="reversed", dtick=1)
                fig = apply_line_chart_style(fig)
                chart_container(fig)
            else:
                st.info("No hay EPS con Score Final para el filtro actual.")

    divider()
    section_header("Score Financiero vs. % de Mercado", "Último año disponible")
    if score_final_df is None or score_final_df.empty:
        st.info("No hay datos de mercado para el gráfico de dispersión.")
    else:
        scatter_df = score_final_df.copy()
        scatter_df["eps_key"] = scatter_df["EPS"].map(normalize_eps_name)
        scatter_df = scatter_df[scatter_df["eps_key"].isin(analysis_keys)]
        scatter_df = scatter_df[scatter_df["year"] == last_year].dropna(subset=["score_financiero", "market_share"])
        scatter_df["EPS_display"] = scatter_df["EPS"].map(lambda x: str(x).upper())
        if scatter_df.empty:
            st.info("No hay EPS para mostrar con el filtro actual.")
        else:
            color_map_scatter = build_color_map(scatter_df["EPS_display"].dropna().unique().tolist())
            fig = px.scatter(
                scatter_df,
                x="market_share",
                y="score_financiero",
                color="EPS_display",
                size="score_financiero",
                hover_name="EPS_display",
                title="Score Financiero vs % de Mercado (Cali)",
                labels={"market_share": "% de mercado (Cali)", "score_financiero": "Score Financiero"},
                color_discrete_map=color_map_scatter,
            )
            if len(scatter_df) >= 2 and scatter_df["market_share"].nunique() > 1:
                x_vals = scatter_df["market_share"].astype(float).to_numpy()
                y_vals = scatter_df["score_financiero"].astype(float).to_numpy()
                slope, intercept = np.polyfit(x_vals, y_vals, 1)
                x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                y_line = slope * x_line + intercept
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        name="Regresión lineal",
                        line=dict(color="#0a1414", width=2, dash="dash"),
                    )
                )
            fig.update_traces(marker=dict(opacity=0.85, line=dict(width=0.5, color="#0a1414")))
            fig.update_xaxes(tickformat=".1%")
            fig.update_yaxes(range=[0, 100])
            fig = apply_line_chart_style(fig)
            chart_container(fig)

    section_header("FactorEdad y Potencial Normalizado (Cali)")
    st.caption(
        "El Score Final combina tres factores: desempeño financiero (Score Financiero), "
        "potencial de mercado en Cali (Potencial Normalizado) y perfil etario "
        "de los afiliados (FactorEdad)."
    )
    if age_df.empty:
        st.warning("No se pudo leer EPS_Edad.")
        st.caption(f"Detalle: {age_source}")
    elif mun_df.empty:
        st.warning("No se pudo leer EPS_Afiliados.")
        st.caption(f"Detalle: {mun_source}")
    else:
        if factor_pot is None:
            st.warning("No se pudieron calcular FactorEdad y PotencialNormalizado.")
        else:
            merged = factor_pot.copy()
            score_latest = scored_df[scored_df["year"] == last_year][
                ["entity", "score_financiero"]
            ].rename(columns={"entity": "EPS", "score_financiero": "ScoreFinanciero"})
            score_latest["eps_key"] = score_latest["EPS"].map(normalize_eps_name)
            merged = merged.merge(score_latest[["eps_key", "ScoreFinanciero"]], on="eps_key", how="left")
            merged["ScoreFinal"] = (
                (merged["ScoreFinanciero"] / 100)
                * merged["PotencialNormalizado"]
                * merged["FactorEdad"]
            )
            merged = merged.sort_values("ScoreFinal", ascending=False)

            display_df = merged[
                [
                    "EPS",
                    "FactorEdad",
                    "PotencialNormalizado",
                    "ScoreFinanciero",
                    "ScoreFinal",
                    "market_share",
                ]
            ].rename(columns={"market_share": "MarketShareCali"})

            styled = (
                display_df.round(4).style
                .background_gradient(
                    subset=[
                        "FactorEdad",
                        "PotencialNormalizado",
                        "ScoreFinanciero",
                        "ScoreFinal",
                        "MarketShareCali",
                    ],
                    cmap="Blues",
                )
            )
            st.dataframe(styled, use_container_width=True)

            # Ranking EPS x Año usando ScoreFinal (solo EPS con Cali)
            section_header("Ranking EPS x Año (Score Final)")
            if score_final_df is None or score_final_df.empty:
                st.info("No hay datos para el ranking de Score Final.")
            else:
                pivot = score_final_df.pivot_table(index="EPS", columns="year", values="ScoreFinal", aggfunc="mean")
                if last_year in pivot.columns:
                    pivot = pivot.sort_values(by=last_year, ascending=False)
                st.dataframe(pivot.round(4).reset_index(), use_container_width=True)

with tab_datos:
    section_header("Afiliados mayores de 55 años", "EPS_Edad (Femenino, Masculino y Total)")
    if age_df.empty:
        st.warning("No se pudo leer EPS_Edad.")
        st.caption(f"Detalle: {age_source}")
    else:
        affiliates_55 = affiliates_55_table(age_df)
        if affiliates_55 is None or affiliates_55.empty:
            st.warning("No se pudieron calcular afiliados mayores de 55 años.")
        else:
            styled_aff = affiliates_55.style.format(
                {"Femenino": "{:,.0f}", "Masculino": "{:,.0f}", "Total afiliados": "{:,.0f}"}
            )
            st.dataframe(styled_aff, use_container_width=True)

    divider()
    section_header("Composición por Régimen", "EPS por subsidiado vs contributivo (Cali)")
    pivot, msg = get_regimen_pivot(mun_df, list(valle_eps_keys) if valle_eps_keys else None)
    if pivot is None:
        st.info(msg or "No hay datos de régimen para mostrar.")
    else:
        pivot["Total"] = pivot.get("Subsidiado", 0) + pivot.get("Contributivo", 0)
        pivot = pivot.sort_values("Total", ascending=False)
        y_labels = pivot["EPS_display"]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=y_labels,
                x=pivot.get("Subsidiado", 0),
                name="Subsidiado",
                orientation="h",
                marker_color="#e8590c",
            )
        )
        fig.add_trace(
            go.Bar(
                y=y_labels,
                x=pivot.get("Contributivo", 0),
                name="Contributivo",
                orientation="h",
                marker_color="#1c7ed6",
            )
        )
        fig.update_layout(
            barmode="stack",
            title="Afiliados por Régimen (Cali)",
            title_x=0.5,
            title_xanchor="center",
            xaxis_title="Afiliados",
            yaxis_title="EPS",
            margin=dict(l=18, r=18, t=36, b=24),
        )
        fig = style_chart(fig)
        chart_container(fig)

    divider()
    section_header("Pirámide Poblacional", "Valle del Cauca (agregado)")
    if age_df.empty:
        st.warning("No se pudo leer EPS_Edad.")
        st.caption(f"Detalle: {age_source}")
    else:
        pyramid = population_pyramid(age_df)
        if pyramid is None or pyramid.empty:
            st.info("No hay datos para construir la pirámide poblacional.")
        else:
            y_labels = pyramid["age_group"].tolist()
            male_vals = pyramid["Masculino"].tolist()
            female_vals = pyramid["Femenino"].tolist()
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    y=y_labels,
                    x=[-v for v in male_vals],
                    name="Masculino",
                    orientation="h",
                    marker_color="#1c7ed6",
                    text=male_vals,
                    texttemplate="%{text:,.0f}",
                    textposition="inside",
                    insidetextanchor="middle",
                    hovertemplate="%{y}<br>Masculino: %{customdata:,}<extra></extra>",
                    customdata=male_vals,
                )
            )
            fig.add_trace(
                go.Bar(
                    y=y_labels,
                    x=female_vals,
                    name="Femenino",
                    orientation="h",
                    marker_color="#e8590c",
                    text=female_vals,
                    texttemplate="%{text:,.0f}",
                    textposition="inside",
                    insidetextanchor="middle",
                    hovertemplate="%{y}<br>Femenino: %{x:,}<extra></extra>",
                )
            )
            max_val = max(max(male_vals, default=0), max(female_vals, default=0))
            fig.update_layout(
                barmode="relative",
                title="Pirámide Poblacional Valle del Cauca",
                title_x=0.5,
                title_xanchor="center",
                xaxis=dict(
                    title="Afiliados",
                    tickformat="~s",
                    tickvals=[-max_val, -max_val / 2, 0, max_val / 2, max_val],
                    ticktext=[
                        f"{max_val:,.0f}",
                        f"{(max_val/2):,.0f}",
                        "0",
                        f"{(max_val/2):,.0f}",
                        f"{max_val:,.0f}",
                    ],
                ),
                yaxis=dict(title="Grupo de edad", categoryorder="array", categoryarray=y_labels),
                margin=dict(l=18, r=18, t=36, b=24),
            )
            fig = style_chart(fig)
            chart_container(fig)

with tab_eps:
    section_header("Análisis EPS", "Indicadores y subscores de una EPS")
    selected_eps = st.selectbox("EPS", eps_list, index=0, format_func=lambda x: str(x).upper())

    section_header("Subscores y Score Financiero", "EPS seleccionada")
    st.dataframe(score_table(selected_eps), use_container_width=True)

    divider()
    section_header("Piramide Poblacional", "EPS seleccionada (Valle del Cauca)")
    pyramid_eps = population_pyramid_by_eps(age_df, normalize_eps_name(selected_eps))
    if pyramid_eps is None or pyramid_eps.empty:
        st.info("No hay datos para la piramide poblacional de esta EPS.")
    else:
        y_labels = pyramid_eps["age_group"].tolist()
        male_vals = pyramid_eps["Masculino"].tolist()
        female_vals = pyramid_eps["Femenino"].tolist()
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=y_labels,
                x=[-v for v in male_vals],
                name="Masculino",
                orientation="h",
                marker_color="#1c7ed6",
                text=male_vals,
                texttemplate="%{text:,.0f}",
                textposition="inside",
                insidetextanchor="middle",
                hovertemplate="%{y}<br>Masculino: %{customdata:,}<extra></extra>",
                customdata=male_vals,
            )
        )
        fig.add_trace(
            go.Bar(
                y=y_labels,
                x=female_vals,
                name="Femenino",
                orientation="h",
                marker_color="#e8590c",
                text=female_vals,
                texttemplate="%{text:,.0f}",
                textposition="inside",
                insidetextanchor="middle",
                hovertemplate="%{y}<br>Femenino: %{x:,}<extra></extra>",
            )
        )
        max_val = max(max(male_vals, default=0), max(female_vals, default=0))
        fig.update_layout(
            barmode="relative",
            title="Piramide Poblacional EPS",
            title_x=0.5,
            title_xanchor="center",
            xaxis=dict(
                title="Afiliados",
                tickformat="~s",
                tickvals=[-max_val, -max_val / 2, 0, max_val / 2, max_val],
                ticktext=[
                    f"{max_val:,.0f}",
                    f"{(max_val/2):,.0f}",
                    "0",
                    f"{(max_val/2):,.0f}",
                    f"{max_val:,.0f}",
                ],
            ),
            yaxis=dict(title="Grupo de edad", categoryorder="array", categoryarray=y_labels),
            margin=dict(l=18, r=18, t=36, b=24),
        )
        fig = style_chart(fig)
        chart_container(fig)

    # Radar chart: subscores last year
    radar_row = scored_df[(scored_df["entity"] == selected_eps) & (scored_df["year"] == last_year)]
    if not radar_row.empty:
        values = [
            radar_row["liquidity_score"].iloc[0],
            radar_row["solvency_score"].iloc[0],
            radar_row["profitability_score"].iloc[0],
            radar_row["efficiency_score"].iloc[0],
        ]
        if any(pd.isna(v) for v in values):
            st.info("Radar no disponible: faltan subscores para el último año.")
        else:
            categories = ["Liquidez", "Endeudamiento", "Rentabilidad", "Eficiencia"]
            radar_color = WARM_COLORS[0] if is_intervened_eps(selected_eps) else COOL_COLORS[0]
            fig = go.Figure()
            fig.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=str(selected_eps).upper(),
                    line=dict(color=radar_color, width=2),
                )
            )
            fig.update_layout(
                title="Radar de Subscores (último año)",
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                title_x=0.5,
                title_xanchor="center",
                title_font=dict(size=16, family="Newsreader", color="#0a1414"),
                margin=dict(l=18, r=18, t=36, b=24),
            )
            chart_container(fig)

    divider()
    section_header("Régimen EPS", "Distribución Cali")
    eps_key_selected = normalize_eps_name(selected_eps)
    pivot_eps, msg = get_regimen_pivot(mun_df, [eps_key_selected])
    if pivot_eps is None:
        st.info(msg or "No hay datos de régimen para mostrar.")
    else:
        row = pivot_eps[pivot_eps["eps_key"] == eps_key_selected]
        if row.empty:
            st.info("No hay datos de régimen para la EPS seleccionada.")
        else:
            subs = float(row["Subsidiado"].iloc[0]) if "Subsidiado" in row else 0.0
            contrib = float(row["Contributivo"].iloc[0]) if "Contributivo" in row else 0.0
            especiales = (
                float(row["Especiales/Excepción"].iloc[0])
                if "Especiales/Excepción" in row
                else 0.0
            )
            labels = ["Subsidiado", "Contributivo"]
            values = [subs, contrib]
            colors = ["#e8590c", "#1c7ed6"]
            if especiales > 0:
                labels.append("Especiales/Excepción")
                values.append(especiales)
                colors.append("#0f6a62")
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.35,
                        marker=dict(colors=colors, line=dict(color="#fffdf8", width=1)),
                    )
                ]
            )
            fig.update_layout(
                title="Régimen EPS (Cali)",
                title_x=0.5,
                title_xanchor="center",
                margin=dict(l=18, r=18, t=36, b=24),
                showlegend=True,
            )
            fig = style_chart(fig)
            chart_container(fig)

    divider()
    section_header("Bloques financieros", "Cuentas usadas y valores por año")
    st.caption("Valores en millones de COP.")
    st.caption(f"Fuente: {eps_source}")
    st.dataframe(blocks_table(selected_eps), use_container_width=True)

    divider()
    section_header("Indicadores de Liquidez")
    ratio_df, flags_df, delta_cols = ratio_table(selected_eps, liquidity_ratios, ratio_labels, ratio_kinds)
    st.dataframe(style_ratio_table(ratio_df, flags_df, delta_cols), use_container_width=True)

    divider()
    section_header("Indicadores de Endeudamiento / Solvencia")
    ratio_df, flags_df, delta_cols = ratio_table(selected_eps, solvency_ratios, ratio_labels, ratio_kinds)
    st.dataframe(style_ratio_table(ratio_df, flags_df, delta_cols), use_container_width=True)

    divider()
    section_header("Indicadores de Rentabilidad")
    ratio_df, flags_df, delta_cols = ratio_table(selected_eps, profitability_ratios, ratio_labels, ratio_kinds)
    st.dataframe(style_ratio_table(ratio_df, flags_df, delta_cols), use_container_width=True)

    divider()
    section_header("Indicadores de Eficiencia / Actividad")
    ratio_df, flags_df, delta_cols = ratio_table(selected_eps, efficiency_ratios, ratio_labels, ratio_kinds)
    st.dataframe(style_ratio_table(ratio_df, flags_df, delta_cols), use_container_width=True)
