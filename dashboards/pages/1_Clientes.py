from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path
from typing import List

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
from src.models.eps_montecarlo import (
    EPS_OBJ_DEFAULT,
    PROBABILITY_COLUMNS,
    build_composite_ranking,
    build_income_statement_view,
    compute_market_share_valle,
    impute_missing_probabilities,
    run_eps_montecarlo,
    score_risk_percentiles,
)

st.set_page_config(page_title="Clientes", layout="wide")
apply_theme()

page_header(
    "Clientes",
    "Ranking EPS por riesgo Monte Carlo y mercado en Valle del Cauca.",
    "EPS Monte Carlo",
)


def normalize_account(text: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return " ".join(normalized.lower().split())


def normalize_sheet_name(text: object) -> str:
    return re.sub(r"[^a-z0-9]", "", normalize_account(text))


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


def risk_bucket(score: float | None) -> str:
    if score is None or pd.isna(score):
        return "Sin dato"
    if score >= 70:
        return "Bajo riesgo"
    if score >= 50:
        return "Riesgo medio"
    return "Alto riesgo"


def load_cali_sheet(aliases: List[str]) -> tuple[pd.DataFrame, str]:
    path = ROOT_DIR / "data" / "raw" / "Cali ANALISIS.xlsx"
    if not path.exists():
        alt = ROOT_DIR / "Cali ANALISIS.xlsx"
        if alt.exists():
            path = alt
    if not path.exists():
        return pd.DataFrame(), "Archivo no encontrado: Cali ANALISIS.xlsx"

    try:
        xls = pd.ExcelFile(path)
    except Exception as exc:
        return pd.DataFrame(), f"No se pudo abrir Cali ANALISIS.xlsx: {exc}"

    normalized_sheet_map = {normalize_sheet_name(s): s for s in xls.sheet_names}
    selected_sheet = None
    for alias in aliases:
        key = normalize_sheet_name(alias)
        if key in normalized_sheet_map:
            selected_sheet = normalized_sheet_map[key]
            break
    if selected_sheet is None:
        return pd.DataFrame(), f"No se encontro hoja para aliases {aliases}. Disponibles: {xls.sheet_names}"

    try:
        df = pd.read_excel(path, sheet_name=selected_sheet)
        df.columns = [str(c).strip() for c in df.columns]
        return df, f"Excel: {path.name} (hoja {selected_sheet})"
    except Exception as exc:
        return pd.DataFrame(), f"Error leyendo hoja {selected_sheet}: {exc}"


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

    st.caption("Score final fijo: 80% riesgo Monte Carlo + 20% mercado Valle.")

scenarios = {
    "BASE": {"LR_shift": base_lr_shift, "g_shift": base_g_shift},
    "STRESS_LR": {"LR_shift": stress_lr_shift, "g_shift": stress_lr_g_shift},
    "STRESS_MIX": {"LR_shift": stress_mix_lr_shift, "g_shift": stress_mix_g_shift},
}

section_header("Datos fuente", "Modelo EPS")
explain_box(
    "Como se calcula",
    [
        "Se usa Cali ANALISIS.xlsx (hojas: EPS_EEFF, UPC, EPS_Edad, EPS_Anos, EPS_Afiliados).",
        "Las probabilidades se calculan con Monte Carlo en enfoque PROMEDIO.",
        "El % de mercado se calcula sobre todo Valle del Cauca (denominador total Valle).",
    ],
)

eps_eeff_df, eps_eeff_src = load_cali_sheet(["EPS_EEFF", "EPS EEFF"])
upc_df, upc_src = load_cali_sheet(["UPC"])
eps_edad_df, eps_edad_src = load_cali_sheet(["EPS_Edad", "EPS Edad"])
eps_anos_df, eps_anos_src = load_cali_sheet(["EPS_Años", "EPS_Anos", "EPS Anos", "EPS Anos"])
eps_afiliados_df, eps_afiliados_src = load_cali_sheet(["EPS_Afiliados", "EPS Afiliados"])

sources_ok = [
    ("EPS_EEFF", eps_eeff_df, eps_eeff_src),
    ("UPC", upc_df, upc_src),
    ("EPS_Edad", eps_edad_df, eps_edad_src),
    ("EPS_Anos", eps_anos_df, eps_anos_src),
    ("EPS_Afiliados", eps_afiliados_df, eps_afiliados_src),
]
for label, df_src, detail in sources_ok:
    if df_src.empty:
        st.error(f"No se pudo cargar {label}. Detalle: {detail}")
        st.stop()

with st.spinner("Ejecutando simulacion Monte Carlo..."):
    results_df, diagnostics = run_eps_montecarlo(
        upc_df=upc_df,
        eps_eeff_df=eps_eeff_df,
        eps_edad_df=eps_edad_df,
        eps_afiliados_hist_df=eps_anos_df,
        eps_obj=EPS_OBJ_DEFAULT,
        n_sim=n_sim,
        horizon_end=horizon_end,
        cash_thresholds=(15, 0),
        upc_optimism_factor=upc_optimism_factor,
        upc_growth_start_year=upc_growth_start_year,
        upc_growth_end_year=upc_growth_end_year,
        scenarios=scenarios,
        random_seed=42,
    )
    market_share_df = compute_market_share_valle(
        eps_afiliados_df=eps_afiliados_df,
        eps_obj=EPS_OBJ_DEFAULT,
    )
    results_imputed = impute_missing_probabilities(
        results_df=results_df,
        eps_obj=EPS_OBJ_DEFAULT,
        scenarios=list(scenarios.keys()),
        probability_columns=PROBABILITY_COLUMNS,
    )
    results_scored = score_risk_percentiles(
        df=results_imputed,
        probability_columns=PROBABILITY_COLUMNS,
    )
    results_ranked, ranking_escenario, ranking_global = build_composite_ranking(
        scored_df=results_scored,
        market_share_df=market_share_df,
        risk_weight=0.8,
        market_weight=0.2,
    )

ranking_global_exec = ranking_global.copy()
ranking_global_exec["Riesgo"] = ranking_global_exec["Score_Final"].map(risk_bucket)
ranking_global_exec["Imputada"] = ranking_global_exec["prob_imputada"].map({True: "Si", False: "No"})

tab_analisis, tab_datos, tab_eps = st.tabs(["Analisis", "Datos", "Analisis EPS"])

with tab_analisis:
    section_header("Resumen ejecutivo", "Ranking Monte Carlo + mercado Valle")
    scen_choice = st.selectbox("Escenario", list(scenarios.keys()), index=0)

    ranking_exec = ranking_escenario[ranking_escenario["Escenario"] == scen_choice].copy()
    ranking_exec["Riesgo"] = ranking_exec["Score_Final"].map(risk_bucket)
    ranking_exec["Imputada"] = ranking_exec["prob_imputada"].map({True: "Si", False: "No"})

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

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("EPS evaluadas", f"{len(ranking_exec):,}")
    m2.metric("Lider escenario", str(top_name).upper())
    m3.metric("Score lider", f"{top_score:.1f}" if pd.notna(top_score) else "NA")
    m4.metric("Score promedio", f"{mean_score:.1f}" if pd.notna(mean_score) else "NA")
    m5.metric("Mercado Top 3", f"{top3_share:.1%}" if pd.notna(top3_share) else "NA")
    st.caption(f"EPS con imputacion por datos historicos insuficientes: {imputadas_n}")

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
                    "MarketShare_Valle",
                    "Imputada",
                ]
            ].style.format(
                {
                    "Score_Final": "{:.1f}",
                    "Score_Riesgo": "{:.1f}",
                    "Score_Mercado": "{:.1f}",
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
                    "MarketShare_Valle",
                    "Imputada",
                ]
            ].style.format(
                {
                    "Score_Final": "{:.1f}",
                    "Score_Riesgo": "{:.1f}",
                    "MarketShare_Valle": "{:.2%}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    divider()
    section_header("Ranking por escenario", "Score final = 80% riesgo + 20% mercado")
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
                "MarketShare_Valle",
                "Imputada",
            ]
        ].style.format(
            {
                "Score_Final": "{:.1f}",
                "Score_Riesgo": "{:.1f}",
                "Score_Mercado": "{:.1f}",
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
                "MarketShare_Valle",
                "Afiliados_Valle",
                "Imputada",
            ]
        ].style.format(
            {
                "Score_Final": "{:.1f}",
                "Score_Riesgo": "{:.1f}",
                "Score_Mercado": "{:.1f}",
                "MarketShare_Valle": "{:.2%}",
                "Afiliados_Valle": "{:,.0f}",
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
            ],
        )
        prob_cols = ["Escenario", "EPS"] + PROBABILITY_COLUMNS + ["prob_imputada", "motivo_imputacion"]
        prob_view = results_ranked[prob_cols].copy().sort_values(["Escenario", "EPS"])
        prob_view["prob_imputada"] = prob_view["prob_imputada"].map({True: "Si", False: "No"})
        st.dataframe(
            prob_view.style.format({col: "{:.2%}" for col in PROBABILITY_COLUMNS}),
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
        EPS_OBJ_DEFAULT,
        index=0,
        format_func=lambda x: str(x).upper(),
    )

    eps_global = ranking_global_exec[ranking_global_exec["EPS"] == selected_eps].copy()
    rank_global = int(eps_global["Ranking_Global_Final"].iloc[0]) if not eps_global.empty else np.nan
    score_global = float(eps_global["Score_Final"].iloc[0]) if not eps_global.empty else np.nan
    riesgo_global = str(eps_global["Riesgo"].iloc[0]) if not eps_global.empty else "Sin dato"
    share_valle = float(eps_global["MarketShare_Valle"].iloc[0]) if not eps_global.empty else np.nan

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ranking global", f"#{rank_global}" if pd.notna(rank_global) else "NA")
    k2.metric("Score global", f"{score_global:.1f}" if pd.notna(score_global) else "NA")
    k3.metric("Riesgo", riesgo_global)
    k4.metric("% mercado Valle", f"{share_valle:.2%}" if pd.notna(share_valle) else "NA")

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
    section_header("Probabilidades y scoring de la EPS", "Resultados por escenario")
    eps_prob_cols = [
        "Escenario",
        "Ranking_Escenario_Final",
        "EPS",
        "Score_Final",
        "Score_Riesgo",
        "Score_Mercado",
        "MarketShare_Valle",
    ] + PROBABILITY_COLUMNS + ["prob_imputada", "motivo_imputacion"]
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
        eps_fmt = {
            "Score_Final": "{:.1f}",
            "Score_Riesgo": "{:.1f}",
            "Score_Mercado": "{:.1f}",
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
