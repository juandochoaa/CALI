from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


import unicodedata
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboards.data_loader import load_cifras_eps, load_eps_financials
from dashboards.ui import (
    apply_theme,
    append_avg_column,
    append_total_row,
    chart_container,
    divider,
    explain_box,
    page_header,
    section_header,
    style_chart,
)
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


st.set_page_config(page_title="Competencia", layout="wide")
apply_theme()

PAGE_ID = "competencia"
DEFAULT_WEIGHTS = {
    "liquidity": 30,
    "solvency": 30,
    "profitability": 20,
    "efficiency": 20,
}
if st.session_state.get("active_page") != PAGE_ID:
    st.session_state["active_page"] = PAGE_ID
    st.session_state["competencia_weight_liquidity"] = DEFAULT_WEIGHTS["liquidity"]
    st.session_state["competencia_weight_solvency"] = DEFAULT_WEIGHTS["solvency"]
    st.session_state["competencia_weight_profitability"] = DEFAULT_WEIGHTS["profitability"]
    st.session_state["competencia_weight_efficiency"] = DEFAULT_WEIGHTS["efficiency"]

page_header(
    "Competencia",
    "Análisis financiero de IPS en Cali.",
    "IPS insights",
)

with st.sidebar:
    st.header("Filters")
    st.caption("Archivo: Cali ANALISIS.xlsx (hoja IPS EEFF)")
    st.subheader("Pesos del Score Financiero")
    w_liquidity = st.slider(
        "Liquidez (%)",
        0,
        100,
        key="competencia_weight_liquidity",
    )
    w_solvency = st.slider(
        "Endeudamiento (%)",
        0,
        100,
        key="competencia_weight_solvency",
    )
    w_profitability = st.slider(
        "Rentabilidad (%)",
        0,
        100,
        key="competencia_weight_profitability",
    )
    w_efficiency = st.slider(
        "Eficiencia (%)",
        0,
        100,
        key="competencia_weight_efficiency",
    )


def normalize_account(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = " ".join(normalized.lower().split())
    return normalized


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return " ".join(normalized.upper().split())


def accounts_used(df: pd.DataFrame, accounts: List[str]) -> str:
    if "CUENTA_norm" not in df.columns:
        return "No disponible"
    account_set = set(df["CUENTA_norm"].dropna())
    if accounts == ["Ingresos netos por ventas", "Total Ingreso Operativo"]:
        primary = normalize_account("Ingresos netos por ventas")
        secondary = normalize_account("Total Ingreso Operativo")
        if primary in account_set:
            used = ["Ingresos netos por ventas"]
        elif secondary in account_set:
            used = ["Total Ingreso Operativo"]
        else:
            used = []
    else:
        used = [acc for acc in accounts if normalize_account(acc) in account_set]
    return " + ".join(used) if used else "No disponible"


def fmt_currency(value: float | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "NA"
    return f"${value:,.0f}"


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


def split_financial_statements(df_ips: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df_ips.copy()
    work["_row"] = work.index
    if "CUENTA_norm" not in work.columns and "CUENTA" in work.columns:
        work["CUENTA_norm"] = work["CUENTA"].astype(str).map(normalize_account)
    if "CUENTA_norm" not in work.columns:
        return work, work.iloc[0:0]

    balance_keys = [
        "activos totales",
        "activos corrientes",
        "pasivos totales",
        "total de patrimonio",
        "total de patrimonio y pasivos",
    ]
    pattern = "|".join(balance_keys)
    mask_balance = work["CUENTA_norm"].str.contains(pattern, na=False)
    if mask_balance.any():
        split_row = work.loc[mask_balance, "_row"].min()
        estado = work[work["_row"] < split_row]
        balance = work[work["_row"] >= split_row]
    else:
        estado = work
        balance = work.iloc[0:0]

    estado = estado.sort_values("_row")
    balance = balance.sort_values("_row")
    return estado, balance


def total_ingreso_operativo(df: pd.DataFrame, year_cols: List[str]) -> pd.DataFrame:
    if "CUENTA" not in df.columns:
        return pd.DataFrame(columns=["entity", "year", "REV"])
    work = df[["EPS_clean", "CUENTA"] + year_cols].copy()
    work["account_norm"] = work["CUENTA"].map(normalize_account)
    income_accounts = ["ingresos netos por ventas", "total ingreso operativo"]
    work = work[work["account_norm"].isin(income_accounts)]
    if work.empty:
        return pd.DataFrame(columns=["entity", "year", "REV"])
    long_df = work.melt(
        id_vars=["EPS_clean", "account_norm"],
        value_vars=year_cols,
        var_name="year",
        value_name="REV",
    )
    long_df["REV"] = pd.to_numeric(long_df["REV"], errors="coerce")
    long_df["year"] = long_df["year"].astype(str).str.strip().astype(int)
    grouped = (
        long_df.groupby(["EPS_clean", "year", "account_norm"], dropna=False)["REV"]
        .sum(min_count=1)
        .reset_index()
    )
    order_map = {acc: idx for idx, acc in enumerate(income_accounts)}
    grouped["order"] = grouped["account_norm"].map(order_map)
    grouped = (
        grouped.sort_values("order")
        .groupby(["EPS_clean", "year"], dropna=False, as_index=False)
        .first()
    )
    grouped = grouped.rename(columns={"EPS_clean": "entity"})
    return grouped[["entity", "year", "REV"]]


def blocks_table(selected: str) -> pd.DataFrame:
    df_ips = ips_df[ips_df["EPS_clean"] == selected]
    rows = []
    for code, accounts, _, _ in BLOCK_DEFS:
        row = {"Bloque": code, "Cuenta usada": accounts_used(df_ips, accounts)}
        subset = blocks_df[(blocks_df["entity"] == selected) & (blocks_df["year"].isin([int(y) for y in year_cols]))]
        for year in year_cols:
            value = subset.loc[subset["year"] == int(year), code]
            row[str(year)] = fmt_currency(value.iloc[0] if not value.empty else None)
        rows.append(row)
    return pd.DataFrame(rows)


ips_df, ips_source = load_eps_financials("IPS EEFF")
serv_df, serv_source = load_cifras_eps("Serv_IPS")
ci_df, ci_source = load_cifras_eps("CI_IPS")
tarifas_comp_df, tarifas_comp_source = load_cifras_eps("TarifasCompetencia")

if ips_df.empty:
    st.warning("No se encontro el archivo de estados financieros IPS o no se pudo leer.")
    st.caption(f"Detalle: {ips_source}")
    st.stop()

year_cols = [c for c in ips_df.columns if str(c).strip().isdigit()]
year_cols = sorted(year_cols, key=lambda x: int(x))
if not year_cols:
    st.warning("No se encontraron columnas de anos en la hoja IPS EEFF.")
    st.stop()

if "IPS" in ips_df.columns and "EPS_clean" not in ips_df.columns:
    ips_df["EPS_clean"] = ips_df["IPS"].astype(str).str.replace(".xlsx", "", regex=False).str.strip()

if "CUENTA" in ips_df.columns:
    ips_df["CUENTA_norm"] = ips_df["CUENTA"].astype(str).map(normalize_account)

ips_list = sorted(ips_df["EPS_clean"].dropna().unique().tolist())

blocks_df = build_blocks_long(ips_df, year_cols)
ing_oper_df = total_ingreso_operativo(ips_df, year_cols)

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
    df = pd.DataFrame(rows)
    df = append_avg_column(df, [str(y) for y in year_cols], label="Promedio")
    return df


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


def ratio_table(selected: str, ratio_list: List[str]) -> pd.DataFrame:
    import numpy as np

    subset = ratios_df[ratios_df["entity"] == selected]
    rows = []
    for ratio in ratio_list:
        row = {"Indicador": ratio_labels.get(ratio, ratio)}
        numeric_vals: List[float] = []
        for year in year_cols:
            value = subset.loc[subset["year"] == int(year), ratio]
            curr = value.iloc[0] if not value.empty else None
            row[str(year)] = fmt_ratio(curr, ratio_kinds.get(ratio, "ratio"))
            if curr is not None and pd.notna(curr):
                numeric_vals.append(float(curr))
        avg_val = np.mean(numeric_vals) if numeric_vals else np.nan
        row["Promedio"] = fmt_ratio(avg_val, ratio_kinds.get(ratio, "ratio"))
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

tab_analisis, tab_ips, tab_cap, tab_serv, tab_tarifas = st.tabs(
    ["Analisis", "Analisis IPS", "Capacidad instalada", "Servicios", "Tarifas IPS"]
)

with tab_analisis:
    section_header("Clasificación por ingresos", "IPS por ingresos totales (último año)")
    explain_box(
        "Como se calcula",
        [
            "Se usa el ingreso operativo (REV) del último año disponible.",
            "Se clasifican IPS en Alta/Media/Baja según terciles.",
            "El gráfico usa ingresos totales por IPS.",
        ],
    )
    last_year = max(int(y) for y in year_cols)
    rev_last = ing_oper_df[ing_oper_df["year"] == last_year][["entity", "REV"]].dropna()
    if rev_last.empty:
        st.warning("No hay datos de Total Ingreso Operativo para clasificar.")
    else:
        q33 = rev_last["REV"].quantile(0.33)
        q66 = rev_last["REV"].quantile(0.66)

        def segment(rev: float) -> str:
            if rev >= q66:
                return "Alta"
            if rev >= q33:
                return "Media"
            return "Baja"

        rev_last = rev_last.copy()
        rev_last["Segmento"] = rev_last["REV"].map(segment)
        rev_last["IPS"] = rev_last["entity"].map(lambda x: str(x).upper())
        rev_last = rev_last.sort_values("REV", ascending=False)

        fig = px.bar(
            rev_last,
            x="REV",
            y="IPS",
            color="Segmento",
            orientation="h",
            title="Total Ingreso Operativo por IPS (último año)",
            labels={"REV": "Total Ingreso Operativo", "IPS": "IPS"},
            color_discrete_map={"Alta": "#e8590c", "Media": "#1c7ed6", "Baja": "#0f6a62"},
        )
        fig.update_layout(title_x=0.5, title_xanchor="center")
        fig = style_chart(fig)
        chart_container(fig)

        divider()
        section_header("Evolución de ingresos", "Top 5 IPS por ingresos del último año")
        explain_box(
            "Como se calcula",
            [
                "Selecciona las 5 IPS con mayores ingresos del último año.",
                "Se grafica la evolución anual del ingreso operativo.",
            ],
        )
        top_ips = rev_last.head(5)["entity"].tolist()
        rev_line = ing_oper_df[ing_oper_df["entity"].isin(top_ips)][["entity", "year", "REV"]].dropna()
        rev_line["IPS"] = rev_line["entity"].map(lambda x: str(x).upper())
        fig = px.line(
            rev_line,
            x="year",
            y="REV",
            color="IPS",
            markers=True,
            title="Total Ingreso Operativo (Top 5 IPS)",
            labels={"year": "Año", "REV": "Total Ingreso Operativo"},
        )
        fig.update_layout(title_x=0.5, title_xanchor="center")
        fig = style_chart(fig)
        chart_container(fig)

        divider()
        section_header("Ingresos vs Utilidad Neta", "Todas las IPS (ultimo ano)")
        explain_box(
            "Como se calcula",
            [
                "Se cruza el ingreso operativo con la utilidad neta del último año.",
                "Permite comparar tamaño vs rentabilidad absoluta.",
            ],
        )
        net_last = blocks_df[blocks_df["year"] == last_year][["entity", "NET_INCOME"]]
        bar_df = rev_last[["entity", "REV"]].merge(net_last, on="entity", how="left")
        bar_df["IPS"] = bar_df["entity"].map(lambda x: str(x).upper())
        long_df = bar_df.melt(
            id_vars=["IPS"],
            value_vars=["REV", "NET_INCOME"],
            var_name="Metric",
            value_name="Value",
        )
        long_df["Metric"] = long_df["Metric"].map(
            {"REV": "Ingresos", "NET_INCOME": "Utilidad Neta"}
        )
        fig = px.bar(
            long_df,
            x="Value",
            y="IPS",
            color="Metric",
            barmode="group",
            orientation="h",
            title="Ingresos vs Utilidad Neta (ultimo ano)",
            labels={"Value": "COP", "IPS": "IPS"},
        )
        fig.update_layout(title_x=0.5, title_xanchor="center")
        fig = style_chart(fig)
        chart_container(fig)

        section_header("Clasificación IPS", "Segmento por ingresos")
        explain_box(
            "Como se calcula",
            [
                "Tabla con ingresos y utilidad neta del último año.",
                "Incluye fila TOTAL para agregar el mercado IPS.",
            ],
        )
        net_income_last = blocks_df[blocks_df["year"] == last_year][
            ["entity", "NET_INCOME"]
        ].rename(columns={"NET_INCOME": "Utilidad Neta"})
        class_table = rev_last[["entity", "IPS", "Segmento", "REV"]].merge(
            net_income_last, on="entity", how="left"
        )
        class_table = class_table.drop(columns=["entity"]).rename(
            columns={"REV": f"Total Ingreso Operativo {last_year}"}
        )
        class_table = append_total_row(
            class_table,
            "IPS",
            [f"Total Ingreso Operativo {last_year}", "Utilidad Neta"],
        )
        st.dataframe(
            class_table.style.format(
                {
                    f"Total Ingreso Operativo {last_year}": fmt_currency,
                    "Utilidad Neta": fmt_currency,
                }
            ),
            use_container_width=True,
        )

        divider()
        section_header("Score Financiero IPS", "Último año")
        explain_box(
            "Como se calcula",
            [
                "Score basado en subscores de liquidez, solvencia, rentabilidad y eficiencia.",
                "Se aplican pesos configurables en la barra lateral.",
                "Incluye fila PROMEDIO del mercado IPS.",
            ],
        )
        score_last = scored_df[scored_df["year"] == last_year][
            ["entity", "score_financiero"]
        ].dropna()
        if score_last.empty:
            st.info("No hay datos para el score financiero.")
        else:
            score_last = score_last.copy()
            score_last["IPS"] = score_last["entity"].map(lambda x: str(x).upper())
            score_last = score_last.sort_values("score_financiero", ascending=False)
            score_last = score_last[["entity", "IPS", "score_financiero"]].rename(
                columns={"score_financiero": "Score Financiero"}
            )
            avg_row = {
                "entity": "PROMEDIO",
                "IPS": "PROMEDIO",
                "Score Financiero": score_last["Score Financiero"].mean(skipna=True),
            }
            score_last = pd.concat([score_last, pd.DataFrame([avg_row])], ignore_index=True)
            st.dataframe(
                score_last.style.format({"Score Financiero": "{:.1f}"}),
                use_container_width=True,
            )

            divider()
            section_header("Evolución Score Financiero", "Top 5 IPS por score del último año")
            explain_box(
                "Como se calcula",
                [
                    "Top 5 IPS por score del último año.",
                    "Se grafica la evolución anual del Score Financiero.",
                ],
            )
            top_ips_score = score_last.head(5)["entity"].tolist()
            score_line = scored_df[scored_df["entity"].isin(top_ips_score)][
                ["entity", "year", "score_financiero"]
            ].dropna()
            score_line["IPS"] = score_line["entity"].map(lambda x: str(x).upper())
            fig = px.line(
                score_line,
                x="year",
                y="score_financiero",
                color="IPS",
                markers=True,
                title="Score Financiero (Top 5 IPS)",
                labels={"year": "Año", "score_financiero": "Score Financiero"},
            )
            fig.update_layout(title_x=0.5, title_xanchor="center")
            fig = style_chart(fig)
            chart_container(fig)

            divider()
            section_header("Score Financiero histórico", "IPS x Año")
            explain_box(
                "Como se calcula",
                [
                    "Matriz IPS x Año con el Score Financiero.",
                    "Incluye fila PROMEDIO por año.",
                ],
            )
            score_pivot = scored_df.pivot_table(
                index="entity", columns="year", values="score_financiero", aggfunc="mean"
            )
            score_pivot["IPS"] = score_pivot.index.map(lambda x: str(x).upper())
            score_pivot = score_pivot.reset_index(drop=True)
            score_pivot = score_pivot.set_index("IPS")
            if last_year in score_pivot.columns:
                score_pivot = score_pivot.sort_values(last_year, ascending=False)
            avg_row = score_pivot.mean(skipna=True).to_frame().T
            avg_row.index = ["PROMEDIO"]
            score_pivot = pd.concat([score_pivot, avg_row])
            st.dataframe(
                score_pivot.style.format({year: "{:.1f}" for year in score_pivot.columns}),
                use_container_width=True,
            )

    divider()
    section_header("Ubicacion estrategica", "Sector Melendez y Comuna 18")
    explain_box(
        "Como se calcula",
        [
            "Seccion descriptiva con datos demograficos y urbanisticos.",
            "No hay calculos financieros, solo contexto territorial.",
        ],
    )
    st.markdown("""
**4. Ubicacion estrategica: Sector Melendez y Comuna 18**

**4.1 Caracterizacion del Sector Melendez**
- Ubicacion propuesta: Calle 5 con Carrera 95
- Barrio/Sector: Melendez
- Comuna: 18 (sur-occidente de Cali)
""")

    demo_df = pd.DataFrame(
        {
            "Indicador": [
                "Poblacion total",
                "% poblacion Cali",
                "Distribucion sexo",
                "Area",
                "Densidad poblacional",
                "Viviendas",
                "Predios construidos",
            ],
            "Valor": [
                "100,276 habitantes",
                "4.9%",
                "Hombres: 49.2% / Mujeres: 50.8%",
                "542.9 hectareas (4.5% de Cali)",
                "184.7 hab/ha (promedio Cali: 168.7)",
                "24,705 (4.9% del total de Cali)",
                "16,782",
            ],
        }
    )
    st.dataframe(demo_df, use_container_width=True, hide_index=True)

    st.markdown("**Estratificacion Comuna 18**")
    strat_df = pd.DataFrame(
        {
            "Estrato": ["1", "2", "3 (Moda)", "4", "5 y 6", "TOTAL E2+E3"],
            "% Lados de Manzana": ["Minoritario", "~30%", "~43%", "1.2%", "0%", "72.9%"],
            "Observacion": ["Presente", "Significativo", "Predominante", "Marginal", "Ausentes", "Poblacion objetivo"],
        }
    )
    st.dataframe(strat_df, use_container_width=True, hide_index=True)
    st.caption("Hallazgo: 72.9% de viviendas en estratos 2 y 3, alineado con el segmento objetivo.")

    st.markdown("""
**4.2 Proyecto urbanistico Ciudad Melendez**
- Plan parcial aprobado en 2010.
- Area total de planificacion: 152 ha
- Area de desarrollo: 75 ha
- Area de reserva: 54 ha
- Cinturon ecologico: 23 ha
- Ubicacion: entre calles 59 y 61, carreras 93 y 95.

**Ventajas de la ubicacion**
- Desarrollo moderno con infraestructura planificada
- Conexion al transporte publico masivo (MIO)
- Cercania a centros comerciales e instituciones educativas
- Acceso vial directo a Calle 5 (eje principal este-oeste)
""")

    st.markdown("**4.3 Ventaja geografica y estrategica**")
    st.markdown("""
- Acceso zona oriente (comunas 13, 14, 15, 16, 21): 8-10 km, 15-25 min, poblacion potencial 620,000.
- Cobertura zona ladera (comunas 1, 18, 20): 340,000 habitantes.
- Conexion sur y centro-sur (comunas 17, 22, 11, 12): ~200,000 habitantes.
- Mercado total accesible: >1,160,000 habitantes estratos 1-2-3 en 10-15 km.
""")

    comp_df = pd.DataFrame(
        {
            "Institucion": ["ICB Melendez", "Valle del Lili", "Imbanaco", "DIME", "HUV"],
            "Distancia a Oriente": ["8-10 km", "15-18 km", "12-15 km", "12-16 km", "10-12 km"],
            "Tiempo estimado": ["15-25 min", "30-45 min", "25-40 min", "25-40 min", "20-35 min"],
        }
    )
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    st.caption(
        "Ventaja competitiva: el ICB seria el centro especializado mas cercano "
        "a la zona con mayor concentracion de poblacion vulnerable."
    )
with tab_ips:
    section_header("Estados financieros", "IPS individual")
    explain_box(
        "Como se calcula",
        [
            "Vista detallada por IPS seleccionada.",
            "Se separa estado de resultados y balance general según orden del Excel.",
        ],
    )
    selected_ips = st.selectbox("IPS", ips_list, index=0, format_func=lambda x: str(x).upper())

    divider()
    section_header("Analisis cualitativo", "Resumen ejecutivo por IPS")
    explain_box(
        "Como se calcula",
        [
            "Texto cualitativo preparado por el equipo.",
            "Se enfoca en factores financieros, riesgos y contexto operativo.",
        ],
    )

    def norm_key(text: str) -> str:
        return normalize_text(text).lower().strip()

    QUALITATIVE_NOTES = {
        "fundacion valle del lili": """
**1. Situacion financiera general (2024 vs 2023)**
- 2024: **Utilidad** de **$1.056 millones**
- 2023: **Perdida** de **$10.467 millones**
- Cambio: **recuperacion de $11.523 millones (+110%)**

**Hallazgo clave:** la utilidad es **minima** (0,07% sobre ingresos de $1,6 billones). Aunque se salio de perdidas, las utilidades siguen muy bajas.

**2. Principales factores que afectaron utilidades**

**A. Explosion del deterioro de cartera (Factor #1)**
- Deterioro de cartera 2024: **$122.388 millones** (vs $47.857M en 2023) **+156%**
- Provision de glosas 2024: **$1.802 millones** (vs $52M en 2023) **+3.365%**
- Total provisiones 2024: **$124.190 millones** (vs $47.909M) **+159%**

**Por que:** crisis del sector salud; EPS intervenidas con bajo pago (Nueva EPS, SOS, Sanitas, Coosalud, Asmet Salud, Emssanar).
- Cartera vencida: **$622.972M -> $866.550M (+39%)**
- Cartera >360 dias: **$173.058M (20% del total)**
- Texto del documento: incremento en provisiones por deudores debido a disminucion de pagos y aumento de cuentas por cobrar.

**B. Aumento de gastos financieros (Factor #2)**
- Intereses prestamos: **$64.460M** (vs $37.652M) **+71%**
- Gasto financiero neto: **$33.138M** (vs $68.170M) **-51%**
- Deuda total: **$495.206M** (vs $310.794M) **+59%**

**Drivers:** expansion, capital de trabajo por cartera y tasas de interes mas altas.
Nota positiva: menor capitalizacion de intereses en 2024 ($2.244M vs $6.487M en 2023).

**C. Aumento de costos operacionales**
- Costo de ventas: **$1.330.592M** (vs $1.213.666M) **+9,6%**
- Gastos de administracion: **$236.589M** (vs $156.643M) **+51%**
- Drivers: materiales (+12,3%), honorarios medicos (+11,6%), depreciacion (+9,9%), personal (+7,7%).

**3. Expansiones e inversiones (explican caida de utilidades)**
- Inversion en activos fijos 2024: **$90.990M**
- Torre 2 en construccion: **$63.732M** (vs $43.561M)
- Adquisicion nuda propiedad: **$28.507M**
- Usufructo -> propiedad plena: terrenos **+ $61.627M**, edificios **+ $128.944M**
- Torre 8 (anteproyecto): inversion proyectada **$150.000M**
- Expansion Sede Limonar: crecimiento de consultorios y camas
- Equipamiento medico y tecnologico: **$20.068M** aprox.

**4. Red de sedes (6)**
- Sede Principal (Carrera 98 # 18-49)
- Sede Limonar
- Sede Centenario
- Sede Av. Estacion
- Sede Betania
- Sede Alfaguara (Jamundi)

**5. Multas y sanciones**
- No se evidencian multas materiales Supersalud 2023-2024.
- Certificaciones: ICONTEC 2024, Hospital Universitario, JCI (dic 2024).

**6. Politicas contables / cambios**
- Deterioro ampliado 2024: provision general adicional **$14.585M**
- Base: 10% del ingreso promedio mensual presupuestado 2025.
- Efecto: prudente, pero reduce utilidades.

**7. Otras notas**
- Inversiones USD: **$127.700M**; ganancia por devaluacion: **$6.843M**.
- Donaciones: **$4.991M** (vs $1.028M), foco en investigacion.
- Episodios 2022: 1.255.144; ingresos 2024 crecieron 12,14%.
""",
        "dime clinica neurocardiovascular": """
**Informacion general**
- Fundacion: 25 de enero de 1988 (mas de 35 anos de experiencia)
- Ubicacion: Avenida 5 Norte #20-75, Cali, Valle del Cauca
- Empleados: 417 personas (2025)
- Especializacion: clinica de alta complejidad en enfermedades neurocardiovasculares
- Subsidiarias: DIME Cardiovascular S.A. (51%) y DIME Angiografia S.A. (10%)

**Eventos relevantes para analisis financiero**

**1. Acreditacion en salud - ICONTEC (2019-2023)**
- Primera acreditacion: enero 2019 (44 instituciones acreditadas en Colombia).
- Reconocimientos 2019-2023: Medalla Santiago de Cali, Merito Civico, Merito Vallecaucano.
- Renovacion junio 2023: ratificada por ICONTEC.
- Impacto: inversion sostenida en estandarizacion, capacitacion y mejoramiento continuo.

**2. Desarrollo de programas especializados**
- CACI activos desde 2019: insuficiencia cardiaca, trasplante cardiaco, sindrome coronario agudo, ataque cerebrovascular (WSO).
- Trasplante cardiaco: una de 10 instituciones autorizadas en Colombia; una de 3 en Cali.
- Impacto financiero: alto costo operativo, insumos especializados y equipo multidisciplinario.

**3. Tecnologia e inversiones**
- Equipos de alta tecnologia (desde 2007): angiografia avanzada, 3D roadmapping, Expert CT, escaner multicorte 64, resonancia magnetica, mamografia digital, densitometria.
- 2023-2024: sin informacion publica detallada; el sector ha tenido restricciones presupuestarias.

**4. Programas y servicios nuevos**
- PAINT (atencion integral nutricional)
- "Pierde peso, gana vida" (obesidad)
- Programa de riesgo cardiovascular
- Estrategia "Corazon a Corazon" (humanizacion)
- Grupo de apoyo psicologico "GRASPI"
- Plan de beneficios "Dime por ti y para todos"
""",
    }

    key = norm_key(selected_ips)
    note = QUALITATIVE_NOTES.get(key)
    if note is None and ("valle" in key and "lili" in key):
        note = QUALITATIVE_NOTES.get("fundacion valle del lili")
    if note is None and "dime" in key:
        note = QUALITATIVE_NOTES.get("dime clinica neurocardiovascular")
    if note:
        st.markdown(note)
    else:
        st.info("Aun no hay analisis cualitativo para esta IPS.")


    section_header("Bloques financieros", "Cuentas usadas y valores por año")
    explain_box(
        "Como se calcula",
        [
            "Bloques construidos a partir de cuentas contables específicas.",
            "Valores anuales en millones de COP.",
        ],
    )
    st.caption(f"Fuente: {ips_source}")
    st.dataframe(blocks_table(selected_ips), use_container_width=True)

    divider()
    section_header("Estados financieros por cuenta", "IPS seleccionada")
    explain_box(
        "Como se calcula",
        [
            "Se respetan las cuentas y el orden del Excel.",
            "Se muestran valores anuales con formato moneda.",
        ],
    )
    df_ips = ips_df[ips_df["EPS_clean"] == selected_ips]
    if df_ips.empty:
        st.info("No hay datos para la IPS seleccionada.")
    else:
        estado_df, balance_df = split_financial_statements(df_ips)

        cols = ["CUENTA"] + year_cols if "CUENTA" in df_ips.columns else year_cols

        def render_financial_table(title: str, df_view: pd.DataFrame) -> None:
            st.subheader(title)
            if df_view.empty:
                st.info("No hay datos para esta sección.")
                return
            view = df_view[cols].copy()
            for year in year_cols:
                if year in view.columns:
                    view[year] = pd.to_numeric(view[year], errors="coerce")
            st.dataframe(
                view.style.format({year: fmt_currency for year in year_cols}),
                use_container_width=True,
            )

        render_financial_table("Estado de resultados", estado_df)
        render_financial_table("Balance general", balance_df)

    divider()
    section_header("Subscores y Score Financiero", "IPS seleccionada")
    explain_box(
        "Como se calcula",
        [
            "Subscores = promedio de ratios por categoría.",
            "Score Financiero = promedio ponderado con pesos del sidebar.",
            "Incluye columna Promedio por score.",
        ],
    )
    score_df = score_table(selected_ips)
    score_fmt = {str(year): "{:.1f}" for year in year_cols}
    score_fmt["Promedio"] = "{:.1f}"
    st.dataframe(score_df.style.format(score_fmt), use_container_width=True)

    divider()
    section_header("Indicadores de Liquidez")
    explain_box(
        "Como se calcula",
        [
            "Ratios de corto plazo (liquidez).",
            "Se agrega columna Promedio por indicador.",
        ],
    )
    st.dataframe(ratio_table(selected_ips, liquidity_ratios), use_container_width=True)

    divider()
    section_header("Indicadores de Endeudamiento / Solvencia")
    explain_box(
        "Como se calcula",
        [
            "Ratios de apalancamiento y solvencia.",
            "Se agrega columna Promedio por indicador.",
        ],
    )
    st.dataframe(ratio_table(selected_ips, solvency_ratios), use_container_width=True)

    divider()
    section_header("Indicadores de Rentabilidad")
    explain_box(
        "Como se calcula",
        [
            "Ratios de margen y retorno sobre activos.",
            "Se agrega columna Promedio por indicador.",
        ],
    )
    st.dataframe(ratio_table(selected_ips, profitability_ratios), use_container_width=True)

    divider()
    section_header("Indicadores de Eficiencia / Actividad")
    explain_box(
        "Como se calcula",
        [
            "Ratios de rotación y eficiencia de costos.",
            "Se agrega columna Promedio por indicador.",
        ],
    )
    st.dataframe(ratio_table(selected_ips, efficiency_ratios), use_container_width=True)

with tab_cap:
    section_header("Capacidad instalada", "Fuente: CI_IPS")
    explain_box(
        "Como se calcula",
        [
            "Fuente: CI_IPS (capacidad instalada por prestador).",
            "Se agrupa y suma la cantidad por IPS.",
        ],
    )
    if ci_df.empty:
        st.warning("No se pudo leer CI_IPS.")
        st.caption(f"Detalle: {ci_source}")
    else:
        cols = [str(c) for c in ci_df.columns]
        ips_col = next((c for c in cols if "prestador" in c.lower()), None)
        grupo_col = next((c for c in cols if "grupo" in c.lower()), None)
        cantidad_col = next((c for c in cols if "cantidad" in c.lower()), None)

        if not all([ips_col, cantidad_col]):
            st.info("No se encontraron columnas necesarias en CI_IPS.")
        else:
            work = ci_df.copy()
            work["IPS"] = work[ips_col].map(normalize_text)
            work["Cantidad"] = pd.to_numeric(work[cantidad_col], errors="coerce")
            work = work.dropna(subset=["Cantidad"])

            total_cap = work.groupby("IPS")["Cantidad"].sum().reset_index()
            total_cap = total_cap.sort_values("Cantidad", ascending=False)

            section_header("Detalle por IPS", "Distribucion por grupo de capacidad")
            explain_box(
                "Como se calcula",
                [
                    "Distribuye la capacidad por grupos de servicio dentro de la IPS.",
                    "Gráfico de barras horizontal por grupo.",
                ],
            )
            ips_list = total_cap["IPS"].tolist()
            selected_ips = st.selectbox("IPS", ips_list, index=0, key="ips_cap")
            ips_detail = work[work["IPS"] == selected_ips]
            if grupo_col and not ips_detail.empty:
                ips_detail = ips_detail.copy()
                ips_detail["Grupo"] = ips_detail[grupo_col].map(normalize_text)
                grp = ips_detail.groupby("Grupo")["Cantidad"].sum().reset_index()
                fig = px.bar(
                    grp.sort_values("Cantidad", ascending=False),
                    x="Cantidad",
                    y="Grupo",
                    orientation="h",
                    title=f"Capacidad por grupo - {selected_ips}",
                    labels={"Cantidad": "Capacidad", "Grupo": "Grupo"},
                )
                fig.update_layout(title_x=0.5, title_xanchor="center")
                fig = style_chart(fig)
                chart_container(fig)
            else:
                st.info("No hay detalle por grupo de capacidad disponible.")

with tab_serv:
    section_header("Servicios por complejidad", "Fuente: Serv_IPS")
    explain_box(
        "Como se calcula",
        [
            "Fuente: Serv_IPS (procedimientos por complejidad).",
            "Se agrupa por IPS y grupo de servicios.",
        ],
    )
    if serv_df.empty:
        st.warning("No se pudo leer Serv_IPS.")
        st.caption(f"Detalle: {serv_source}")
    else:
        cols = [str(c) for c in serv_df.columns]
        ips_col = next((c for c in cols if "prestador" in c.lower()), None)
        comp_col = next((c for c in cols if "complej" in c.lower()), None)
        group_col = next((c for c in cols if "grse" in c.lower()), None)

        if not all([ips_col, comp_col]):
            st.info("No se encontraron columnas necesarias para servicios en Serv_IPS.")
        else:
            work = serv_df.copy()
            work["IPS"] = work[ips_col].map(normalize_text)
            work["Complejidad"] = work[comp_col].map(normalize_text)
            if group_col:
                work["Grupo"] = work[group_col].map(normalize_text)
                group_options = ["Todos"] + sorted(work["Grupo"].dropna().unique().tolist())
            else:
                group_options = ["Todos"]

            selected_group = st.selectbox("Grupo de servicios", group_options, index=0)
            if selected_group != "Todos" and group_col:
                work = work[work["Grupo"] == selected_group]

            ips_counts = work.groupby("IPS").size().reset_index(name="Procedimientos")
            ips_counts = ips_counts.sort_values("Procedimientos", ascending=False)
            default_ips = ips_counts.head(5)["IPS"].tolist()
            selected_ips = st.multiselect("IPS", ips_counts["IPS"].tolist(), default=default_ips)

            if not selected_ips:
                st.info("Selecciona al menos una IPS.")
            else:
                filtered = work[work["IPS"].isin(selected_ips)]
                grouped = (
                    filtered.groupby(["IPS", "Complejidad"]).size().reset_index(name="Procedimientos")
                )
                fig = px.bar(
                    grouped,
                    x="IPS",
                    y="Procedimientos",
                    color="Complejidad",
                    barmode="group",
                    title="Procedimientos por complejidad (IPS seleccionadas)",
                    labels={"IPS": "IPS", "Procedimientos": "Numero de procedimientos"},
                )
                fig.update_layout(title_x=0.5, title_xanchor="center")
                fig = style_chart(fig)
                chart_container(fig)

with tab_tarifas:
    section_header("Tarifas IPS", "Fuente: TarifasCompetencia")
    explain_box(
        "Como se calcula",
        [
            "Tabla informativa de tarifas por IPS.",
            "No se usa para calculos del modelo.",
        ],
    )
    if tarifas_comp_df.empty:
        st.warning("No se pudo leer la hoja TarifasCompetencia.")
        st.caption(f"Detalle: {tarifas_comp_source}")
    else:
        view = tarifas_comp_df.copy()
        money_cols = [
            c
            for c in view.columns
            if any(token in normalize_text(c) for token in ["tarifa", "precio", "valor", "venta", "costo"])
        ]
        if money_cols:
            st.dataframe(
                view.style.format({col: fmt_currency for col in money_cols}),
                use_container_width=True,
            )
        else:
            st.dataframe(view, use_container_width=True)
