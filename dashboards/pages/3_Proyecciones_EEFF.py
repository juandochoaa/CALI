from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


import re
from datetime import date, datetime
import unicodedata
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboards.data_loader import (
    load_capacidad_dotacion,
    load_cifras_eps,
    load_cifras_eps_raw,
    load_financials,
)
from dashboards.ui import apply_theme, chart_container, divider, page_header, section_header, style_chart
from src.models.financials import npv


st.set_page_config(page_title="Proyecciones (EEFF)", layout="wide")
apply_theme()

page_header(
    "Proyecciones (EEFF)",
    "Ingresos, costos, EBITDA, prevalencia y comparacion.",
    "Financial outlook",
)

with st.sidebar:
    st.header("Parametros")
    discount_rate = st.slider("Tasa de descuento", 0.05, 0.2, 0.12)
    st.caption("Archivo: data/raw/proyecciones.xlsx + Cali ANALISIS.xlsx")


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return " ".join(normalized.lower().split())


def find_col(columns: List[str], includes: List[str]) -> str | None:
    for col in columns:
        norm = normalize_text(col)
        if all(token in norm for token in includes):
            return col
    return None


def parse_prevalencia(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["GrupoEdad", "Prevalencia"])
    header_idx = None
    for idx, row in df.iterrows():
        row_vals = [normalize_text(x) for x in row.tolist()]
        if "grupoedad" in row_vals and "prevalencia" in row_vals:
            header_idx = idx
            break
    if header_idx is None:
        return pd.DataFrame(columns=["GrupoEdad", "Prevalencia"])

    records = []
    for _, row in df.iloc[header_idx + 1 :].iterrows():
        group = row.iloc[0]
        prev = row.iloc[1] if len(row) > 1 else None
        if pd.isna(group) or group == "":
            break
        records.append({"GrupoEdad": str(group).strip(), "Prevalencia": prev})
    out = pd.DataFrame(records)
    out["Prevalencia"] = pd.to_numeric(out["Prevalencia"], errors="coerce")
    if not out["Prevalencia"].dropna().empty and out["Prevalencia"].max() > 1:
        out["Prevalencia"] = out["Prevalencia"] / 100
    return out.dropna(subset=["Prevalencia"])


def map_quinquenio_to_group(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    norm = normalize_text(text)
    if "80" in norm and ("mas" in norm or "mas" in norm):
        return "80+"
    nums = re.findall(r"\d+", norm)
    if len(nums) >= 2:
        start = int(nums[0])
        if start <= 19:
            return "0-19"
        if start <= 39:
            return "20-39"
        if start <= 59:
            return "40-59"
        if start <= 79:
            return "60-79"
    if len(nums) == 1:
        start = int(nums[0])
        if start >= 80:
            return "80+"
    return None


INTERV_PCT_MAP = {
    "mdni": 0.4602,
    "consulta": 1.3599,
    "repro": 0.4607,
    "hemo": 0.0899,
    "electro": 0.0872,
    "cirugia": 0.0915,
}


def compute_comparacion_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"error": "empty"}
    cols = [str(c) for c in df.columns]
    ent_col = find_col(cols, ["entidad"]) or cols[0]
    sant_col = find_col(cols, ["santander", "afiliados"])
    valle_col = find_col(cols, ["valle", "afiliados"])
    icb_col = find_col(cols, ["icb", "atendidos"])
    foscal_col = find_col(cols, ["foscal", "atendidos"])
    if not all([sant_col, valle_col, icb_col, foscal_col]):
        return {"error": "missing_cols", "columns": cols}

    comp_work = df.copy()
    comp_work[ent_col] = comp_work[ent_col].astype(str).str.strip()
    comp_work["ent_norm"] = comp_work[ent_col].map(normalize_text)
    total_row = comp_work[comp_work["ent_norm"] == "total"]
    total_vals = total_row.iloc[0] if not total_row.empty else None

    sant_total = valle_total = atendidos_total = pct_atendido = posibles_valle = np.nan
    if total_vals is not None:
        sant_total = pd.to_numeric(total_vals[sant_col], errors="coerce")
        valle_total = pd.to_numeric(total_vals[valle_col], errors="coerce")
        atendidos_total = pd.to_numeric(total_vals[icb_col], errors="coerce") + pd.to_numeric(
            total_vals[foscal_col], errors="coerce"
        )
        pct_atendido = atendidos_total / sant_total if sant_total and sant_total > 0 else np.nan
        posibles_valle = pct_atendido * valle_total if pd.notna(pct_atendido) else np.nan

    return {
        "comp_work": comp_work,
        "ent_col": ent_col,
        "sant_col": sant_col,
        "valle_col": valle_col,
        "icb_col": icb_col,
        "foscal_col": foscal_col,
        "sant_total": sant_total,
        "valle_total": valle_total,
        "atendidos_total": atendidos_total,
        "pct_atendido": pct_atendido,
        "posibles_valle": posibles_valle,
    }


def is_date_like(value: object) -> bool:
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return True
    try:
        parsed = pd.to_datetime(value, errors="coerce")
        return pd.notna(parsed)
    except Exception:
        return False


def normalize_account(text: str) -> str:
    return normalize_text(text).replace("%", "")


def filter_valid_accounts(df: pd.DataFrame, account_col: str) -> pd.DataFrame:
    work = df.copy()
    work[account_col] = work[account_col].astype(str).str.strip()
    norm = work[account_col].map(normalize_account)
    mask = (
        work[account_col].notna()
        & (work[account_col].astype(str).str.strip() != "")
        & ~norm.str.contains("margen")
        & ~norm.str.contains("%")
        & ~norm.str.contains("ratio")
    )
    return work[mask]


def find_col_any(columns: List[str], candidates: List[List[str]]) -> str | None:
    for includes in candidates:
        col = find_col(columns, includes)
        if col:
            return col
    return None


def map_interv_pct(service: str, interv_map: dict[str, float]) -> float:
    norm = normalize_text(service or "")
    for token, pct in interv_map.items():
        if token in norm:
            return pct
    return np.nan


def map_cap_service_to_tariff(service: str) -> str | None:
    norm = normalize_text(service or "")
    mapping = {
        "hemodinamia": "HEMO",
        "electrofisiologia": "ELECTRO",
        "cirugia": "CIRUGIA",
        "vascular": "CIRUGIA",
        "neurointervencionismo": "HEMO",
    }
    for key, val in mapping.items():
        if key in norm:
            return val
    return None


def tarifas_por_servicio(tarifas_df: pd.DataFrame, ciudad: str) -> pd.DataFrame:
    cols = [str(c) for c in tarifas_df.columns]
    sede_col = find_col_any(cols, [["ciudad"], ["sede"], ["departamento"]])
    servicio_col = find_col_any(cols, [["servicio"]])
    tarifa_col = find_col_any(cols, [["tarifa"], ["precio"], ["valor"]])
    pacientes_col = find_col_any(cols, [["pacientes"], ["paciente"]])

    if not all([servicio_col, tarifa_col, pacientes_col]):
        return pd.DataFrame()

    work = tarifas_df.copy()
    work[servicio_col] = work[servicio_col].astype(str)
    if sede_col:
        work[sede_col] = work[sede_col].astype(str)
        work = work[
            work[sede_col].map(normalize_text).str.contains(normalize_text(ciudad), na=False)
        ]
    work[tarifa_col] = pd.to_numeric(work[tarifa_col], errors="coerce")
    work[pacientes_col] = pd.to_numeric(work[pacientes_col], errors="coerce")

    def weighted_tarifa(group: pd.DataFrame) -> float:
        pac = group[pacientes_col].sum()
        if pac and pac > 0:
            return (group[tarifa_col] * group[pacientes_col]).sum() / pac
        return group[tarifa_col].mean()

    grouped = (
        work.groupby(servicio_col, dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "TarifaPromedio": weighted_tarifa(g),
                    "Pacientes": g[pacientes_col].sum(),
                }
            )
        )
        .reset_index()
        .rename(columns={servicio_col: "ServicioTarifa"})
    )
    grouped["ServicioTarifa"] = grouped["ServicioTarifa"].astype(str).str.upper().str.strip()
    total_pac = grouped["Pacientes"].sum()
    grouped["PctPacientes"] = grouped["Pacientes"] / total_pac if total_pac else np.nan
    grouped["PctIntervenciones"] = grouped["ServicioTarifa"].map(
        lambda x: map_interv_pct(x, INTERV_PCT_MAP)
    )
    return grouped


def build_tarifa_table(
    df: pd.DataFrame,
    sede_col: str,
    target_sede: str,
    servicio_col: str,
    procedimiento_col: str | None,
    tarifa_col: str,
    pacientes_col: str,
    posibles_valle: float | None,
    inter_col: str | None = None,
    interv_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    work = df.copy()
    work[sede_col] = work[sede_col].astype(str)
    work["sede_norm"] = work[sede_col].map(normalize_text)
    work = work[work["sede_norm"].str.contains(normalize_text(target_sede), na=False)]
    if work.empty:
        return pd.DataFrame()

    work[pacientes_col] = pd.to_numeric(work[pacientes_col], errors="coerce")
    work[tarifa_col] = pd.to_numeric(work[tarifa_col], errors="coerce")
    if inter_col:
        work[inter_col] = pd.to_numeric(work[inter_col], errors="coerce")

    total_pacientes = work[pacientes_col].sum()

    if interv_map:
        work["interv_pct_map"] = work[servicio_col].map(lambda x: map_interv_pct(x, interv_map))
        service_totals = work.groupby(servicio_col)[pacientes_col].sum().rename("service_total_pac")
        work = work.merge(service_totals, left_on=servicio_col, right_index=True, how="left")
        work["pct_interv_row"] = work["interv_pct_map"] * (
            work[pacientes_col] / work["service_total_pac"]
        )
    elif inter_col:
        total_interv = work[inter_col].sum()
        work["pct_interv_row"] = (
            work[inter_col] / total_interv if total_interv and total_interv > 0 else np.nan
        )
    else:
        work["pct_interv_row"] = np.nan

    group_col = procedimiento_col or servicio_col

    def weighted_tarifa(group: pd.DataFrame) -> float:
        pac = group[pacientes_col].sum()
        if pac and pac > 0:
            return (group[tarifa_col] * group[pacientes_col]).sum() / pac
        return group[tarifa_col].mean()

    grouped = (
        work.groupby(group_col, dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "TarifaPromedio": weighted_tarifa(g),
                    "Pacientes": g[pacientes_col].sum(),
                    "PctIntervenciones": g["pct_interv_row"].sum(min_count=1),
                }
            )
        )
        .reset_index()
        .rename(columns={group_col: "Servicio" if procedimiento_col is None else "Procedimiento"})
    )

    grouped["PctPacientes"] = (
        grouped["Pacientes"] / total_pacientes if total_pacientes and total_pacientes > 0 else np.nan
    )
    grouped["PctIntervenciones"] = grouped["PctIntervenciones"].fillna(grouped["PctPacientes"])

    if posibles_valle is not None and pd.notna(posibles_valle):
        grouped["PacientesValle"] = grouped["PctPacientes"] * posibles_valle
        grouped["Intervenciones"] = grouped["PacientesValle"] * (1 + grouped["PctIntervenciones"])
        grouped["Ventas"] = grouped["Intervenciones"] * grouped["TarifaPromedio"]
    else:
        grouped["PacientesValle"] = np.nan
        grouped["Intervenciones"] = np.nan
        grouped["Ventas"] = np.nan

    total_row = {
        ("Servicio" if procedimiento_col is None else "Procedimiento"): "TOTAL",
        "TarifaPromedio": weighted_tarifa(work),
        "Pacientes": grouped["Pacientes"].sum(min_count=1),
        "PctPacientes": grouped["PctPacientes"].sum(min_count=1),
        "PctIntervenciones": grouped["PctIntervenciones"].sum(min_count=1),
        "PacientesValle": grouped["PacientesValle"].sum(min_count=1),
        "Intervenciones": grouped["Intervenciones"].sum(min_count=1),
        "Ventas": grouped["Ventas"].sum(min_count=1),
    }
    grouped = pd.concat([grouped, pd.DataFrame([total_row])], ignore_index=True)

    return grouped


def fmt_currency(val) -> str:
    if pd.isna(val):
        return ""
    return f"${val:,.0f}"


def fmt_percent(val, digits: int = 1) -> str:
    if pd.isna(val):
        return ""
    return f"{val:.{digits}%}"


def format_currency_df(df: pd.DataFrame, columns: list[str]) -> pd.io.formats.style.Styler:
    def fmt(val):
        return fmt_currency(val)

    return df.style.format({col: fmt for col in columns if col in df.columns})


def pick_ingresos_series(annual_pivot: pd.DataFrame) -> pd.Series | None:
    targets = [
        "ingresos netos por ventas",
        "total ingreso operativo",
        "ingresos",
    ]
    norm_index = annual_pivot.index.map(normalize_account)
    for target in targets:
        mask = norm_index == target
        if mask.any():
            return annual_pivot.loc[mask].iloc[0].sort_index()
    return None


def weighted_growth(yoy: pd.Series) -> float:
    valid = yoy.dropna()
    if valid.empty:
        return float("nan")
    values = valid.to_numpy()
    weights = np.arange(1, len(values) + 1, dtype=float)
    return float(np.average(values, weights=weights))


proj, proj_source = load_financials()
prev_raw, prev_source = load_cifras_eps_raw("Prevalencia", header=None)
comp_df, comp_source = load_cifras_eps("Comparacion")
edad_df, edad_source = load_cifras_eps("EPS_Edad")
tarifas_df, tarifas_source = load_cifras_eps("Tarifas")
sant_df, sant_source = load_cifras_eps("SANTANDER")

comp_metrics = compute_comparacion_metrics(comp_df)

tab_eeff, tab_prev, tab_comp, tab_tar, tab_sant = st.tabs(
    ["EEFF", "Prevalencia", "Comparacion", "Tarifas", "Santander"]
)

with tab_eeff:
    st.selectbox(
        "Escenario de precios",
        ["Santander", "Bogota", "Competencia Cali (pendiente)"],
        index=0,
        key="precio_scenario",
    )
    objetivo_valle = comp_metrics.get("posibles_valle")

    # Santander ratios + growth (para EEFF y proyecciones)
    ratio_df = None
    growth_avg = np.nan
    if not sant_df.empty:
        account_col = sant_df.columns[0]
        month_cols = [c for c in sant_df.columns[1:] if is_date_like(c)]
        if month_cols:
            sant_work = filter_valid_accounts(sant_df, account_col)
            long_df = sant_work[[account_col] + month_cols].melt(
                id_vars=account_col, var_name="Month", value_name="Valor"
            )
            long_df["Valor"] = pd.to_numeric(long_df["Valor"], errors="coerce")
            long_df["Year"] = pd.to_datetime(long_df["Month"], errors="coerce").dt.year
            long_df = long_df.dropna(subset=["Year"])

            annual_df = (
                long_df.groupby([account_col, "Year"], dropna=False)["Valor"]
                .sum(min_count=1)
                .reset_index()
                .rename(columns={account_col: "Cuenta"})
            )
            annual_pivot = annual_df.pivot_table(
                index="Cuenta", columns="Year", values="Valor", aggfunc="sum"
            ).sort_index()
            ingresos_series = pick_ingresos_series(annual_pivot)
            if ingresos_series is not None:
                yoy = ingresos_series / ingresos_series.shift(1) - 1
                growth_avg = weighted_growth(yoy)
                ratio_df = annual_pivot.div(ingresos_series, axis=1).replace(
                    [np.inf, -np.inf], np.nan
                )
                ratio_df["Promedio"] = ratio_df.mean(axis=1, skipna=True)

    if proj.empty:
        pass
    else:
        required_cols = {
            "year",
            "revenue_cop_bn",
            "costs_cop_bn",
            "ebitda_cop_bn",
            "cashflow_cop_bn",
            "capex_cop_bn",
        }
        if not required_cols.issubset(set(proj.columns)):
            st.warning("La hoja de proyecciones no tiene las columnas requeridas.")
        else:
            cashflows = proj["cashflow_cop_bn"].to_numpy()
            npv_value = npv(cashflows, discount_rate)

            section_header("KPIs financieros")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("NPV (COP bn)", f"{npv_value:.1f}")
            col2.metric("EBITDA 2030", f"{proj.loc[proj['year'] == 2030, 'ebitda_cop_bn'].iat[0]:.1f} bn")
            col3.metric("Margen EBITDA", f"{proj['ebitda_cop_bn'].iloc[-1] / proj['revenue_cop_bn'].iloc[-1]:.0%}")
            col4.metric("Capex total", f"{proj['capex_cop_bn'].sum():.1f} bn")

            left, right = st.columns([1.1, 0.9])

            with left:
                plot_df = proj.melt(
                    id_vars="year",
                    value_vars=["revenue_cop_bn", "costs_cop_bn", "ebitda_cop_bn"],
                    var_name="metric",
                    value_name="value",
                )
                fig = px.line(
                    plot_df,
                    x="year",
                    y="value",
                    color="metric",
                    markers=True,
                    title="Ingresos, costos y EBITDA (COP bn)",
                )
                fig = style_chart(fig)
                chart_container(fig)
                st.caption(f"Fuente: {proj_source}")

            with right:
                fig = px.bar(
                    proj,
                    x="year",
                    y="cashflow_cop_bn",
                    title="Flujo de caja (COP bn)",
                    text="cashflow_cop_bn",
                )
                fig = style_chart(fig)
                chart_container(fig)
                st.caption(f"Fuente: {proj_source}")

            section_header("Detalle anual")
            money_cols = [
                "revenue_cop_bn",
                "costs_cop_bn",
                "ebitda_cop_bn",
                "cashflow_cop_bn",
                "capex_cop_bn",
            ]
            st.dataframe(
                proj.style.format({col: (lambda v: "" if pd.isna(v) else f"${v:,.1f}") for col in money_cols}),
                use_container_width=True,
            )

    divider()
    section_header("Ventas Año 1 por capacidad", "Capacidad Cali + productividad Bogotá")
    cap_df = load_capacidad_dotacion()
    if cap_df.empty:
        st.warning("No hay tabla de capacidad instalada.")
    else:
        objetivo_valle = comp_metrics.get("posibles_valle")
        if pd.isna(objetivo_valle):
            st.warning("No se pudo calcular el objetivo Valle (posibles atendidos).")
        else:
            scenario = st.session_state.get("precio_scenario", "Santander")
            scenario_norm = normalize_text(scenario)
            ciudad = "Bogota" if "bogota" in scenario_norm else "Santander"
            tarifas_serv = tarifas_por_servicio(tarifas_df, ciudad)
            tarifas_sant = tarifas_por_servicio(tarifas_df, "Santander")
            tarifas_bog = tarifas_por_servicio(tarifas_df, "Bogota")
            if tarifas_serv.empty:
                tarifas_serv = tarifas_sant if not tarifas_sant.empty else tarifas_bog
            st.caption(f"Tarifas usadas: {ciudad}.")

            cap_work = cap_df.copy()
            cap_work["ServicioTarifa"] = cap_work["Servicio"].map(map_cap_service_to_tariff)
            cap_work["Capacidad_mensual"] = pd.to_numeric(
                cap_work.get("Capacidad_mensual"), errors="coerce"
            )
            if cap_work["Capacidad_mensual"].isna().all():
                cap_work["Capacidad_mensual"] = (
                    cap_work["Pacientes_por_dia"] * cap_work["Dias_semana"] * 4
                )
            cap_work["Cantidad_equipo"] = pd.to_numeric(
                cap_work.get("Cantidad_equipo"), errors="coerce"
            )
            cap_work["Capacidad_unitaria"] = cap_work["Capacidad_mensual"] / cap_work["Cantidad_equipo"]

            unit_map = (
                cap_work.groupby("ServicioTarifa")["Capacidad_unitaria"]
                .mean()
                .to_dict()
            )

            cap_cali_angio = 2
            cap_hemo = unit_map.get("HEMO", np.nan) * cap_cali_angio
            pct_base = (cap_hemo * 12) / objetivo_valle if pd.notna(cap_hemo) else np.nan

            pct_prom = None
            if not tarifas_sant.empty or not tarifas_bog.empty:
                pct_prom = pd.merge(
                    tarifas_sant[["ServicioTarifa", "PctPacientes"]].rename(
                        columns={"PctPacientes": "PctPac_Sant"}
                    ),
                    tarifas_bog[["ServicioTarifa", "PctPacientes"]].rename(
                        columns={"PctPacientes": "PctPac_Bog"}
                    ),
                    on="ServicioTarifa",
                    how="outer",
                )
                pct_prom["PctPacProm"] = pct_prom[["PctPac_Sant", "PctPac_Bog"]].mean(
                    axis=1, skipna=True
                )
                total_prom = pct_prom["PctPacProm"].sum()
                if total_prom and total_prom > 0:
                    pct_prom["PctPacProm"] = pct_prom["PctPacProm"] / total_prom

            col1, col2, col3 = st.columns(3)
            col1.metric("Objetivo Valle (pacientes)", f"{objetivo_valle:,.0f}")
            col2.metric(
                "% capacidad base (Hemo)",
                f"{pct_base:.1%}" if pd.notna(pct_base) else "NA",
            )
            col3.metric(
                "Capacidad anual Hemo",
                f"{cap_hemo * 12:,.0f}" if pd.notna(cap_hemo) else "NA",
            )

            st.caption(
                "Base: capacidad instalada de angiografos para HEMO. "
                "Los demas servicios se escalan con el mix promedio Santander/Bogota."
            )

            if tarifas_serv.empty:
                st.warning("No se encontraron tarifas para el escenario seleccionado.")
            else:
                tariffs = tarifas_serv.copy()
                if pct_prom is not None:
                    tariffs = tariffs.merge(
                        pct_prom[["ServicioTarifa", "PctPacProm"]], on="ServicioTarifa", how="left"
                    )
                else:
                    tariffs["PctPacProm"] = np.nan

                if tariffs["PctPacProm"].isna().all():
                    st.warning("No se pudo calcular % promedio de pacientes; se reparte en partes iguales.")
                    tariffs["PctPacProm"] = 1 / len(tariffs) if len(tariffs) else np.nan
                else:
                    tariffs["PctPacProm"] = tariffs["PctPacProm"].fillna(0)
                    total_prom = tariffs["PctPacProm"].sum()
                    if total_prom and total_prom > 0:
                        tariffs["PctPacProm"] = tariffs["PctPacProm"] / total_prom

                pct_pac_base = tariffs.loc[
                    tariffs["ServicioTarifa"] == "HEMO", "PctPacProm"
                ].dropna()
                pct_pac_base = pct_pac_base.iloc[0] if not pct_pac_base.empty else np.nan

                if pd.isna(pct_pac_base):
                    st.warning("No se encontro PctPacProm para HEMO; se usa promedio general.")
                    pct_pac_base = tariffs["PctPacProm"].dropna().mean()

                def pct_for_service(pct_pac: float | None) -> float | None:
                    if pd.notna(pct_base) and pd.notna(pct_pac_base) and pd.notna(pct_pac):
                        return pct_base * (pct_pac / pct_pac_base)
                    return pct_base

                tariffs["PctCapacidad"] = tariffs.apply(
                    lambda row: pct_for_service(row.get("PctPacProm")),
                    axis=1,
                )

                tariffs["Pacientes_Ano1"] = objetivo_valle * tariffs["PctCapacidad"]
                tariffs["Intervenciones_Ano1"] = tariffs.apply(
                    lambda row: row["Pacientes_Ano1"] * (1 + row["PctIntervenciones"])
                    if pd.notna(row.get("PctIntervenciones"))
                    else row["Pacientes_Ano1"],
                    axis=1,
                )
                tariffs["Ventas_Ano1"] = tariffs["Intervenciones_Ano1"] * tariffs["TarifaPromedio"]

                show_cols = [
                    "ServicioTarifa",
                    "PctCapacidad",
                    "Pacientes_Ano1",
                    "TarifaPromedio",
                    "PctIntervenciones",
                    "Ventas_Ano1",
                ]
                show_df = tariffs[show_cols].rename(columns={"ServicioTarifa": "Servicio"})

                total_row = {
                    "Servicio": "TOTAL",
                    "PctCapacidad": np.nan,
                    "Pacientes_Ano1": show_df["Pacientes_Ano1"].sum(min_count=1),
                    "TarifaPromedio": np.nan,
                    "PctIntervenciones": np.nan,
                    "Ventas_Ano1": show_df["Ventas_Ano1"].sum(min_count=1),
                }
                show_df = pd.concat([show_df, pd.DataFrame([total_row])], ignore_index=True)

                styled = show_df.style.format(
                    {
                        "PctCapacidad": lambda v: "" if pd.isna(v) else f"{v:.1%}",
                        "Pacientes_Ano1": "{:,.1f}",
                        "TarifaPromedio": lambda v: "" if pd.isna(v) else f"${v:,.0f}",
                        "PctIntervenciones": lambda v: "" if pd.isna(v) else f"{v:.1%}",
                        "Ventas_Ano1": lambda v: "" if pd.isna(v) else f"${v:,.0f}",
                    }
                )
                st.dataframe(styled, use_container_width=True)

                divider()
                section_header("Proyeccion anual por servicio", "Estandarizacion de mix desde Año 2")
                total_year1 = show_df.loc[show_df["Servicio"] != "TOTAL", "Pacientes_Ano1"].sum()

                growth_default = growth_avg
                if pd.isna(growth_default):
                    st.warning("No se pudo calcular crecimiento historico; se asume 0%.")
                    growth_default = 0.0

                growth_required = None
                if (
                    pd.notna(objetivo_valle)
                    and pd.notna(total_year1)
                    and total_year1 > 0
                    and objetivo_valle > 0
                ):
                    steps_to_target = len(range(2026, 2031)) - 1
                    if steps_to_target > 0:
                        growth_required = (objetivo_valle / total_year1) ** (1 / steps_to_target) - 1
                else:
                    st.info(
                        "No se puede calcular la tasa objetivo: faltan datos de pacientes A??o 1 u objetivo Valle."
                    )

                if "pending_growth_rate" in st.session_state:
                    st.session_state["growth_rate"] = st.session_state.pop("pending_growth_rate")
                if "growth_rate" not in st.session_state:
                    st.session_state["growth_rate"] = float(growth_default)
                min_growth = -0.2
                max_growth = 0.5
                if growth_required is not None and pd.notna(growth_required):
                    max_growth = max(max_growth, float(growth_required))
                    min_growth = min(min_growth, float(growth_required))
                st.session_state["growth_rate"] = max(
                    min_growth, min(max_growth, float(st.session_state["growth_rate"]))
                )
                growth_rate = st.number_input(
                    "Tasa de crecimiento anual (predeterminada)",
                    min_value=min_growth,
                    max_value=max_growth,
                    value=float(st.session_state["growth_rate"]),
                    step=0.01,
                    format="%.2f",
                    key="growth_rate",
                )
                st.caption(f"Tasa predeterminada aplicada: {growth_default:.2%}.")
                st.caption(
                    f"Crecimiento historico calculado: {growth_default:.2%}. "
                    "Puedes ajustar la tasa para las proyecciones."
                )
                if growth_required is not None and pd.notna(growth_required):
                    st.caption(
                        f"Tasa requerida para llegar al objetivo de pacientes en el a??o 5: {growth_required:.2%}."
                    )
                    if st.button("Ajustar crecimiento para llegar al objetivo en a??o 5"):
                        st.session_state["pending_growth_rate"] = float(growth_required)
                        st.rerun()

                proj_years = list(range(2026, 2031))
                service_rows = []
                for year in proj_years:
                    if year == 2026:
                        patients_by_service = tariffs["Pacientes_Ano1"]
                    else:
                        total_year = total_year1 * (1 + growth_rate) ** (year - 2026)
                        patients_by_service = total_year * tariffs["PctPacProm"]
                    interventions = patients_by_service * (1 + tariffs["PctIntervenciones"].fillna(0))
                    sales = interventions * tariffs["TarifaPromedio"]
                    for idx, svc in enumerate(tariffs["ServicioTarifa"]):
                        service_rows.append(
                            {
                                "Servicio": svc,
                                "Año": year,
                                "Pacientes": patients_by_service.iloc[idx],
                                "Intervenciones": interventions.iloc[idx],
                                "Ventas": sales.iloc[idx],
                            }
                        )
                service_proj = pd.DataFrame(service_rows)

                # tabla completa por servicio (wide)
                def pivot_metric(metric: str) -> pd.DataFrame:
                    pivot = service_proj.pivot_table(
                        index="Servicio", columns="Año", values=metric, aggfunc="sum"
                    )
                    pivot.columns = [f"{col} {metric}" for col in pivot.columns]
                    return pivot

                full_table = pivot_metric("Pacientes").join(
                    pivot_metric("Intervenciones")
                ).join(pivot_metric("Ventas"))
                full_table = full_table.reset_index()
                st.dataframe(
                    full_table.style.format(
                        {
                            **{c: "{:,.0f}" for c in full_table.columns if "Pacientes" in c},
                            **{c: "{:,.0f}" for c in full_table.columns if "Intervenciones" in c},
                            **{
                                c: (lambda v: "" if pd.isna(v) else f"${v:,.0f}")
                                for c in full_table.columns
                                if "Ventas" in c
                            },
                        }
                    ),
                    use_container_width=True,
                )

                # totales por año
                proj_totals = (
                    service_proj.groupby("Año")[["Pacientes", "Ventas"]]
                    .sum()
                    .reset_index()
                )
                st.dataframe(
                    proj_totals.style.format(
                        {
                            "Pacientes": "{:,.0f}",
                            "Ventas": lambda v: "" if pd.isna(v) else f"${v:,.0f}",
                        }
                    ),
                    use_container_width=True,
                )

                divider()
                section_header("Estructura de proyeccion anual (2026-2030)")
                if ratio_df is None:
                    st.warning("No hay ratios de Santander para proyectar EEFF.")
                else:
                    st.caption("Proyeccion EEFF basada en tarifas por servicio + ratios Santander.")
                    revenue_by_year = proj_totals.set_index("Año")["Ventas"]
                    proj_table = pd.DataFrame(index=ratio_df.index)
                    for year in proj_years:
                        proj_table[year] = ratio_df["Promedio"].values * revenue_by_year.get(
                            year, np.nan
                        )
                    proj_table = proj_table.reset_index().rename(columns={"index": "Cuenta"})
                    year_cols = [c for c in proj_table.columns if isinstance(c, int)]
                    st.dataframe(
                        proj_table.style.format(
                            {col: (lambda v: "" if pd.isna(v) else f"${v:,.0f}") for col in year_cols}
                        ),
                        use_container_width=True,
                    )

with tab_prev:
    section_header("Prevalencia por rango de edad", "Fuente: Prevalencia")
    prev_df = parse_prevalencia(prev_raw)
    if prev_df.empty:
        st.warning("No se pudo leer la tabla de prevalencia.")
        st.caption(f"Detalle: {prev_source}")
    else:
        fig = px.bar(
            prev_df,
            x="GrupoEdad",
            y="Prevalencia",
            text="Prevalencia",
            title="Prevalencia de enfermedades cardiovasculares por edad",
        )
        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        fig.update_layout(title_x=0.5, title_xanchor="center")
        fig = style_chart(fig)
        chart_container(fig)
        st.dataframe(prev_df, use_container_width=True)

with tab_comp:
    section_header("Comparacion Santander vs Valle", "Fuente: Comparacion")
    if comp_df.empty or comp_metrics.get("error"):
        st.warning("No se pudo leer Comparacion.")
        st.caption(f"Detalle: {comp_source}")
        st.stop()

    comp_work = comp_metrics["comp_work"]
    ent_col = comp_metrics["ent_col"]
    sant_col = comp_metrics["sant_col"]
    valle_col = comp_metrics["valle_col"]
    icb_col = comp_metrics["icb_col"]
    foscal_col = comp_metrics["foscal_col"]
    sant_total = comp_metrics["sant_total"]
    valle_total = comp_metrics["valle_total"]
    atendidos_total = comp_metrics["atendidos_total"]
    pct_atendido = comp_metrics["pct_atendido"]
    posibles_valle = comp_metrics["posibles_valle"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("% atendido Santander", f"{pct_atendido:.2%}" if pd.notna(pct_atendido) else "NA")
    col2.metric("Atendidos Santander", f"{atendidos_total:,.0f}" if pd.notna(atendidos_total) else "NA")
    col3.metric("Posibles atendidos Valle", f"{posibles_valle:,.0f}" if pd.notna(posibles_valle) else "NA")
    col4.metric("Afiliados Valle del Cauca", f"{valle_total:,.0f}" if pd.notna(valle_total) else "NA")

    eps_table = comp_work[~comp_work["ent_norm"].isin(["otras", "total"])].copy()
    eps_table["Atendidos"] = pd.to_numeric(eps_table[icb_col], errors="coerce") + pd.to_numeric(
        eps_table[foscal_col], errors="coerce"
    )
    eps_view = eps_table[[ent_col, sant_col, valle_col, icb_col, foscal_col, "Atendidos"]].rename(
        columns={
            ent_col: "EPS",
            sant_col: "Santander afiliados",
            valle_col: "Valle afiliados",
            icb_col: "ICB atendidos",
            foscal_col: "Grupo Foscal atendidos",
        }
    )
    st.dataframe(eps_view, use_container_width=True)
    st.caption("Atendidos = ICB atendidos + Grupo Foscal atendidos.")

    divider()
    section_header("Distribucion por edad", "Calculado con EPS_Edad + Prevalencia")
    if edad_df.empty:
        st.warning("No se pudo leer EPS_Edad.")
        st.caption(f"Detalle: {edad_source}")
        st.stop()

    edad_cols = [str(c) for c in edad_df.columns]
    dept_col = find_col(edad_cols, ["departamento"]) or "DEPARTAMENTO"
    age_col = find_col(edad_cols, ["quinquenio"]) or "QUINQUENIOS DANE"
    total_col = find_col(edad_cols, ["total", "afiliados"]) or "TOTAL AFILIADOS"

    edad_work = edad_df.copy()
    edad_work[dept_col] = edad_work[dept_col].astype(str)
    edad_work = edad_work[edad_work[dept_col].str.contains("valle del cauca", case=False, na=False)]
    edad_work["GrupoEdad"] = edad_work[age_col].map(map_quinquenio_to_group)
    edad_work[total_col] = pd.to_numeric(edad_work[total_col], errors="coerce")
    edad_work = edad_work.dropna(subset=["GrupoEdad", total_col])
    grouped = edad_work.groupby("GrupoEdad")[total_col].sum().reset_index()
    total_aff_valle = grouped[total_col].sum()

    if pd.notna(valle_total) and total_aff_valle > 0:
        if abs(total_aff_valle - valle_total) / max(valle_total, 1) > 0.05:
            st.warning(
                "El total de afiliados Valle en EPS_Edad difiere del TOTAL en Comparacion."
            )

    prev_map = prev_df.set_index("GrupoEdad")["Prevalencia"].to_dict()
    grouped["Prevalencia"] = grouped["GrupoEdad"].map(prev_map)
    grouped["PctPoblacion"] = grouped[total_col] / total_aff_valle if total_aff_valle else np.nan
    grouped["PacientesEstimados"] = grouped["Prevalencia"] * grouped["PctPoblacion"] * total_aff_valle
    total_pacientes_est = grouped["PacientesEstimados"].sum()
    grouped["Ponderacion"] = (
        grouped["PacientesEstimados"] / total_pacientes_est if total_pacientes_est else np.nan
    )
    if pd.notna(posibles_valle):
        grouped["PacientesPorEdad"] = grouped["Ponderacion"] * posibles_valle
    else:
        grouped["PacientesPorEdad"] = np.nan

    order = ["0-19", "20-39", "40-59", "60-79", "80+"]
    grouped["GrupoEdad"] = pd.Categorical(grouped["GrupoEdad"], categories=order, ordered=True)
    grouped = grouped.sort_values("GrupoEdad")

    st.dataframe(
        grouped.rename(columns={total_col: "Afiliados"}).reset_index(drop=True),
        use_container_width=True,
    )

    divider()
    section_header("Formulas de calculo", "Guia de interpretacion")
    formula_df = pd.DataFrame(
        {
            "Campo": [
                "% atendido Santander",
                "Posibles atendidos Valle",
                "% poblacion Valle",
                "Pacientes estimados",
                "Ponderacion",
                "Pacientes por edad",
                "Atendidos (EPS)",
            ],
            "Formula": [
                "(ICB atendidos + Grupo Foscal atendidos) / Afiliados Santander (TOTAL)",
                "% atendido Santander * Afiliados Valle (TOTAL)",
                "Afiliados grupo / Total afiliados Valle",
                "Prevalencia * % poblacion Valle * Total afiliados Valle",
                "Pacientes estimados grupo / Pacientes estimados totales",
                "Ponderacion * Posibles atendidos Valle",
                "ICB atendidos + Grupo Foscal atendidos",
            ],
        }
    )
    st.dataframe(formula_df, use_container_width=True)

    if grouped["PacientesPorEdad"].notna().any():
        fig = px.bar(
            grouped,
            x="GrupoEdad",
            y="PacientesPorEdad",
            title="Pacientes por edad (Valle del Cauca)",
            labels={"GrupoEdad": "Grupo de edad", "PacientesPorEdad": "Pacientes"},
        )
        fig.update_layout(title_x=0.5, title_xanchor="center")
        fig = style_chart(fig)
        chart_container(fig)

with tab_tar:
    section_header("Tarifas por sede y servicio", "Fuente: Tarifas")
    if tarifas_df.empty:
        st.warning("No se pudo leer Tarifas.")
        st.caption(f"Detalle: {tarifas_source}")
        st.stop()

    tarifa_cols = [str(c) for c in tarifas_df.columns]
    sede_col = find_col_any(tarifa_cols, [["sede"], ["ciudad"], ["departamento"]])
    servicio_col = find_col_any(tarifa_cols, [["servicio"]])
    procedimiento_col = find_col_any(tarifa_cols, [["procedimiento"]])
    tarifa_col = find_col_any(tarifa_cols, [["tarifa"], ["precio"], ["valor"]])
    pacientes_col = find_col_any(tarifa_cols, [["pacientes"], ["paciente"]])
    inter_col = find_col_any(tarifa_cols, [["intervencion"], ["intervenciones"]])

    missing = [name for name, col in [("Sede", sede_col), ("Servicio", servicio_col), ("Tarifa", tarifa_col), ("Pacientes", pacientes_col)] if col is None]
    if missing:
        st.warning("Faltan columnas requeridas en Tarifas.")
        st.caption(f"Columnas encontradas: {tarifa_cols}")
        st.stop()

    posibles_valle = comp_metrics.get("posibles_valle")
    if posibles_valle is None or pd.isna(posibles_valle):
        st.warning("No se pudo calcular posibles pacientes Valle. Revisa Comparacion.")

    scenario = st.session_state.get("precio_scenario", "Santander")
    st.caption(f"Escenario de precios activo: {scenario}.")
    if "Competencia Cali" in scenario:
        st.warning("Escenario Competencia Cali pendiente: no hay tarifas Cali en el Excel.")
    st.caption(
        "%% Intervenciones usados: MDNI 46.02%, CONSULTA 135.99%, REPRO 46.07%, HEMO 8.99%, "
        "ELECTRO 8.72%, CIRUGIA 9.15%."
    )

    missing_services = []
    for srv in tarifas_df[servicio_col].dropna().astype(str).unique():
        if pd.isna(map_interv_pct(srv, INTERV_PCT_MAP)):
            missing_services.append(srv)
    if missing_services:
        st.warning(
            "Servicios sin % de intervenciones mapeado: "
            + ", ".join(sorted(missing_services)[:10])
            + ("..." if len(missing_services) > 10 else "")
        )

    sant_tab = build_tarifa_table(
        tarifas_df,
        sede_col=sede_col,
        target_sede="Santander",
        servicio_col=servicio_col,
        procedimiento_col=None,
        tarifa_col=tarifa_col,
        pacientes_col=pacientes_col,
        posibles_valle=posibles_valle,
        inter_col=inter_col,
        interv_map=INTERV_PCT_MAP,
    )
    bog_tab = build_tarifa_table(
        tarifas_df,
        sede_col=sede_col,
        target_sede="Bogota",
        servicio_col=servicio_col,
        procedimiento_col=None,
        tarifa_col=tarifa_col,
        pacientes_col=pacientes_col,
        posibles_valle=posibles_valle,
        inter_col=inter_col,
        interv_map=INTERV_PCT_MAP,
    )

    proc_sant = build_tarifa_table(
        tarifas_df,
        sede_col=sede_col,
        target_sede="Santander",
        servicio_col=servicio_col,
        procedimiento_col=procedimiento_col,
        tarifa_col=tarifa_col,
        pacientes_col=pacientes_col,
        posibles_valle=posibles_valle,
        inter_col=inter_col,
        interv_map=INTERV_PCT_MAP,
    )
    proc_bog = build_tarifa_table(
        tarifas_df,
        sede_col=sede_col,
        target_sede="Bogota",
        servicio_col=servicio_col,
        procedimiento_col=procedimiento_col,
        tarifa_col=tarifa_col,
        pacientes_col=pacientes_col,
        posibles_valle=posibles_valle,
        inter_col=inter_col,
        interv_map=INTERV_PCT_MAP,
    )

    if procedimiento_col is None:
        st.caption("No se encontro columna de Procedimiento; se usa Servicio como proxy.")

    money_cols = ["TarifaPromedio", "Ventas"]

    section_header("Tabla Santander (por servicio)")
    st.dataframe(format_currency_df(sant_tab, money_cols), use_container_width=True)
    section_header("Tabla Bogotá (por servicio)")
    st.dataframe(format_currency_df(bog_tab, money_cols), use_container_width=True)
    section_header("Procedimientos Santander")
    st.dataframe(format_currency_df(proc_sant, money_cols), use_container_width=True)
    section_header("Procedimientos Bogotá")
    st.dataframe(format_currency_df(proc_bog, money_cols), use_container_width=True)

with tab_sant:
    section_header("Sede Santander (Estados de resultados)")
    if sant_df.empty:
        st.warning("No se pudo leer la hoja SANTANDER.")
        st.caption(f"Detalle: {sant_source}")
        st.stop()

    account_col = sant_df.columns[0]
    month_cols = [c for c in sant_df.columns[1:] if is_date_like(c)]
    if not month_cols:
        st.warning("No se detectaron columnas mensuales con fechas en SANTANDER.")
        st.caption(f"Columnas: {[str(c) for c in sant_df.columns]}")
        st.stop()

    sant_work = filter_valid_accounts(sant_df, account_col)
    long_df = sant_work[[account_col] + month_cols].melt(
        id_vars=account_col, var_name="Month", value_name="Valor"
    )
    long_df["Valor"] = pd.to_numeric(long_df["Valor"], errors="coerce")
    long_df["Year"] = pd.to_datetime(long_df["Month"], errors="coerce").dt.year
    long_df = long_df.dropna(subset=["Year"])

    annual_df = (
        long_df.groupby([account_col, "Year"], dropna=False)["Valor"]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={account_col: "Cuenta"})
    )

    if annual_df.empty:
        st.warning("No se pudo construir la tabla anual de Santander.")
        st.stop()

    annual_pivot = annual_df.pivot_table(
        index="Cuenta", columns="Year", values="Valor", aggfunc="sum"
    ).sort_index()

    ingresos_series = pick_ingresos_series(annual_pivot)
    if ingresos_series is None:
        st.warning("No se encontro la cuenta 'Ingresos' en Santander.")
        st.stop()

    ingresos_series = ingresos_series.sort_index()

    yoy = ingresos_series / ingresos_series.shift(1) - 1
    growth_avg = weighted_growth(yoy)

    section_header("Ingresos anuales y crecimiento YoY")
    ingresos_view = pd.DataFrame(
        {"Ingresos": ingresos_series, "YoY": yoy}
    ).reset_index().rename(columns={"Year": "Año"})
    st.dataframe(
        ingresos_view.style.format({"Ingresos": fmt_currency, "YoY": fmt_percent}),
        use_container_width=True,
    )
    st.caption("Cifras en millones. YoY = crecimiento anual de ingresos.")

    section_header("Proporciones vs ingresos (historico)")
    ratio_df = annual_pivot.div(ingresos_series, axis=1).replace([np.inf, -np.inf], np.nan)
    ratio_df["Promedio"] = ratio_df.mean(axis=1, skipna=True)
    st.dataframe(ratio_df.reset_index(), use_container_width=True)

    st.caption("La estructura de proyeccion anual se muestra en el tab EEFF.")
