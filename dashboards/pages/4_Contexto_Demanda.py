from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dashboards.data_loader import load_cifras_eps, load_cifras_eps_raw
from dashboards.ui import (
    apply_theme,
    bullet_card,
    chart_container,
    explain_box,
    insight_cards,
    page_header,
    section_header,
    style_chart,
    subsection_selector,
    takeaway_box,
    text_card,
)
from src.models.target_population import compute_target_population, normalize_text

st.set_page_config(page_title="Contexto y Demanda", layout="wide")
apply_theme()

page_header(
    "Contexto y Demanda",
    "Poblacion objetivo, causas de mortalidad y comparacion EPS.",
    "Resumen",
)


def _compute_and_store_target_population_snapshot() -> dict:
    comp_df = pd.DataFrame()
    comp_source = "No disponible"
    comp_sheet_used = "Comparacion"
    for sheet in ["Comparacion_Desglose", "Comparacion Desglose", "ComparacionDesglose", "Comparacion"]:
        cand_df, cand_source = load_cifras_eps(sheet)
        if not cand_df.empty:
            comp_df = cand_df
            comp_source = cand_source
            comp_sheet_used = sheet
            break

    edad_df, edad_source = load_cifras_eps("EPS_Edad")
    prev_raw, prev_source = load_cifras_eps_raw("Prevalencia", header=None)

    snapshot = compute_target_population(
        comparacion_df=comp_df,
        eps_edad_df=edad_df,
        prevalencia_df_raw=prev_raw,
    )
    metadata = snapshot.setdefault("metadata", {})
    metadata["sources"] = {
        "comparacion_desglose": comp_source if "desglose" in normalize_text(comp_sheet_used) else "NA",
        "comparacion": comp_source if normalize_text(comp_sheet_used) == "comparacion" else "NA",
        "eps_edad": edad_source,
        "prevalencia": prev_source,
    }

    st.session_state["target_population_snapshot_v1"] = snapshot
    objetivo_valle = pd.to_numeric(
        snapshot.get("summary_metrics", {}).get("posibles_atendidos_valle"),
        errors="coerce",
    )
    st.session_state["objetivo_valle_pacientes"] = objetivo_valle
    return snapshot


def _get_target_population_snapshot() -> dict:
    snapshot = st.session_state.get("target_population_snapshot_v1")
    if isinstance(snapshot, dict) and "summary_metrics" in snapshot:
        return snapshot
    return _compute_and_store_target_population_snapshot()


def _fmt_metric(value: object, *, percent: bool = False) -> str:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return "NA"
    return f"{num:.2%}" if percent else f"{num:,.0f}"


def _render_target_population(
    snapshot: dict,
    *,
    show_age_block: bool,
    show_formula_block: bool,
) -> None:
    summary = snapshot.get("summary_metrics", {})
    eps_view = snapshot.get("eps_view", pd.DataFrame())
    edad_view = snapshot.get("edad_view", pd.DataFrame())
    formula_view = snapshot.get("formula_view", pd.DataFrame())
    edad_chart_df = snapshot.get("edad_chart_df", pd.DataFrame())
    warnings = snapshot.get("warnings", [])
    metadata = snapshot.get("metadata", {})
    sources = metadata.get("sources", {})
    method = str(metadata.get("method", "legacy"))

    for msg in warnings:
        st.warning(msg)

    if method == "comparacion_desglose":
        explain_box(
            "Metodologia",
            [
                "Conceptualmente, primero se estima TAM por edad con afiliados y prevalencia.",
                "Luego se aplica un factor de captura historico de Santander para obtener SOM.",
                "Operativamente, la columna 'Pacientes por edad' ya viene ajustada por ese factor y se suma directamente.",
            ],
        )
        st.markdown("### Formulacion")
        st.latex(r"TAM_g = Afiliados_g \times Prev_g")
        st.latex(r"TAM = \sum_g TAM_g")
        st.latex(r"\alpha = \frac{Atendidos_{Santander}}{Afiliados_{Santander}}")
        st.latex(r"SOM_g = TAM_g \times \alpha")
        st.latex(r"Poblacion\ Objetivo = \sum_g PacientesPorEdad_g")
    else:
        explain_box(
            "Como se calcula",
            [
                "Se usa Comparacion para calcular % atendido Santander y objetivo de pacientes en Valle.",
                "Se construye tabla EPS con afiliados y atendidos (ICB + Grupo Foscal).",
                "La distribucion por edad se calcula con EPS_Edad + Prevalencia.",
            ],
        )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Poblacion objetivo",
        _fmt_metric(summary.get("posibles_atendidos_valle")),
    )
    col2.metric(
        "Atendidos Santander",
        _fmt_metric(summary.get("atendidos_santander")),
    )
    col3.metric(
        "% atendido Santander",
        _fmt_metric(summary.get("pct_atendido_santander"), percent=True),
    )
    col4.metric(
        "Afiliados Valle del Cauca",
        _fmt_metric(summary.get("afiliados_valle_total")),
    )

    section_header("Tabla comparativa EPS")
    if isinstance(eps_view, pd.DataFrame) and not eps_view.empty:
        st.dataframe(eps_view, width="stretch")
        st.caption("Atendidos = ICB atendidos + Grupo Foscal atendidos.")
    elif method != "comparacion_desglose":
        st.warning("No hay tabla comparativa EPS disponible.")
    else:
        st.info("En modo Comparacion_Desglose no se usa tabla comparativa EPS para el objetivo.")

    if show_age_block:
        subtitle = (
            "Fuente oficial: Comparacion_Desglose (Pacientes por edad)"
            if method == "comparacion_desglose"
            else "EPS_Edad + Prevalencia"
        )
        section_header("Distribucion por edad", subtitle)
        if isinstance(edad_view, pd.DataFrame) and not edad_view.empty:
            st.dataframe(edad_view, width="stretch")
        else:
            st.warning("No hay distribucion por edad disponible.")

        if isinstance(edad_chart_df, pd.DataFrame) and not edad_chart_df.empty:
            chart = edad_chart_df.copy()
            chart["PacientesPorEdad"] = pd.to_numeric(chart["PacientesPorEdad"], errors="coerce")
            chart = chart.dropna(subset=["PacientesPorEdad"])
            if not chart.empty:
                st.bar_chart(
                    chart.set_index("GrupoEdad")["PacientesPorEdad"],
                    use_container_width=True,
                )

    if show_formula_block:
        section_header("Formulas de calculo")
        if isinstance(formula_view, pd.DataFrame) and not formula_view.empty:
            st.dataframe(formula_view, width="stretch")
        else:
            st.warning("No hay tabla de formulas disponible.")

    if sources:
        source_parts = []
        if sources.get("comparacion_desglose", "NA") != "NA":
            source_parts.append(f"Comparacion_Desglose: {sources.get('comparacion_desglose')}")
        if sources.get("comparacion", "NA") != "NA":
            source_parts.append(f"Comparacion: {sources.get('comparacion')}")
        source_parts.extend(
            [
                f"EPS_Edad: {sources.get('eps_edad', 'NA')}",
                f"Prevalencia: {sources.get('prevalencia', 'NA')}",
            ]
        )
        st.caption(" | ".join(source_parts))


def _build_barrera_salud_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Departamento": ["Santander", "Valle del Cauca"],
            2019: [3.8, 2.4],
            2020: [1.3, 1.2],
            2021: [1.0, 1.7],
            2022: [0.9, 1.6],
            2023: [1.3, 0.7],
            2024: [0.5, 1.2],
        }
    )


def _render_barrera_salud_section() -> None:
    section_header("Barrera en salud", "Comparativo Santander vs Valle del Cauca")
    explain_box(
        "Fuente y lectura",
        [
            "Tomado del Indice de pobreza multidimensional del DANE con datos de la Encuesta de Calidad de Vida.",
            "Indica el % de poblacion u hogares con barreras para acceder a servicios de salud.",
            "Entre mas alto el valor, mas personas reportan dificultades para acceder.",
        ],
    )

    barrera_df = _build_barrera_salud_df()
    year_cols = [c for c in barrera_df.columns if isinstance(c, int)]

    st.dataframe(
        barrera_df.style.format({y: "{:.1f}%" for y in year_cols}),
        width="stretch",
        hide_index=True,
    )

    long_df = barrera_df.melt(
        id_vars=["Departamento"],
        value_vars=year_cols,
        var_name="Año",
        value_name="BarreraSaludPct",
    )
    long_df["Año"] = pd.to_numeric(long_df["Año"], errors="coerce").astype(int)
    long_df["BarreraSaludPct"] = pd.to_numeric(long_df["BarreraSaludPct"], errors="coerce")

    fig = px.line(
        long_df,
        x="Año",
        y="BarreraSaludPct",
        color="Departamento",
        markers=True,
        title="Evolucion de barreras de acceso en salud",
        labels={"BarreraSaludPct": "% con barreras", "Departamento": "Departamento"},
    )
    fig.update_yaxes(ticksuffix="%")
    fig = style_chart(fig)
    chart_container(fig)

    idx_df = long_df.copy()
    base_vals = (
        idx_df[idx_df["Año"] == min(year_cols)][["Departamento", "BarreraSaludPct"]]
        .rename(columns={"BarreraSaludPct": "Base2019"})
    )
    idx_df = idx_df.merge(base_vals, on="Departamento", how="left")
    idx_df["IndiceBase2019"] = (idx_df["BarreraSaludPct"] / idx_df["Base2019"]) * 100.0

    fig_idx = px.line(
        idx_df,
        x="Año",
        y="IndiceBase2019",
        color="Departamento",
        markers=True,
        title="Comparacion relativa (Indice base 2019 = 100)",
        labels={"IndiceBase2019": "Indice", "Departamento": "Departamento"},
    )
    fig_idx = style_chart(fig_idx)
    chart_container(fig_idx)

    avg_df = (
        long_df.groupby("Departamento", as_index=False)["BarreraSaludPct"]
        .mean()
        .rename(columns={"BarreraSaludPct": "Promedio_2019_2024"})
        .sort_values("Promedio_2019_2024", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(
        avg_df.style.format({"Promedio_2019_2024": "{:.2f}%"}),
        width="stretch",
        hide_index=True,
    )


view = subsection_selector(
    ["Resumen", "Mortalidad", "Contexto regional", "Comparacion EPS"],
    key="contexto_demanda_view",
    label="Vista",
)

if view == "Resumen":
    section_header("Poblacion objetivo", "Panorama sintetico")
    insight_cards(
        [
            (
                "Tamano del mercado",
                "Dimensionar poblacion en Cali y Valle del Cauca para estimar demanda potencial.",
            ),
            (
                "Perfil epidemiologico",
                "Cruzar prevalencia cardiovascular y cerebrovascular por grupos de edad.",
            ),
            (
                "Base de pagadores",
                "Usar afiliacion EPS para identificar volumen comercial posible.",
            ),
        ],
        columns=3,
    )
    bullet_card(
        "Enfoques de trabajo",
        [
            "Cuantificar afiliacion y atencion por EPS.",
            "Usar comparativo Santander para construir referencia de atencion.",
            "Traducir hallazgos en estimacion de pacientes potenciales.",
        ],
    )
    takeaway_box(
        "Lectura ejecutiva",
        "Esta pagina prioriza contexto para demanda: no muestra calculos financieros, sino base de mercado y riesgo.",
    )

    section_header("Calculo central de poblacion objetivo")
    snapshot = _compute_and_store_target_population_snapshot()
    _render_target_population(snapshot, show_age_block=True, show_formula_block=True)
    _render_barrera_salud_section()

elif view == "Mortalidad":
    section_header("Distribucion de causas de defuncion", "Colombia")
    explain_box(
        "Como se calcula",
        [
            "Se muestran imagenes oficiales de causas de defuncion.",
            "No hay calculos en esta seccion.",
        ],
    )

    raw_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    img_2024 = raw_dir / "Pastel_muertes_2024.png"
    img_2025 = raw_dir / "Pastel_muertes_2025.png"

    col1, col2 = st.columns(2)
    with col1:
        if img_2024.exists():
            st.image(str(img_2024), width="stretch")
        else:
            st.warning("No se encontro Pastel_muertes_2024.png en data/raw.")
    with col2:
        if img_2025.exists():
            st.image(str(img_2025), width="stretch")
        else:
            st.warning("No se encontro Pastel_muertes_2025.png en data/raw.")

    st.caption("Fuente: DANE - Estadisticas Vitales. 2025pr: cifras preliminares.")

elif view == "Contexto regional":
    section_header("Contexto regional (Santander)")
    explain_box(
        "Como se calcula",
        [
            "Resumen de contexto regional (mortalidad cardiovascular).",
            "No hay calculos en esta seccion.",
        ],
    )
    text_card(
        "Hallazgo principal",
        (
            "La principal causa de mortalidad en la region son las enfermedades cardiovasculares, "
            "con tasa de 183.8 por cada 100,000 habitantes en 2022."
        ),
    )
    bullet_card(
        "Municipios mas afectados",
        [
            "Guapota",
            "Betulia",
            "Charta",
            "Landazuri",
            "Chipata",
        ],
    )
    takeaway_box(
        "Implicacion para el proyecto",
        "El comportamiento regional refuerza la necesidad de una oferta especializada y de alto acceso oportuno.",
    )

else:
    section_header("Comparacion EPS (Santander vs Valle)")
    snapshot = _get_target_population_snapshot()
    _render_target_population(snapshot, show_age_block=False, show_formula_block=False)
