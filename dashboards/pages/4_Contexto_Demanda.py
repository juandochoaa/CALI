from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dashboards.data_loader import load_cifras_eps, load_cifras_eps_raw
from dashboards.ui import (
    apply_theme,
    bullet_card,
    explain_box,
    insight_cards,
    page_header,
    section_header,
    subsection_selector,
    takeaway_box,
    text_card,
)
from src.models.target_population import compute_target_population

st.set_page_config(page_title="Contexto y Demanda", layout="wide")
apply_theme()

page_header(
    "Contexto y Demanda",
    "Poblacion objetivo, causas de mortalidad y comparacion EPS.",
    "Resumen",
)


def _compute_and_store_target_population_snapshot() -> dict:
    comp_df, comp_source = load_cifras_eps("Comparacion")
    edad_df, edad_source = load_cifras_eps("EPS_Edad")
    prev_raw, prev_source = load_cifras_eps_raw("Prevalencia", header=None)

    snapshot = compute_target_population(
        comparacion_df=comp_df,
        eps_edad_df=edad_df,
        prevalencia_df_raw=prev_raw,
    )
    metadata = snapshot.setdefault("metadata", {})
    metadata["sources"] = {
        "comparacion": comp_source,
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
    sources = snapshot.get("metadata", {}).get("sources", {})

    for msg in warnings:
        st.warning(msg)

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
        "% atendido Santander",
        _fmt_metric(summary.get("pct_atendido_santander"), percent=True),
    )
    col2.metric(
        "Atendidos Santander",
        _fmt_metric(summary.get("atendidos_santander")),
    )
    col3.metric(
        "Posibles atendidos Valle",
        _fmt_metric(summary.get("posibles_atendidos_valle")),
    )
    col4.metric(
        "Afiliados Valle del Cauca",
        _fmt_metric(summary.get("afiliados_valle_total")),
    )

    section_header("Tabla comparativa EPS")
    if isinstance(eps_view, pd.DataFrame) and not eps_view.empty:
        st.dataframe(eps_view, width="stretch")
        st.caption("Atendidos = ICB atendidos + Grupo Foscal atendidos.")
    else:
        st.warning("No hay tabla comparativa EPS disponible.")

    if show_age_block:
        section_header("Distribucion por edad", "EPS_Edad + Prevalencia")
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
        source_parts = [
            f"Comparacion: {sources.get('comparacion', 'NA')}",
            f"EPS_Edad: {sources.get('eps_edad', 'NA')}",
            f"Prevalencia: {sources.get('prevalencia', 'NA')}",
        ]
        st.caption(" | ".join(source_parts))


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
