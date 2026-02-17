from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from dashboards.data_loader import load_capacidad_objetivo
from dashboards.ui import (
    append_total_row,
    apply_theme,
    bullet_card,
    insight_cards,
    page_header,
    section_header,
    takeaway_box,
    text_card,
)

st.set_page_config(page_title="Resumen", layout="wide")
apply_theme()

page_header(
    "Estudio de Mercado",
    "Hospital Cardio-Cerebro-Vascular en Cali",
    "Resumen",
)


tab_proyecto, tab_demanda, tab_competencia, tab_talento = st.tabs(
    ["Proyecto", "Demanda", "Competencia", "Talento"]
)

with tab_proyecto:
    section_header("Proyecto", "Vision general y alcance")
    insight_cards(
        [
            (
                "Proposito",
                "Hospital especializado en atencion cardio-cerebro-vascular para Cali y Valle del Cauca.",
            ),
            (
                "Cobertura",
                "Mercado objetivo concentrado en Cali con alcance regional para alta complejidad.",
            ),
            (
                "Enfoque",
                "Integrar demanda potencial, competencia y talento para validar viabilidad integral.",
            ),
        ],
        columns=3,
    )
    takeaway_box(
        "Resumen ejecutivo",
        "El proyecto combina posicionamiento geografico, capacidad instalada y especializacion clinica.",
    )

    section_header("Capacidad objetivo", "Dotacion inicial propuesta")
    cap_df = load_capacidad_objetivo()
    cap_df = append_total_row(cap_df, "Equipo", ["Cantidad"])
    st.dataframe(cap_df, width='content', hide_index=True)

    text_card(
        "Alcance geografico",
        "Valle del Cauca, con enfasis operativo y comercial en Cali.",
    )

with tab_demanda:
    section_header("Analisis de demanda", "Frentes de evaluacion")
    insight_cards(
        [
            (
                "EPS y pagadores",
                "Analizar capacidad de pago, liquidez y contratacion para servicios de alta complejidad.",
            ),
            (
                "Poblacion objetivo",
                "Estimar pacientes potenciales con base en afiliacion y prevalencia cardio-cerebro-vascular.",
            ),
            (
                "Proyeccion comercial",
                "Definir potencial de ingresos usando pro rata y comparativos entre regiones.",
            ),
        ],
        columns=3,
    )
    bullet_card(
        "Lineas de trabajo",
        [
            "Identificar EPS prioritarias en Cali y Valle.",
            "Cruzar afiliacion, prevalencia y estructura demografica.",
            "Proyectar ventas con supuestos trazables y comparables.",
        ],
    )

with tab_competencia:
    section_header("Analisis de la competencia", "Capacidad instalada y posicionamiento")
    insight_cards(
        [
            (
                "Infraestructura",
                "Revisar angiografos, quirofanos, camas y oferta de alta complejidad por competidor.",
            ),
            (
                "Solidez financiera",
                "Comparar estados financieros y desempeno operativo por institucion.",
            ),
        ],
        columns=2,
    )
    bullet_card(
        "Instituciones a revisar",
        [
            "Fundacion Valle de Lili",
            "Clinica Imbanaco",
            "Angiografia de Occidente",
            "DIME Clinica Neurocardiovascular",
            "Cardioprevent",
            "Corazon y Aorta",
            "Instituto Diagnostico",
        ],
    )

with tab_talento:
    section_header("Talento humano", "Disponibilidad de especialistas")
    insight_cards(
        [
            (
                "Oferta academica",
                "Identificar universidades con programas de cardiologia, neurocirugia y subespecialidades.",
            ),
            (
                "Cantera clinica",
                "Mapear hospitales formadores y convenios docentes en la region.",
            ),
            (
                "Riesgo de dotacion",
                "Detectar brechas de perfiles criticos para apertura y escalamiento.",
            ),
        ],
        columns=3,
    )
    bullet_card(
        "Referencias iniciales",
        [
            "Univalle - Cirugia vascular periferica.",
            "Univalle - Especializacion en neurocirugia.",
            "Univalle - Especializacion en cardiologia.",
            "Hospitales universitarios de la region y convenios con Clinica DIME.",
        ],
    )

