from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from dashboards.data_loader import load_capacidad_objetivo
from dashboards.ui import apply_theme, append_total_row, explain_box, page_header, section_header


st.set_page_config(page_title="Resumen", layout="wide")
apply_theme()

page_header(
    "Estudio de Mercado",
    "Hospital Cardio-Cerebro-Vascular en Cali",
    "Resumen",
)

section_header("Proyecto")
explain_box(
    "Como se calcula",
    [
        "Seccion descriptiva del alcance del proyecto.",
        "No hay calculos, solo contexto del hospital propuesto.",
    ],
)
st.markdown("**Proyecto:** Hospital Cardio-Cerebro-Vascular en Cali")
st.markdown(
    "**Descripcion:** Creacion de un hospital especializado en atencion cardiovascular y "
    "cerebrovascular en la ciudad de Cali, con cobertura para el Valle del Cauca."
)

section_header("Capacidad")
explain_box(
    "Como se calcula",
    [
        "Capacidad inicial instalada definida por el proyecto.",
        "Cantidades por equipo y camas (valores absolutos).",
    ],
)
cap_df = load_capacidad_objetivo()
cap_df = append_total_row(cap_df, "Equipo", ["Cantidad"])
st.dataframe(cap_df, use_container_width=False, hide_index=True)

section_header("Alcance Geografico")
explain_box(
    "Como se calcula",
    [
        "Definicion del mercado geografico objetivo.",
        "No hay calculos.",
    ],
)
st.markdown("Valle del Cauca con enfasis en Cali.")

section_header("Analisis de Demanda")
explain_box(
    "Como se calcula",
    [
        "Resumen de frentes de analisis: EPS, poblacion, prevalencia y afiliacion.",
        "No hay calculos en esta seccion.",
    ],
)
st.markdown("**EPS y Pagadores**")
st.markdown("Objetivo: identificar viabilidad financiera del mercado.")
st.markdown(
    """
- Identificacion de EPS presentes en Cali y Valle
- Valoracion financiera de las EPS (estados financieros, liquidez, riesgo de pago)
- Revisar cuentas contables para el dashboard Juan David
- Capacidad de contratacion para servicios de alta complejidad
"""
)

st.markdown("**Poblacion Objetivo**")
st.markdown("Objetivo: estimar el volumen potencial de pacientes.")
st.markdown(
    """
- Tamano de la poblacion en Cali y Valle del Cauca
- Prevalencia de enfermedad cardiovascular y enfermedad cerebrovascular
- Cifras de afiliacion en Salud
"""
)

st.markdown("**Proyeccion ventas**")
st.markdown("Objetivo: estimar ingresos esperados.")
st.markdown(
    """
- Analisis pro rata con base en poblacion Santander sabiendo que existe comuneros, HIC y FCV (Ratio)
"""
)

section_header("Analisis de la Competencia")
explain_box(
    "Como se calcula",
    [
        "Lista de competidores y variables a revisar (dotacion y EEFF).",
        "No hay calculos en esta seccion.",
    ],
)
st.markdown("Objetivo: entender la capacidad instalada y el posicionamiento actual.")
st.markdown(
    """
- Revisar numeros de angiografos, quirofanos, camas, estados financieros
- Fundacion Valle de Lili
- Clinica Imbanaco
- Angiografia de Occidente
- Otros
- Dime Clinica Neurocardiovascular S.A.
- Cardioprevent (diagnostico)
- Corazon y aorta
- Instituto Diagnostico (diagnostico, solo dos anos)
"""
)

section_header("Analisis del Talento Humano")
explain_box(
    "Como se calcula",
    [
        "Resumen de fuentes y universidades para talento especializado.",
        "No hay calculos en esta seccion.",
    ],
)
st.markdown("Objetivo: evaluar disponibilidad de personal especializado.")
st.markdown("**Formacion Academica**")
st.markdown(
    """
- Universidades que ofrecen Cardiologia, Cardiologia intervencionista
  - https://salud.univalle.edu.co/la-universidad/horarios-de-atencion/28-especializaciones-clinicas/239-cirugia-vascular-periferica
- Universidades que ofrecen programas de residencia y subespecializacion
  - https://salud.univalle.edu.co/especializacion-en-neurocirugia
  - https://salud.univalle.edu.co/especializacion-en-cardiologia
"""
)

st.markdown("**Hospitales Universitarios**")
st.markdown(
    """
- Hospitales formadores en la region
- Convenios educativos - Clinica DIME
"""
)
