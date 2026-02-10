from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


import unicodedata
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

from dashboards.data_loader import load_cifras_eps
from dashboards.ui import apply_theme, append_total_row, divider, explain_box, page_header, section_header


st.set_page_config(page_title="Contexto y Demanda", layout="wide")
apply_theme()

page_header(
    "Contexto y Demanda",
    "Poblacion objetivo, causas de mortalidad y comparacion EPS.",
    "Resumen",
)


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


section_header("Poblacion objetivo")
explain_box(
    "Como se calcula",
    [
        "Seccion descriptiva del mercado objetivo.",
        "Se apoya en poblacion, prevalencia y afiliacion en salud.",
    ],
)
st.markdown("Objetivo: estimar el volumen potencial de pacientes.")
st.markdown(
    """
- Tamano de la poblacion en Cali y Valle del Cauca
- Prevalencia de enfermedad cardiovascular y enfermedad cerebrovascular
- Cifras afiliacion en Salud
- Analisis pro rata con base en poblacion Santander sabiendo que existe comuneros, HIC y FCV (Ratio)
"""
)

divider()
section_header("Distribucion porcentual de causas de defuncion (Colombia)")
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
        st.image(str(img_2024), use_container_width=True)
    else:
        st.warning("No se encontro Pastel_muertes_2024.png en data/raw.")
with col2:
    if img_2025.exists():
        st.image(str(img_2025), use_container_width=True)
    else:
        st.warning("No se encontro Pastel_muertes_2025.png en data/raw.")

st.caption("Fuente: DANE - Estadisticas Vitales. 2025pr: cifras preliminares.")

divider()
section_header("Contexto regional (Santander)")
explain_box(
    "Como se calcula",
    [
        "Resumen de contexto regional (mortalidad cardiovascular).",
        "No hay calculos en esta seccion.",
    ],
)
st.markdown(
    "La principal causa de mortalidad en la region son las enfermedades cardiovasculares, "
    "con una tasa alarmante de 183.8 por cada 100,000 habitantes en el ano 2022. "
    "Destacan los municipios de Guapota, Betulia, Charta, Landazuri y Chipata como los mas "
    "afectados por mortalidad prematura, segun datos de estadisticas vitales del SISPRO."
)

divider()
section_header("Comparacion EPS (Santander vs Valle)")
explain_box(
    "Como se calcula",
    [
        "Cruce de afiliados y atendidos por EPS.",
        "Atendidos = ICB atendidos + Grupo Foscal atendidos.",
        "Se agrega fila TOTAL.",
    ],
)

comp_df, comp_source = load_cifras_eps("Comparacion")
if comp_df.empty:
    st.warning("No se pudo leer la hoja Comparacion.")
    st.caption(f"Detalle: {comp_source}")
else:
    cols = [str(c) for c in comp_df.columns]
    ent_col = find_col(cols, ["entidad"]) or cols[0]
    sant_col = find_col(cols, ["santander", "afiliados"])
    valle_col = find_col(cols, ["valle", "afiliados"])
    icb_col = find_col(cols, ["icb", "atendidos"])
    foscal_col = find_col(cols, ["foscal", "atendidos"])

    if not all([sant_col, valle_col, icb_col, foscal_col]):
        st.warning("No se encontraron columnas requeridas en Comparacion.")
        st.caption(f"Columnas: {cols}")
    else:
        comp_work = comp_df.copy()
        comp_work[ent_col] = comp_work[ent_col].astype(str).str.strip()
        comp_work["ent_norm"] = comp_work[ent_col].map(normalize_text)

        eps_table = comp_work[~comp_work["ent_norm"].isin(["total"])].copy()
        eps_table.loc[eps_table["ent_norm"] == "otras", ent_col] = "OTROS"
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
        eps_view = append_total_row(
            eps_view,
            "EPS",
            [
                "Santander afiliados",
                "Valle afiliados",
                "ICB atendidos",
                "Grupo Foscal atendidos",
                "Atendidos",
            ],
        )
        st.dataframe(eps_view, use_container_width=True)
        st.caption("Atendidos = ICB atendidos + Grupo Foscal atendidos.")
