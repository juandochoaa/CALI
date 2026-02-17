from __future__ import annotations

import sys
import unicodedata
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

from dashboards.data_loader import load_cifras_eps
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

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


st.set_page_config(page_title="Talento Humano", layout="wide")
apply_theme()

page_header(
    "Talento Humano",
    "Disponibilidad y perfil del talento para el proyecto.",
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


def is_pct_col(col: str) -> bool:
    norm = normalize_text(col)
    return "%" in col or "porcentaje" in norm or "pct" in norm


def add_total_and_avg_row(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if df.empty:
        return df
    numeric_cols = []
    avg_cols = []
    for col in df.columns:
        if col == label_col:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().any():
            numeric_cols.append(col)
            if is_pct_col(col):
                avg_cols.append(col)
    if not numeric_cols:
        return df
    total_row = {label_col: "TOTAL"}
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        if col in avg_cols:
            total_row[col] = series.mean()
        else:
            total_row[col] = series.sum(min_count=1)
    return pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)


th_df, th_source = load_cifras_eps("TH")

view = subsection_selector(
    ["Resumen", "Base de talento", "Notas"],
    key="talento_view",
    label="Vista",
)

if view == "Resumen":
    section_header("Panorama de talento", "Lectura ejecutiva")
    explain_box(
        "Como se calcula",
        [
            "Resumen narrativo de disponibilidad y formacion.",
            "No modifica calculos ni transformaciones.",
        ],
    )
    insight_cards(
        [
            (
                "Oferta local",
                "La region tiene base academica y clinica para perfiles cardio-cerebro-vasculares.",
            ),
            (
                "Riesgo de cobertura",
                "Los perfiles de alta especializacion deben planearse con anticipacion por curva de formacion.",
            ),
            (
                "Accion recomendada",
                "Consolidar convenios universidad-hospital para asegurar embudo de talento en fases de crecimiento.",
            ),
        ],
        columns=3,
    )
    bullet_card(
        "Frentes de gestion",
        [
            "Mapear especialidades y subespecialidades criticas.",
            "Definir estrategia de atraccion y retencion por perfil.",
            "Alinear formacion clinica con capacidad instalada proyectada.",
        ],
    )

elif view == "Base de talento":
    section_header("Base de talento humano", "Fuente: hoja TH")
    explain_box(
        "Como se calcula",
        [
            "Se carga la hoja TH del Excel Cali ANALISIS.",
            "Se eliminan filas y columnas vacias.",
            "Se agrega fila TOTAL (sumas) y PROMEDIO en columnas tipo porcentaje.",
        ],
    )

    if th_df.empty:
        st.warning("No se pudo leer la hoja TH.")
        st.caption(f"Detalle: {th_source}")
        st.stop()

    work = th_df.copy()
    work = work.dropna(axis=0, how="all").dropna(axis=1, how="all")
    cols = [str(c) for c in work.columns]
    label_col = (
        find_col(cols, ["categoria"])
        or find_col(cols, ["cargo"])
        or find_col(cols, ["perfil"])
        or find_col(cols, ["programa"])
        or cols[0]
    )
    work[label_col] = work[label_col].astype(str).str.strip()

    view_df = add_total_and_avg_row(work, label_col)

    numeric_cols = []
    fmt = {}
    for col in view_df.columns:
        if col == label_col:
            continue
        series = pd.to_numeric(view_df[col], errors="coerce")
        if series.notna().any():
            numeric_cols.append(col)
            if is_pct_col(col):
                fmt[col] = lambda v: "" if pd.isna(v) else (f"{v:.2%}" if abs(v) <= 1.5 else f"{v:,.2f}%")
            else:
                fmt[col] = lambda v: "" if pd.isna(v) else f"{v:,.0f}"

    st.caption(f"Fuente: {th_source}")
    if numeric_cols:
        st.dataframe(view_df.style.format(fmt), width='stretch')
    else:
        st.dataframe(view_df, width='stretch')

else:
    section_header("Notas de interpretacion", "Como leer la tabla")
    explain_box(
        "Como se calcula",
        [
            "TOTAL suma columnas con datos aditivos (personas, cupos, vacantes).",
            "PROMEDIO se usa solo para columnas tipo porcentaje.",
            "Si una columna representa porcentajes en 0-100, se mantiene el valor.",
        ],
    )
    text_card(
        "Lectura sugerida",
        "Valida la escala de cada columna antes de interpretar porcentajes o comparativos entre perfiles.",
    )
    bullet_card(
        "Buenas practicas",
        [
            "Separar indicadores de volumen y de eficiencia en visualizaciones diferentes.",
            "Mantener definiciones consistentes para porcentaje, tasa y conteo.",
            "Documentar supuestos de disponibilidad por especialidad.",
        ],
    )
    takeaway_box(
        "Siguiente paso",
        "Cuando agregues nuevas subtablas de talento, usa este mismo formato: resumen, evidencia y notas.",
    )

