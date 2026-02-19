from __future__ import annotations

import sys
import unicodedata
from pathlib import Path
from typing import List

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboards.data_loader import load_cifras_eps
from dashboards.ui import (
    apply_theme,
    bullet_card,
    chart_container,
    explain_box,
    insight_cards,
    page_header,
    section_header,
    style_chart,
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


def parse_numeric_col(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.str.replace(r"[^0-9,.\-]", "", regex=True)

    has_comma = cleaned.str.contains(",", regex=False, na=False)
    has_dot = cleaned.str.contains(r"\.", regex=True, na=False)
    both = has_comma & has_dot
    cleaned.loc[both] = (
        cleaned.loc[both]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    comma_only = has_comma & ~has_dot
    cleaned.loc[comma_only] = cleaned.loc[comma_only].str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def resolve_raw_image(filename: str) -> Path | None:
    path = ROOT_DIR / "data" / "raw" / filename
    if path.exists():
        return path
    return None


especialistas_df, especialistas_source = load_cifras_eps("Especialistas")

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

section_header("Formacion academica", "Universidades que ofrecen cardiologia, intervencionismo y neurointervencionismo")
explain_box(
    "Como se calcula",
    [
        "Fuente estructurada a partir de la informacion consolidada del equipo (sin dependencia de hoja TH).",
        "Se separa en dos niveles: universidad/base clinica y oferta de programas por universidad.",
    ],
)

universidades_df = pd.DataFrame(
    [
        {
            "Universidad": "Universidad del Valle (Univalle)",
            "Base / Convenio clinico": (
                "Principal formador clinico en la region. Base en HUV Evaristo Garcia, "
                "con rotaciones en Imbanaco, Valle del Lili y otras IPS."
            ),
        },
        {
            "Universidad": "Universidad ICESI",
            "Base / Convenio clinico": "Convenio exclusivo de formacion con Fundacion Valle del Lili.",
        },
        {
            "Universidad": "Universidad Santiago de Cali",
            "Base / Convenio clinico": "Convenios docencia-servicio con varias IPS (Imbanaco, entre otras).",
        },
        {
            "Universidad": "Pontificia Universidad Javeriana Cali",
            "Base / Convenio clinico": "Convenios docencia-servicio con varias IPS (Valle del Lili, entre otras).",
        },
        {
            "Universidad": "Universidad Libre Cali",
            "Base / Convenio clinico": "Convenios docencia-servicio con varias IPS (HUV, Imbanaco, etc.).",
        },
    ]
)
st.dataframe(universidades_df, width='stretch', hide_index=True)

programas_df = pd.DataFrame(
    [
        ("Universidad del Valle (Univalle)", "Pregrado en Medicina", "Internado"),
        ("Universidad del Valle (Univalle)", "Especializacion en Cardiologia", "Residencia"),
        ("Universidad del Valle (Univalle)", "Especializacion en Cirugia Vascular Periferica", "Residencia"),
        ("Universidad del Valle (Univalle)", "Especializacion en Neurocirugia", "Residencia"),
        ("Universidad del Valle (Univalle)", "Especializacion en Cirugia Pediatrica", "Residencia"),
        ("Universidad del Valle (Univalle)", "Especializacion en Anestesiologia", "Residencia"),
        ("Universidad del Valle (Univalle)", "Especializacion en Medicina Interna", "Residencia"),
        ("Universidad del Valle (Univalle)", "Especializacion en Medicina Critica y Cuidado Intensivo", "Residencia"),
        ("Universidad ICESI", "Pregrado en Medicina", "Internado"),
        ("Universidad ICESI", "Especializacion en Cardiologia", "Residencia"),
        ("Universidad ICESI", "Especializacion en Cardiologia Pediatrica", "Residencia"),
        ("Universidad ICESI", "Especializacion en Cirugia de Cabeza y Cuello", "Residencia"),
        ("Universidad ICESI", "Especializacion en Neurocirugia", "Residencia"),
        ("Universidad ICESI", "Especializacion en Radiologia Intervencionista", "Residencia"),
        ("Universidad ICESI", "Especializacion en Anestesiologia", "Residencia"),
        ("Universidad ICESI", "Especializacion en Medicina Interna", "Residencia"),
        ("Universidad ICESI", "Especializacion en Medicina Critica y Cuidado Intensivo", "Residencia"),
        ("Universidad Santiago de Cali", "Pregrado en Medicina", "Internado"),
        ("Universidad Santiago de Cali", "Especializacion en Medicina Interna", "Residencia"),
        ("Pontificia Universidad Javeriana Cali", "Pregrado en Medicina", "Internado"),
        ("Pontificia Universidad Javeriana Cali", "Especializacion en Cirugia Pediatrica", "Residencia"),
        ("Pontificia Universidad Javeriana Cali", "Especializacion en Anestesiologia", "Residencia"),
        ("Universidad Libre Cali", "Pregrado en Medicina", "Internado"),
        ("Universidad Libre Cali", "Especializacion en Medicina Interna", "Residencia"),
        ("Universidad Libre Cali", "Especializacion en Pediatria", "Residencia"),
    ],
    columns=["Universidad", "Programa", "Tipo"],
)
st.dataframe(programas_df, width='stretch', hide_index=True)

section_header("Residencia y subespecializacion", "Fundacion Valle del Lili - ICESI")
subespecializaciones_df = pd.DataFrame(
    [
        ("Fundacion Valle del Lili - ICESI", "Entrenamiento Avanzado en Ecocardiografia"),
        ("Fundacion Valle del Lili - ICESI", "Cardiologia intervencionista"),
        ("Fundacion Valle del Lili - ICESI", "Electrofisiologia"),
        ("Fundacion Valle del Lili - ICESI", "Hemodinamia estructural"),
        ("Fundacion Valle del Lili - ICESI", "Endovascular periferico"),
        ("Fundacion Valle del Lili - ICESI", "Neurointervencionismo"),
    ],
    columns=["Centro", "Programa avanzado"],
)
st.dataframe(subespecializaciones_df, width='stretch', hide_index=True)

section_header(
    "Número de Especialistas que cotizaron al SGSS a Mayo 2024",
    "Comparativo Santander vs Valle del Cauca",
)
if especialistas_df.empty:
    st.warning("No se pudo leer la hoja Especialistas.")
    st.caption(f"Detalle: {especialistas_source}")
else:
    esp_view = especialistas_df.copy()
    esp_view = esp_view.dropna(axis=0, how="all").dropna(axis=1, how="all")
    st.caption(f"Fuente: {especialistas_source}")
    st.dataframe(esp_view, width='stretch', hide_index=True)

    if not esp_view.empty:
        cols = [str(c) for c in esp_view.columns]
        specialist_col = (
            find_col(cols, ["especialista"])
            or find_col(cols, ["especialidad"])
            or find_col(cols, ["perfil"])
            or cols[0]
        )
        valle_col = find_col(cols, ["valle"])
        santander_col = find_col(cols, ["santander"])

        if specialist_col is None or valle_col is None or santander_col is None:
            st.warning(
                "No se pudieron identificar columnas para especialista, Santander y Valle en la hoja Especialistas."
            )
        else:
            chart_base = esp_view[[specialist_col, valle_col, santander_col]].copy()
            chart_base = chart_base.dropna(subset=[specialist_col], how="all").copy()
            chart_base[specialist_col] = chart_base[specialist_col].astype(str).str.strip()
            chart_base[valle_col] = parse_numeric_col(chart_base[valle_col])
            chart_base[santander_col] = parse_numeric_col(chart_base[santander_col])
            chart_base = chart_base.dropna(subset=[valle_col, santander_col], how="all").copy()

            if chart_base.empty:
                st.info("No hay datos numericos suficientes para graficar especialistas.")
            else:
                sort_order = (
                    chart_base[[valle_col, santander_col]]
                    .max(axis=1, skipna=True)
                    .sort_values(ascending=True)
                    .index
                )
                chart_base = chart_base.loc[sort_order]
                long_df = chart_base.melt(
                    id_vars=[specialist_col],
                    value_vars=[valle_col, santander_col],
                    var_name="Departamento",
                    value_name="Especialistas",
                )
                long_df["Departamento"] = long_df["Departamento"].replace(
                    {
                        valle_col: "Valle del Cauca",
                        santander_col: "Santander",
                    }
                )
                fig = px.bar(
                    long_df,
                    x="Especialistas",
                    y=specialist_col,
                    color="Departamento",
                    barmode="group",
                    orientation="h",
                    title="Especialistas cotizantes al SGSS - Santander vs Valle del Cauca",
                    labels={
                        "Especialistas": "Numero de especialistas",
                        specialist_col: "Especialidad",
                    },
                    color_discrete_map={
                        "Valle del Cauca": "#1f77b4",
                        "Santander": "#d62728",
                    },
                )
                fig = style_chart(fig)
                fig.update_layout(
                    font={"color": "#000000"},
                    title_font={"color": "#000000"},
                    legend_title_font={"color": "#000000"},
                    legend_font={"color": "#000000"},
                )
                fig.update_xaxes(title_font={"color": "#000000"}, tickfont={"color": "#000000"})
                fig.update_yaxes(title_font={"color": "#000000"}, tickfont={"color": "#000000"})
                chart_container(fig)

section_header("Perfil ReTHUS - Numero Residentes", "Imagenes de referencia")
santander_img = resolve_raw_image("BeneficiariosdelSNRMSantander.png")
valle_img = resolve_raw_image("BeneficiariosdelSNRMValle.png")

col_sant, col_valle = st.columns(2)
with col_sant:
    st.markdown("**Santander**")
    if santander_img is None:
        st.warning("No se encontro la imagen de Santander en data/raw.")
    else:
        st.image(str(santander_img), use_container_width=True)
with col_valle:
    st.markdown("**Valle del Cauca**")
    if valle_img is None:
        st.warning("No se encontro la imagen de Valle en data/raw.")
    else:
        st.image(str(valle_img), use_container_width=True)

