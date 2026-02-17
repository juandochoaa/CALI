from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

from dashboards.ui import (
    apply_theme,
    bullet_card,
    explain_box,
    page_header,
    section_header,
    text_card,
)

st.set_page_config(page_title="Ubicacion", layout="wide")
apply_theme()

page_header(
    "Ubicacion",
    "Contexto territorial y ventaja estrategica del proyecto en Cali.",
    "Territorio",
)

section_header("Ubicacion estrategica", "Sector Melendez y Comuna 18")
explain_box(
    "Como se calcula",
    [
        "Seccion descriptiva con datos demograficos y urbanisticos.",
        "No hay calculos financieros, solo contexto territorial.",
    ],
)

text_card(
    "Ubicacion propuesta",
    "Calle 5 con Carrera 95, sector Melendez, Comuna 18 (sur-occidente de Cali).",
)

bullet_card(
    "Proyecto urbanistico Ciudad Melendez",
    [
        "Plan parcial aprobado en 2010.",
        "Area total de planificacion: 152 ha.",
        "Area de desarrollo: 75 ha.",
        "Area de reserva: 54 ha.",
        "Cinturon ecologico: 23 ha.",
        "Ubicacion: entre calles 59 y 61, carreras 93 y 95.",
    ],
)

bullet_card(
    "Ventajas de la ubicacion",
    [
        "Desarrollo moderno con infraestructura planificada.",
        "Conexion al transporte publico masivo (MIO).",
        "Cercania a centros comerciales e instituciones educativas.",
        "Acceso vial directo a Calle 5 (eje principal este-oeste).",
    ],
)

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
section_header("Caracterizacion del sector")
st.dataframe(demo_df, width='stretch', hide_index=True)

strat_df = pd.DataFrame(
    {
        "Estrato": ["1", "2", "3 (Moda)", "4", "5 y 6", "TOTAL E2+E3"],
        "% Lados de Manzana": ["Minoritario", "~30%", "~43%", "1.2%", "0%", "72.9%"],
        "Observacion": ["Presente", "Significativo", "Predominante", "Marginal", "Ausentes", "Poblacion objetivo"],
    }
)
section_header("Estratificacion Comuna 18")
st.dataframe(strat_df, width='stretch', hide_index=True)
st.caption("Hallazgo: 72.9% de viviendas en estratos 2 y 3, alineado con el segmento objetivo.")

bullet_card(
    "Ventaja geografica y estrategica",
    [
        "Acceso zona oriente (comunas 13, 14, 15, 16, 21): 8-10 km, 15-25 min, poblacion potencial 620,000.",
        "Cobertura zona ladera (comunas 1, 18, 20): 340,000 habitantes.",
        "Conexion sur y centro-sur (comunas 17, 22, 11, 12): ~200,000 habitantes.",
        "Mercado total accesible: >1,160,000 habitantes estratos 1-2-3 en 10-15 km.",
    ],
)

comp_df = pd.DataFrame(
    {
        "Institucion": ["ICB Melendez", "Valle del Lili", "Imbanaco", "DIME", "HUV"],
        "Distancia a Oriente": ["8-10 km", "15-18 km", "12-15 km", "12-16 km", "10-12 km"],
        "Tiempo estimado": ["15-25 min", "30-45 min", "25-40 min", "25-40 min", "20-35 min"],
    }
)
section_header("Comparativo de acceso")
st.dataframe(comp_df, width='stretch', hide_index=True)
st.caption(
    "Ventaja competitiva: el ICB seria el centro especializado mas cercano a la zona con mayor concentracion de poblacion vulnerable."
)

