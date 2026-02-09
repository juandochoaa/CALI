from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


import plotly.express as px
import streamlit as st

from dashboards.data_loader import load_market
from dashboards.ui import apply_theme, chart_container, page_header, section_header, style_chart


st.set_page_config(page_title="Mercado & Share", layout="wide")
apply_theme()

page_header(
    "Mercado & Share",
    "TAM / SAM / SOM y participacion proyectada.",
    "Market sizing",
)

with st.sidebar:
    st.header("Supuestos")
    st.slider("Crecimiento mercado", 0.02, 0.12, 0.06)
    st.slider("Market share objetivo", 0.02, 0.12, 0.08)
    st.caption("Archivo: data/raw/mercado.xlsx")

market, market_source = load_market()

section_header("KPIs TAM / SAM / SOM")

if market.empty:
    st.warning("No hay datos de mercado. Carga `mercado.xlsx` en `data/raw/`.")
else:
    required_cols = {"year", "tam_cop_bn", "sam_cop_bn", "som_cop_bn", "share"}
    if not required_cols.issubset(set(market.columns)):
        st.warning("La hoja de mercado no tiene las columnas requeridas.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("TAM 2030 (COP bn)", f"{market.loc[market['year'] == 2030, 'tam_cop_bn'].iat[0]:.0f}")
        col2.metric("SAM 2030 (COP bn)", f"{market.loc[market['year'] == 2030, 'sam_cop_bn'].iat[0]:.0f}")
        col3.metric("SOM 2030 (COP bn)", f"{market.loc[market['year'] == 2030, 'som_cop_bn'].iat[0]:.1f}")
        col4.metric("Share 2030", f"{market.loc[market['year'] == 2030, 'share'].iat[0]:.1%}")

        left, right = st.columns([1.1, 0.9])

        with left:
            funnel = market.loc[market['year'] == 2030, ["tam_cop_bn", "sam_cop_bn", "som_cop_bn"]].T
            funnel = funnel.reset_index()
            funnel.columns = ["segment", "value"]
            fig = px.funnel(
                funnel,
                x="value",
                y="segment",
                title="Funnel TAM -> SAM -> SOM (COP bn)",
            )
            fig = style_chart(fig)
            chart_container(fig)
            st.caption(f"Fuente: {market_source}")

        with right:
            fig = px.line(
                market,
                x="year",
                y="share",
                markers=True,
                title="Participacion proyectada",
            )
            fig.update_yaxes(tickformat=".0%")
            fig = style_chart(fig)
            chart_container(fig)
            st.caption(f"Fuente: {market_source}")

        section_header("Serie TAM / SAM / SOM")

        stack = market.melt(
            id_vars="year",
            value_vars=["tam_cop_bn", "sam_cop_bn", "som_cop_bn"],
            var_name="segment",
            value_name="value",
        )
        fig = px.area(
            stack,
            x="year",
            y="value",
            color="segment",
            title="Evolucion TAM / SAM / SOM (COP bn)",
        )
        fig = style_chart(fig)
        chart_container(fig)
        st.caption(f"Fuente: {market_source}")

        section_header("Detalle anual")
        st.dataframe(market, use_container_width=True)
