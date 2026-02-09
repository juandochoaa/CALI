from __future__ import annotations

import pandas as pd
import streamlit as st


_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Newsreader:wght@400;600&display=swap');

:root {
  --bg-1: #f7f2ea;
  --bg-2: #f0e6d8;
  --ink: #0a1414;
  --ink-soft: #243737;
  --accent: #c25416;
  --accent-2: #0f6a62;
  --panel: #fffdf8;
  --panel-2: rgba(255, 255, 255, 0.92);
  --stroke: rgba(11, 31, 31, 0.22);
  --grid: rgba(11, 31, 31, 0.14);
  --shadow: 0 16px 36px rgba(11, 31, 31, 0.12);
}

html, body, [class*="css"] {
  font-family: 'Space Grotesk', sans-serif;
  color: var(--ink);
}

.stApp,
.stApp p,
.stApp span,
.stApp label,
.stApp li,
.stApp div {
  color: var(--ink);
}

.stApp .stCaption,
.stApp .stMarkdown small {
  color: var(--ink-soft);
}

.stApp {
  background:
    radial-gradient(1200px 600px at 10% -10%, rgba(217, 107, 43, 0.18), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(46, 138, 124, 0.16), transparent 55%),
    linear-gradient(180deg, var(--bg-1), var(--bg-2));
}

h1, h2, h3, h4 {
  font-family: 'Newsreader', serif;
  letter-spacing: -0.02em;
}

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #efe3d0 0%, #e6d6c0 100%);
  border-right: 1px solid rgba(11, 31, 31, 0.28);
}

.hero {
  padding: 22px 26px;
  border-radius: 22px;
  background: linear-gradient(120deg, rgba(255, 255, 255, 0.94), rgba(255, 255, 255, 0.78));
  border: 1px solid var(--stroke);
  box-shadow: var(--shadow);
  margin-bottom: 18px;
}

.hero .kicker {
  text-transform: uppercase;
  font-size: 12px;
  letter-spacing: 0.22em;
  color: var(--accent-2);
  margin-bottom: 6px;
}

.hero .title {
  font-size: 36px;
  font-weight: 600;
  margin-bottom: 8px;
}

.hero .subtitle {
  font-size: 16px;
  color: var(--ink-soft);
  margin-bottom: 0;
}

.section-title {
  font-size: 20px;
  font-weight: 600;
  margin: 12px 0 6px 0;
}

.section-caption {
  color: #2e3f3f;
  font-size: 13px;
  margin-bottom: 10px;
}

.divider {
  height: 1px;
  background: var(--grid);
  margin: 14px 0 18px 0;
}

.metric-label {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.2em;
  color: var(--accent-2);
  margin-bottom: 6px;
}

div[data-testid="stMetric"] {
  background: var(--panel);
  border: 1px solid var(--stroke);
  padding: 14px 16px;
  border-radius: 18px;
  box-shadow: 0 10px 24px rgba(11, 31, 31, 0.12);
}

div[data-testid="stMetric"] > label {
  color: var(--ink-soft);
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.2em;
}

.plot-container {
  background: var(--panel-2);
  border: 1px solid var(--stroke);
  border-radius: 18px;
  padding: 12px 12px 6px 12px;
  box-shadow: 0 12px 26px rgba(11, 31, 31, 0.1);
}
</style>
"""


def apply_theme() -> None:
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str | None = None, kicker: str | None = None) -> None:
    parts = []
    if kicker:
        parts.append(f"<div class='kicker'>{kicker}</div>")
    parts.append(f"<div class='title'>{title}</div>")
    if subtitle:
        parts.append(f"<div class='subtitle'>{subtitle}</div>")
    body = "".join(parts)
    st.markdown(f"<div class='hero'>{body}</div>", unsafe_allow_html=True)


def section_header(title: str, caption: str | None = None) -> None:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if caption:
        st.markdown(f"<div class='section-caption'>{caption}</div>", unsafe_allow_html=True)


def divider() -> None:
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


def style_chart(fig):
    fig.update_layout(
        template="simple_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk", color="#000000"),
        margin=dict(l=18, r=18, t=46, b=18),
        title_font=dict(size=16, family="Newsreader"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        colorway=["#0f6a62", "#c25416", "#2f4858", "#7a4e8a", "#4c7a7a"],
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(15,45,46,0.08)",
        tickfont=dict(color="#000000"),
        title_font=dict(color="#000000"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(15,45,46,0.08)",
        tickfont=dict(color="#000000"),
        title_font=dict(color="#000000"),
    )
    return fig


def chart_container(fig, use_container_width: bool = True) -> None:
    st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=use_container_width)
    st.markdown("</div>", unsafe_allow_html=True)


def explain_box(title: str, bullets: list[str]) -> None:
    if not bullets:
        return
    with st.expander(title):
        st.markdown("\n".join([f"- {item}" for item in bullets]))


def append_total_row(
    df: pd.DataFrame,
    label_col: str,
    numeric_cols: list[str],
    label: str = "TOTAL",
) -> pd.DataFrame:
    if df.empty:
        return df
    totals = {label_col: label}
    for col in numeric_cols:
        if col in df.columns:
            totals[col] = pd.to_numeric(df[col], errors="coerce").sum(min_count=1)
    return pd.concat([df, pd.DataFrame([totals])], ignore_index=True)


def append_avg_column(df: pd.DataFrame, year_cols: list[str], label: str = "Promedio") -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    cols = [c for c in year_cols if c in df.columns]
    if not cols:
        return df
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    df[label] = numeric.mean(axis=1, skipna=True)
    return df
