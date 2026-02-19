from __future__ import annotations

import base64
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800&display=swap');

:root {
  --bg-1: #ffffff;
  --bg-2: #f7f9fc;
  --ink: #0b1f3b;
  --ink-soft: #345072;
  --accent-blue: #0067b8;
  --accent-red: #d62839;
  --panel: #ffffff;
  --panel-2: rgba(255, 255, 255, 0.98);
  --stroke: rgba(0, 103, 184, 0.22);
  --grid: rgba(11, 31, 59, 0.10);
  --shadow: 0 10px 24px rgba(11, 31, 59, 0.10);
}

html, body, [class*="css"] {
  font-family: 'Nunito', sans-serif;
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
    radial-gradient(1100px 540px at 10% -10%, rgba(0, 103, 184, 0.08), transparent 60%),
    radial-gradient(800px 440px at 100% 0%, rgba(214, 40, 57, 0.08), transparent 55%),
    linear-gradient(180deg, var(--bg-1), var(--bg-2));
}

h1, h2, h3, h4 {
  font-family: 'Constantia', 'Times New Roman', serif;
  letter-spacing: -0.02em;
}

section[data-testid="stSidebar"] {
  background: #ffffff;
  border-right: 1px solid rgba(0, 103, 184, 0.22);
}

.sidebar-brand {
  padding: 8px 0 14px 0;
  margin-bottom: 10px;
  border-bottom: 2px solid rgba(0, 103, 184, 0.18);
}

.sidebar-brand img {
  width: 180px;
  max-width: 100%;
  height: auto;
  display: block;
}

.sidebar-brand-fallback {
  font-family: 'Constantia', 'Times New Roman', serif;
  font-size: 32px;
  font-weight: 700;
  color: var(--accent-blue);
  letter-spacing: 0.02em;
  margin-bottom: 10px;
  padding-bottom: 8px;
  border-bottom: 2px solid rgba(0, 103, 184, 0.18);
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
  color: var(--accent-blue);
  margin-bottom: 6px;
}

.hero .title {
  font-size: 36px;
  font-weight: 700;
  margin-bottom: 8px;
}

.hero .subtitle {
  font-size: 16px;
  color: var(--ink-soft);
  margin-bottom: 0;
}

.section-title {
  font-size: 20px;
  font-weight: 700;
  margin: 12px 0 6px 0;
}

.section-caption {
  color: var(--ink-soft);
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
  color: var(--accent-blue);
  margin-bottom: 6px;
}

div[data-testid="stMetric"] {
  background: var(--panel);
  border: 1px solid var(--stroke);
  padding: 14px 16px;
  border-radius: 18px;
  box-shadow: var(--shadow);
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
  box-shadow: var(--shadow);
}

.text-card {
  background: var(--panel-2);
  border: 1px solid var(--stroke);
  border-radius: 16px;
  padding: 14px 16px;
  box-shadow: 0 8px 20px rgba(11, 31, 59, 0.08);
  margin-bottom: 12px;
}

.text-card .card-title {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.16em;
  color: var(--accent-blue);
  margin-bottom: 6px;
}

.text-card .card-body {
  font-size: 14px;
  line-height: 1.45;
  color: var(--ink-soft);
}

.text-card ul {
  margin: 0;
  padding-left: 16px;
}

.text-card li {
  margin-bottom: 5px;
}

.takeaway {
  background: linear-gradient(120deg, rgba(0, 103, 184, 0.09), rgba(214, 40, 57, 0.09));
  border: 1px solid var(--stroke);
  border-radius: 16px;
  padding: 14px 16px;
  margin: 8px 0 14px 0;
}

.takeaway .takeaway-title {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.16em;
  color: var(--ink);
  margin-bottom: 6px;
}

.takeaway .takeaway-body {
  font-size: 14px;
  line-height: 1.45;
  color: var(--ink-soft);
}

.view-label {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.16em;
  color: var(--accent-blue);
  margin: 4px 0 4px 0;
}

div[data-testid="stRadio"] > div {
  background: var(--panel-2);
  border: 1px solid var(--stroke);
  border-radius: 12px;
  padding: 4px 8px;
}
</style>
"""


def _logo_data_uri() -> str | None:
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "data" / "raw" / "icblogonuevoformal-2.png",
        root / "data" / "raw" / "icb_logo.png",
        root / "dashboards" / "assets" / "icb_logo.png",
        root / "dashboards" / "assets" / "icb_logo.svg",
    ]
    for logo_path in candidates:
        if logo_path.exists():
            raw = logo_path.read_bytes()
            suffix = logo_path.suffix.lower()
            if suffix == ".svg":
                mime = "image/svg+xml"
            elif suffix == ".png":
                mime = "image/png"
            elif suffix in {".jpg", ".jpeg"}:
                mime = "image/jpeg"
            elif suffix == ".webp":
                mime = "image/webp"
            else:
                continue
            encoded = base64.b64encode(raw).decode("ascii")
            return f"data:{mime};base64,{encoded}"
    return None


def _render_sidebar_brand() -> None:
    logo_uri = _logo_data_uri()
    if logo_uri:
        st.sidebar.markdown(
            f"<div class='sidebar-brand'><img src='{logo_uri}' alt='ICB logo' /></div>",
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown("<div class='sidebar-brand-fallback'>ICB</div>", unsafe_allow_html=True)


def apply_theme() -> None:
    st.markdown(_THEME_CSS, unsafe_allow_html=True)
    _render_sidebar_brand()


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
    chart_text_color = "#000000"
    fig.update_layout(
        template="simple_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Nunito", color=chart_text_color),
        margin=dict(l=18, r=18, t=46, b=18),
        title_font=dict(size=16, family="Constantia", color=chart_text_color),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=chart_text_color),
            title=dict(font=dict(color=chart_text_color)),
        ),
        colorway=["#0067b8", "#d62839", "#0b1f3b", "#2a6f97", "#a4161a"],
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(11,31,59,0.08)",
        tickfont=dict(color=chart_text_color),
        title_font=dict(color=chart_text_color),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(11,31,59,0.08)",
        tickfont=dict(color=chart_text_color),
        title_font=dict(color=chart_text_color),
    )
    fig.update_annotations(font=dict(color=chart_text_color))
    return fig


def chart_container(fig, use_container_width: bool = True) -> None:
    st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
    chart_width = "stretch" if use_container_width else "content"
    st.plotly_chart(fig, width=chart_width)
    st.markdown("</div>", unsafe_allow_html=True)


def text_card(title: str, body: str) -> None:
    safe_title = escape(title)
    safe_body = escape(body).replace("\n", "<br>")
    st.markdown(
        (
            "<div class='text-card'>"
            f"<div class='card-title'>{safe_title}</div>"
            f"<div class='card-body'>{safe_body}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def bullet_card(title: str, bullets: list[str]) -> None:
    if not bullets:
        return
    safe_title = escape(title)
    items = "".join([f"<li>{escape(item)}</li>" for item in bullets])
    st.markdown(
        (
            "<div class='text-card'>"
            f"<div class='card-title'>{safe_title}</div>"
            f"<div class='card-body'><ul>{items}</ul></div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def insight_cards(items: list[tuple[str, str]], columns: int = 3) -> None:
    if not items:
        return
    col_count = max(1, min(columns, len(items)))
    cols = st.columns(col_count)
    for idx, (title, body) in enumerate(items):
        with cols[idx % col_count]:
            text_card(title, body)


def takeaway_box(title: str, body: str) -> None:
    st.markdown(
        (
            "<div class='takeaway'>"
            f"<div class='takeaway-title'>{escape(title)}</div>"
            f"<div class='takeaway-body'>{escape(body).replace('\\n', '<br>')}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def subsection_selector(options: list[str], key: str, label: str = "Vista") -> str:
    st.markdown(f"<div class='view-label'>{escape(label)}</div>", unsafe_allow_html=True)
    return st.radio(label, options, index=0, horizontal=True, key=key, label_visibility="collapsed")


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
