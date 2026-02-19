from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


import unicodedata
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboards.data_loader import load_cifras_eps, load_eps_financials
from dashboards.ui import (
    append_avg_column,
    append_total_row,
    apply_theme,
    bullet_card,
    chart_container,
    divider,
    explain_box,
    page_header,
    section_header,
    style_chart,
    subsection_selector,
    text_card,
)
from src.models.eps_scoring import (
    BLOCK_DEFS,
    RATIO_SPECS,
    aggregate_scores,
    apply_size_factors,
    build_blocks_long,
    compute_net_income_factor,
    compute_ratios,
    compute_revenue_factor,
    score_ratios,
    winsorize_ratios,
)
from src.models.ips_kpis import build_ips_historical_kpis

st.set_page_config(page_title="Competencia", layout="wide")
apply_theme()

PAGE_ID = "competencia"
DEFAULT_WEIGHTS = {
    "liquidity": 30,
    "solvency": 30,
    "profitability": 20,
    "efficiency": 20,
}
if st.session_state.get("active_page") != PAGE_ID:
    st.session_state["active_page"] = PAGE_ID
    st.session_state["competencia_weight_liquidity"] = DEFAULT_WEIGHTS["liquidity"]
    st.session_state["competencia_weight_solvency"] = DEFAULT_WEIGHTS["solvency"]
    st.session_state["competencia_weight_profitability"] = DEFAULT_WEIGHTS["profitability"]
    st.session_state["competencia_weight_efficiency"] = DEFAULT_WEIGHTS["efficiency"]

page_header(
    "Competencia",
    "Análisis financiero de IPS en Cali.",
    "IPS insights",
)

with st.sidebar:
    st.header("Filters")
    st.caption("Archivo: Cali ANALISIS.xlsx (hoja IPS_EEFF)")
    st.subheader("Pesos del Score Financiero")
    w_liquidity = st.slider(
        "Liquidez (%)",
        0,
        100,
        key="competencia_weight_liquidity",
    )
    w_solvency = st.slider(
        "Endeudamiento (%)",
        0,
        100,
        key="competencia_weight_solvency",
    )
    w_profitability = st.slider(
        "Rentabilidad (%)",
        0,
        100,
        key="competencia_weight_profitability",
    )
    w_efficiency = st.slider(
        "Eficiencia (%)",
        0,
        100,
        key="competencia_weight_efficiency",
    )


def normalize_account(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = " ".join(normalized.lower().split())
    return normalized


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return " ".join(normalized.upper().split())


def find_col(columns: List[str], includes: List[str]) -> str | None:
    tokens = [normalize_text(token) for token in includes]
    for col in columns:
        norm = normalize_text(col)
        if all(token in norm for token in tokens):
            return col
    return None


def accounts_used(df: pd.DataFrame, accounts: List[str]) -> str:
    if "CUENTA_norm" not in df.columns:
        return "No disponible"
    account_set = set(df["CUENTA_norm"].dropna())
    if accounts == ["Ingresos netos por ventas", "Total Ingreso Operativo"]:
        primary = normalize_account("Ingresos netos por ventas")
        secondary = normalize_account("Total Ingreso Operativo")
        if primary in account_set:
            used = ["Ingresos netos por ventas"]
        elif secondary in account_set:
            used = ["Total Ingreso Operativo"]
        else:
            used = []
    else:
        used = [acc for acc in accounts if normalize_account(acc) in account_set]
    return " + ".join(used) if used else "No disponible"


def fmt_currency(value: float | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "NA"
    return f"${value:,.0f}"


def fmt_ratio(value: float | None, kind: str) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "NA"
    if kind == "percent":
        return f"{value * 100:.1f}%"
    if kind == "days":
        return f"{value:,.0f} dias"
    if kind == "x":
        return f"{value:.2f}x"
    return f"{value:,.2f}"


def split_financial_statements(df_ips: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df_ips.copy()
    work["_row"] = work.index
    if "CUENTA_norm" not in work.columns and "CUENTA" in work.columns:
        work["CUENTA_norm"] = work["CUENTA"].astype(str).map(normalize_account)
    if "CUENTA_norm" not in work.columns:
        return work, work.iloc[0:0]

    balance_keys = [
        "activos totales",
        "activos corrientes",
        "pasivos totales",
        "total de patrimonio",
        "total de patrimonio y pasivos",
    ]
    pattern = "|".join(balance_keys)
    mask_balance = work["CUENTA_norm"].str.contains(pattern, na=False)
    if mask_balance.any():
        split_row = work.loc[mask_balance, "_row"].min()
        estado = work[work["_row"] < split_row]
        balance = work[work["_row"] >= split_row]
    else:
        estado = work
        balance = work.iloc[0:0]

    estado = estado.sort_values("_row")
    balance = balance.sort_values("_row")
    return estado, balance


def total_ingreso_operativo(df: pd.DataFrame, year_cols: List[str]) -> pd.DataFrame:
    if "CUENTA" not in df.columns:
        return pd.DataFrame(columns=["entity", "year", "REV"])
    work = df[["EPS_clean", "CUENTA"] + year_cols].copy()
    work["account_norm"] = work["CUENTA"].map(normalize_account)
    income_accounts = ["ingresos netos por ventas", "total ingreso operativo"]
    work = work[work["account_norm"].isin(income_accounts)]
    if work.empty:
        return pd.DataFrame(columns=["entity", "year", "REV"])
    long_df = work.melt(
        id_vars=["EPS_clean", "account_norm"],
        value_vars=year_cols,
        var_name="year",
        value_name="REV",
    )
    long_df["REV"] = pd.to_numeric(long_df["REV"], errors="coerce")
    long_df["year"] = long_df["year"].astype(str).str.strip().astype(int)
    grouped = (
        long_df.groupby(["EPS_clean", "year", "account_norm"], dropna=False)["REV"]
        .sum(min_count=1)
        .reset_index()
    )
    order_map = {acc: idx for idx, acc in enumerate(income_accounts)}
    grouped["order"] = grouped["account_norm"].map(order_map)
    grouped = (
        grouped.sort_values("order")
        .groupby(["EPS_clean", "year"], dropna=False, as_index=False)
        .first()
    )
    grouped = grouped.rename(columns={"EPS_clean": "entity"})
    return grouped[["entity", "year", "REV"]]


def blocks_table(selected: str) -> pd.DataFrame:
    df_ips = ips_df[ips_df["EPS_clean"] == selected]
    rows = []
    for code, accounts, _, _ in BLOCK_DEFS:
        row = {"Bloque": code, "Cuenta usada": accounts_used(df_ips, accounts)}
        subset = blocks_df[(blocks_df["entity"] == selected) & (blocks_df["year"].isin([int(y) for y in year_cols]))]
        for year in year_cols:
            value = subset.loc[subset["year"] == int(year), code]
            row[str(year)] = fmt_currency(value.iloc[0] if not value.empty else None)
        rows.append(row)
    return pd.DataFrame(rows)


ips_df, ips_source = load_eps_financials("IPS_EEFF")
servicios_matrix_df, servicios_matrix_source = load_cifras_eps("Servicios")
capacidad_df, capacidad_source = load_cifras_eps("Capacidad")
if capacidad_df.empty:
    alt_capacidad_df, alt_capacidad_source = load_cifras_eps("CAPACIDAD")
    if not alt_capacidad_df.empty:
        capacidad_df, capacidad_source = alt_capacidad_df, alt_capacidad_source
tarifas_comp_df, tarifas_comp_source = load_cifras_eps("TarifasCompetencia")
if ips_df.empty:
    st.warning("No se encontro el archivo de estados financieros IPS o no se pudo leer.")
    st.caption(f"Detalle: {ips_source}")
    st.stop()

year_cols = [c for c in ips_df.columns if str(c).strip().isdigit()]
year_cols = sorted(year_cols, key=lambda x: int(x))
if not year_cols:
    st.warning("No se encontraron columnas de anos en la hoja IPS_EEFF.")
    st.stop()

if "IPS" in ips_df.columns and "EPS_clean" not in ips_df.columns:
    ips_df["EPS_clean"] = ips_df["IPS"].astype(str).str.replace(".xlsx", "", regex=False).str.strip()

if "CUENTA" in ips_df.columns:
    ips_df["CUENTA_norm"] = ips_df["CUENTA"].astype(str).map(normalize_account)

ips_list = sorted(ips_df["EPS_clean"].dropna().unique().tolist())

blocks_df = build_blocks_long(ips_df, year_cols)
ing_oper_df = total_ingreso_operativo(ips_df, year_cols)

blocks_df["net_income_factor"] = compute_net_income_factor(blocks_df)
blocks_df["revenue_factor"] = compute_revenue_factor(blocks_df)
ratios_df = compute_ratios(blocks_df)
ratio_cols = list(RATIO_SPECS.keys())
wins_df = winsorize_ratios(ratios_df, ratio_cols, lower=0.02, upper=0.98)
scored_df = score_ratios(wins_df, RATIO_SPECS, dpo_range=(20, 60), dpo_zero=(0, 120))
weights_input = {
    "liquidity": w_liquidity / 100,
    "solvency": w_solvency / 100,
    "profitability": w_profitability / 100,
    "efficiency": w_efficiency / 100,
}
weight_sum = sum(weights_input.values())
used_default_weights = False
if weight_sum > 0:
    weights = {k: v / weight_sum for k, v in weights_input.items()}
else:
    default_sum = sum(DEFAULT_WEIGHTS.values())
    weights = {k: v / default_sum for k, v in DEFAULT_WEIGHTS.items()}
    used_default_weights = True
scored_df = aggregate_scores(scored_df, RATIO_SPECS, weights)
factor_cols = ["net_income_factor", "revenue_factor"]
if not all(col in scored_df.columns for col in factor_cols):
    scored_df = scored_df.merge(
        blocks_df[["entity", "year"] + factor_cols],
        on=["entity", "year"],
        how="left",
    )
scored_df = apply_size_factors(
    scored_df, weights, factor_cols=["net_income_factor", "revenue_factor"]
)


def score_table(selected: str) -> pd.DataFrame:
    subset = scored_df[scored_df["entity"] == selected]
    rows = []
    mapping = [
        ("Liquidez", "liquidity_score"),
        ("Endeudamiento", "solvency_score"),
        ("Rentabilidad", "profitability_score"),
        ("Eficiencia", "efficiency_score"),
        ("Score Financiero", "score_financiero"),
    ]
    for label, col in mapping:
        row = {"Score": label}
        for year in year_cols:
            value = subset.loc[subset["year"] == int(year), col]
            row[str(year)] = round(value.iloc[0], 1) if not value.empty and pd.notna(value.iloc[0]) else None
        rows.append(row)
    df = pd.DataFrame(rows)
    df = append_avg_column(df, [str(y) for y in year_cols], label="Promedio")
    return df


ratio_labels = {
    "current_ratio": "Razón corriente",
    "cash_ratio": "Razón de caja",
    "wc_to_rev": "Capital de trabajo neto / REV",
    "days_cash": "Días de caja (proxy)",
    "current_assets_ratio": "Activos corrientes / Activos totales",
    "debt_to_assets": "Deuda total / Activos",
    "equity_ratio": "Patrimonio / Activos",
    "assets_to_liabilities": "Activos / Pasivos totales",
    "current_liab_share": "Peso del corto plazo",
    "net_debt_to_ebitda": "Deuda neta CP / EBITDA",
    "gross_margin": "Margen bruto",
    "ebitda_margin": "Margen EBITDA",
    "ebit_margin": "Margen operativo (EBIT)",
    "net_margin": "Margen neto",
    "roa": "ROA",
    "asset_turnover": "Rotación de activos",
    "dso": "DSO (días de cartera)",
    "dpo": "DPO (días de proveedores)",
    "opex_cash_ratio": "Índice OPEX en efectivo",
    "da_intensity": "Intensidad dep/amort",
}

ratio_kinds = {
    "current_ratio": "x",
    "cash_ratio": "x",
    "wc_to_rev": "percent",
    "days_cash": "days",
    "current_assets_ratio": "percent",
    "debt_to_assets": "percent",
    "equity_ratio": "percent",
    "assets_to_liabilities": "x",
    "current_liab_share": "percent",
    "net_debt_to_ebitda": "x",
    "gross_margin": "percent",
    "ebitda_margin": "percent",
    "ebit_margin": "percent",
    "net_margin": "percent",
    "roa": "percent",
    "asset_turnover": "x",
    "dso": "days",
    "dpo": "days",
    "opex_cash_ratio": "percent",
    "da_intensity": "percent",
}

liquidity_ratios = ["current_ratio", "cash_ratio", "wc_to_rev", "days_cash", "current_assets_ratio"]
solvency_ratios = ["debt_to_assets", "equity_ratio", "assets_to_liabilities", "current_liab_share", "net_debt_to_ebitda"]
profitability_ratios = ["gross_margin", "ebitda_margin", "ebit_margin", "net_margin", "roa"]
efficiency_ratios = ["asset_turnover", "dso", "dpo", "opex_cash_ratio", "da_intensity"]


def ratio_table(selected: str, ratio_list: List[str]) -> pd.DataFrame:

    subset = ratios_df[ratios_df["entity"] == selected]
    rows = []
    for ratio in ratio_list:
        row = {"Indicador": ratio_labels.get(ratio, ratio)}
        numeric_vals: List[float] = []
        for year in year_cols:
            value = subset.loc[subset["year"] == int(year), ratio]
            curr = value.iloc[0] if not value.empty else None
            row[str(year)] = fmt_ratio(curr, ratio_kinds.get(ratio, "ratio"))
            if curr is not None and pd.notna(curr):
                numeric_vals.append(float(curr))
        avg_val = np.mean(numeric_vals) if numeric_vals else np.nan
        row["Promedio"] = fmt_ratio(avg_val, ratio_kinds.get(ratio, "ratio"))
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def norm_key(text: str) -> str:
    return normalize_text(text).lower().strip()


def qualitative_profile(selected_ips: str) -> dict | None:
    key = norm_key(selected_ips)

    if "angiografia de occidente" in key:
        return {
            "title": "Angiografia de Occidente S.A.",
            "sections": [
                (
                    "Participaciones / vinculos accionarios",
                    [
                        "Senal de vinculo accionario con Clinica de Occidente: en fuentes periodisticas referidas se menciona que Angiografia de Occidente habria sido mayor accionista de Clinica de Occidente (contexto 2016).",
                    ],
                ),
                (
                    "Dotacion / tecnologia (marca/modelo)",
                    [
                        "Marcas: MEDTRONIC, BAYER, ST JUDE MEDICAL, ABBOTT, CTP MEDICAS, ALLERS, BOSTON SCIENTIFIC y TERUMO.",
                        "Compras realizadas a estas 8 empresas en 2024: $40.146 millones.",
                    ],
                ),
                (
                    "Unidades de Cardiologia No Invasiva",
                    [
                        "Ecocardiograma Modo M, Bidimensional y Doppler Color.",
                        "Ecocardiograma de Estres con Ejercicio o Suministro de Farmaco.",
                        "Ecocardiograma Transesofagico.",
                        "Prueba Ergometrica o Test de Ejercicio.",
                        "Electrocardiografia Dinamica 24 Horas (Test de Holter).",
                        "Monitoria Ambulatoria de Presion Arterial (MAPA).",
                        "Electrocardiograma de Ritmo o de Superficie SOD.",
                    ],
                ),
                (
                    "Unidad de consulta externa",
                    [
                        "Cardiologia Adulto.",
                        "Cardiologia Pediatrica.",
                        "Programa de Falla Cardiaca y Anticoagulados.",
                        "Hemodinamia.",
                        "Electrofisiologia.",
                        "Reprogramacion de Marcapasos.",
                        "Cardiologia no Invasiva.",
                        "Neuroradiologia.",
                        "Radiologia Intervencionista.",
                        "Cirugia Vascular periferico.",
                        "Riesgo Metabolico.",
                        "Medicina Interna.",
                        "Medicina General.",
                    ],
                ),
                (
                    "Unidades IPS",
                    [
                        "Cardiologia.",
                        "Cirugia General.",
                        "Cirugia Vascular Periferica.",
                        "Citologias.",
                        "Dermatologia.",
                        "Enfermeria.",
                        "Fisioterapeuta.",
                        "Gastroenterologia.",
                        "Ginecologia y Obstetricia.",
                        "Higiene Oral.",
                        "Medicina Fisica y Rehabilitacion.",
                        "Medicina General.",
                        "Medicina Interna.",
                        "Nefrologia.",
                        "Neumologia.",
                        "Neurologia.",
                        "Nutricionista.",
                        "Obstetricia.",
                        "Odontologia.",
                        "Ortopedia y Traumatologia.",
                        "Otorrinolaringologia.",
                        "P&P.",
                        "Pediatria.",
                        "Programas especiales (Diabetes, Hipertension, EPOC, Insuficiencia Renal Cronica).",
                        "Psicologia.",
                        "Trabajo Social.",
                        "Urologia.",
                        "Vacunacion.",
                    ],
                ),
                (
                    "Contratos mencionados",
                    [
                        "Cirugia Vascular: Contrato IPS Nueva EPS; Contrato PGP Coosalud; consultas por evento con entidades varias.",
                        "Cardiologia - Falla Cardiaca y Anticoagulacion: Contrato IPS Nueva EPS; Contrato PGP Coosalud; Contrato VIVA 1A para poblacion Cali, Jamundi y Yumbo; consultas por evento con entidades varias.",
                        "Hemodinamia: Contrato PGP Nueva EPS; Contrato PGP Coosalud; consultas por evento con entidades varias.",
                        "Electrofisiologia: revision de marcapaso; Contrato PGP Nueva EPS; Contrato PGP Coosalud; consultas por evento con entidades varias.",
                        "Neuroradiologia: Contrato PGP Coosalud; consultas por evento con entidades varias.",
                        "Radiologia Intervencionista: Contrato PGP Coosalud; consultas por evento con entidades varias.",
                    ],
                ),
                (
                    "Unidades de Angiografia - Hemodinamia",
                    [
                        "Cateterismos cardiacos izquierdos y derechos.",
                        "Arteriografias coronarias.",
                        "Angioplastias coronarias.",
                        "Cierre percutaneo de defectos cardiacos septales.",
                        "TAVI - Implante valvular aortico percutaneo.",
                        "Valvuloplastia aortica, pulmonar y mitral.",
                        "Implante de balon de contrapulsacion intraaortico percutaneo.",
                        "Reparo de valvula mitral con dispositivo MitraClip por via endovascular.",
                        "Cierre de auriculilla izquierda con dispositivo endovascular.",
                        "Tomografia optica coherente intravascular (TOC).",
                        "Cierre de fuga paravalvular aortica o mitral por via endovascular.",
                        "Implante valvular pulmonar.",
                        "Denervacion renal por via endovascular.",
                        "Pericardiocentesis terapeutica.",
                        "Ablacion septal con alcohol.",
                        "Ultrasonido intravascular intracoronario (IVUS).",
                        "Implante valvular mitral percutaneo.",
                        "Reparo de coartacion de aorta por via endovascular.",
                        "Tromboembolectomia intracoronaria.",
                        "Trombectomia coronaria.",
                    ],
                ),
                (
                    "Unidades de Angiografia - Hemodinamia Pediatrica",
                    [
                        "Cateterismo cardiaco derecho e izquierdo.",
                        "Valvuloplastia aortica, pulmonar y mitral con balon.",
                        "Cierre percutaneo de defectos cardiacos septales.",
                        "Correccion de coartacion de aorta.",
                        "Embolizacion de colaterales aorto-pulmonares.",
                    ],
                ),
                (
                    "Unidades de Angiografia - Vascular Periferico y Radiologia Intervencionista",
                    [
                        "Trombectomia de vasos arteriales o venosos.",
                        "Tromboembolectomia periferica.",
                        "Venografia - Cavografia - Flebografia.",
                        "Aortograma abdominal.",
                        "Arteriografia de miembros inferiores/superiores, renal, bronquial o mesenterica.",
                        "Oclusiones perifericas.",
                        "Angioplastia periferica.",
                        "Implante de filtro en vena cava.",
                        "Quimioembolizacion hepatica.",
                        "Implante de cateter de alto flujo para dialisis.",
                        "Implante de cateter de nefrostomia.",
                        "Implante de cateter venoso central.",
                        "Implante de endoprotesis toracica y/o abdominal y/o fenestrada.",
                        "Embolizacion de tumores.",
                        "Trombectomia pulmonar.",
                    ],
                ),
                (
                    "Unidades de Angiografia - Electrofisiologia",
                    [
                        "Extraccion de electrodo de estimulacion, desfibrilacion y/o seno coronario.",
                        "Extraccion de cuerpo extrano.",
                        "Implante de marcapasos unicameral definitivo.",
                        "Implante de marcapasos bicameral definitivo.",
                        "Implante de marcapasos resincronizador definitivo.",
                        "Implante de cardiodesfibrilador definitivo transvenoso.",
                        "Estudio electroanatomico con mapeo no fluoroscopico y ablacion con cateter irrigado (tecnologia EnSite).",
                        "Cambio de electrodo de marcapasos.",
                        "Implante de monitor de eventos.",
                        "Explante de monitor de eventos.",
                        "Estudio electrofisiologico diagnostico.",
                        "Estudio electrofisiologico completo con mapeo y ablacion por radiofrecuencia.",
                        "Cardioversion electrica.",
                        "Implante de marcapasos transitorio.",
                    ],
                ),
                (
                    "Unidades de Angiografia - Neuroradiologia",
                    [
                        "Angiografia cerebral o espinal.",
                        "Cayado aortico.",
                        "Angiografia de vasos de cuello.",
                        "Angioplastia carotida-vertebral.",
                        "Angioplastia quimica.",
                        "Oclusiones vertebrales y cerebrales.",
                        "Trombolisis vertebrales.",
                        "Trombectomia de vasos intracraneales y de cabeza/cuello.",
                    ],
                ),
                (
                    "EPS",
                    [
                        "Septiembre 2025: cierre/suspension temporal y reapertura en contexto de cartera con Nueva EPS, con participacion de actores institucionales en mesas de trabajo (segun fuentes referidas).",
                    ],
                ),
            ],
            "income_mix_2024": [
                {"Cliente": "Nueva EPS", "Participacion": 65.0},
                {"Cliente": "Coosalud", "Participacion": 8.8},
                {"Cliente": "SOS", "Participacion": 4.0},
                {"Cliente": "Clinica Versalles", "Participacion": 4.0},
                {"Cliente": "Clinica Farallones", "Participacion": 3.2},
                {"Cliente": "Otros clientes", "Participacion": 14.9},
            ],
            "references": [
                {
                    "label": "Informe de Gestion y Sostenibilidad 2024 (PDF)",
                    "url": "file:///C:/Users/analistagerencia/Downloads/INFORME%20DE%20GESTI%C3%93N%20Y%20SOSTENIBILIDAD%20ANGIOGRAF%C3%8DA%20DE%20OCCIDENTE%202024%20-%20ADO.pdf",
                },
            ],
        }

    if "clinica de occidente" in key:
        return {
            "title": "Clinica de Occidente",
            "sections": [
                (
                    "Participaciones / integracion patrimonial (relevante)",
                    [
                        "Participacion accionaria en EPS: inversion del 0,25% en Coomeva EPS S.A. en Liquidacion, reportada a 31-dic-2023.",
                        "Subsidiaria 100%: Resonancia de Occidente S.A.S. (100% participacion), segun EEFF referidos.",
                    ],
                ),
                (
                    "Dotacion / tecnologia (marca/modelo)",
                    [
                        "Informe Anual 2022: Siemens Somaton Drive, Syngo VIA, Somaton Go SIM, Somaton GO UP y Arco en C Cios.",
                        "Informe Anual 2022: Canon Toshiba Ecografo Aplio.",
                    ],
                ),
                (
                    "Cadena de suministro / compras (solo decisional)",
                    [
                        "Modelo formal de gestion y calificacion de proveedores (criticidad + evaluacion por cumplimiento/comercial/calidad/post-contractual) reportado por la clinica.",
                    ],
                ),
                (
                    "EPS (riesgo / friccion cuantificable)",
                    [
                        "Cuentas por cobrar al 31-dic-2023 (miles COP): Nueva EPS $136.416.780 (~COP 136,4 mil millones).",
                        "Cuentas por cobrar al 31-dic-2023 (miles COP): Salud Total $29.953.140 (~COP 30,0 mil millones).",
                        "Cuentas por cobrar al 31-dic-2023 (miles COP): Coosalud $26.567.017 (~COP 26,6 mil millones).",
                        "Mencion de renovacion contractual en 1S-2023 con EPS Sanitas (nota referida).",
                    ],
                ),
            ],
            "particular_procedure_mix_2024": [
                {"Procedimiento": "Endoscopia", "Participacion": 19.0},
                {"Procedimiento": "Laboratorio", "Participacion": 15.0},
                {"Procedimiento": "Adscritos", "Participacion": 12.0},
                {"Procedimiento": "Consulta Externa", "Participacion": 12.0},
                {"Procedimiento": "Hospitalizacion", "Participacion": 11.0},
                {"Procedimiento": "Imagenes", "Participacion": 11.0},
                {"Procedimiento": "Cirugia", "Participacion": 7.0},
                {"Procedimiento": "Otros", "Participacion": 14.0},
            ],
            "particular_income_2024": 1372993378,
        }

    if "dime" in key and "neurocardiovascular" in key:
        return {
            "title": "DIME Clinica Neurocardiovascular S.A.",
            "sections": [
                (
                    "Dotacion / tecnologia (marca/modelo)",
                    [
                        "Angiografo biplano Philips Azurion 7 B20/15 (declaracion institucional referida).",
                        "Alianza para inversion tecnologica: Philips + Banco de Occidente (financiacion/condiciones de inversion a largo plazo, segun publicacion referida).",
                    ],
                ),
            ],
        }

    if "imbanaco" in key:
        return {
            "title": "Clinica Imbanaco S.A.S.",
            "sections": [
                (
                    "Control / estructura corporativa (impacto estrategico)",
                    [
                        "Adquisicion mayoritaria/control por Helios Healthcare Spain S.L., controlada por Else Kroner-Fresenius-Stiftung, desde 12-feb-2020 (segun lo citado).",
                    ],
                ),
                (
                    "Cadena de suministro (barrera competitiva)",
                    [
                        "En EEFF se describe uso de contratos de comodato con proveedores (equipos ligados a insumos/condiciones; vida util, sanciones, opcion de compra). Esto es clave por lock-in tecnologico y comercial.",
                    ],
                ),
                (
                    "EPS / acuerdos no estandar",
                    [
                        "Convenio PGP (pago prospectivo) con pago anticipado para gastroenterologia ambulatoria, evidenciado en EEFF 2019/2018 (y verificacion en red PBS segun el documento).",
                        "Enero 2025: referencia como nuevo prestador para ruta oncologica de Servicio Occidental de Salud (SOS).",
                    ],
                ),
            ],
        }

    if "valle del lili" in key or ("fundacion" in key and "lili" in key):
        return {
            "title": "Fundacion Valle del Lili",
            "sections": [
                (
                    "Dotacion / tecnologia (marca/modelo + capacidades)",
                    [
                        "Resonancia Siemens Magnetom Sola.",
                        "Radioterapia: TrueBeam, BRAVOS y tomografo Somaton (segun publicacion institucional referida).",
                        "Cardio intervencionista/estructural: oferta con TAVI/TAVR, MitraClip, y soporte de IVUS y FFR (OCT proximamente segun texto). MitraClip identificado como dispositivo de Abbott en referencia tecnica incluida.",
                    ],
                ),
                (
                    "Cadena de suministro (operativo decisional)",
                    [
                        "Portal de proveedores y proceso formal de relacionamiento/facturacion (onboarding administrativo).",
                    ],
                ),
            ],
        }

    return None


def render_services_matrix(selected_ips: str | None = None) -> None:
    section_header("Matriz de servicios por IPS", "Fuente: hoja Servicios")
    explain_box(
        "Como se calcula",
        [
            "Filas: IPS; columnas: servicios.",
            "Los dummies se transforman de 0/1 a NO/SI.",
            "Se colorea SI en verde y NO en gris para lectura rapida.",
        ],
    )

    if servicios_matrix_df.empty:
        st.warning("No se pudo leer la hoja Servicios.")
        st.caption(f"Detalle: {servicios_matrix_source}")
        return

    matrix = servicios_matrix_df.copy().dropna(axis=0, how="all").dropna(axis=1, how="all")
    cols = [str(c) for c in matrix.columns]
    ips_col = next(
        (
            c
            for c in cols
            if any(token in normalize_text(c) for token in ["ips", "prestador", "entidad", "nombre"])
        ),
        cols[0] if cols else None,
    )

    if ips_col is None or matrix.empty:
        st.info("La hoja Servicios no contiene estructura valida para matriz.")
        return

    def to_dummy(value: object) -> float | None:
        if pd.isna(value):
            return None
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        text = normalize_text(value)
        if text in {"si", "s", "x", "true", "verdadero", "1"}:
            return 1.0
        if text in {"no", "n", "false", "falso", "0", ""}:
            return 0.0
        num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(num):
            return None
        if num == 0:
            return 0.0
        if num == 1:
            return 1.0
        return None

    dummy_cols = []
    parsed_cols: dict[str, pd.Series] = {}
    for col in cols:
        if col == ips_col:
            continue
        parsed = matrix[col].map(to_dummy)
        non_null = parsed.dropna()
        if non_null.empty:
            continue
        if non_null.isin([0.0, 1.0]).all():
            dummy_cols.append(col)
            parsed_cols[col] = parsed

    if not dummy_cols:
        st.info("No se detectaron columnas dummy (0/1) en la hoja Servicios.")
        return

    view = pd.DataFrame()
    view["IPS"] = matrix[ips_col].astype(str).str.strip()
    view = view[view["IPS"].notna() & (view["IPS"] != "")]
    for col in dummy_cols:
        parsed = parsed_cols[col].reindex(view.index)
        view[col] = parsed.map(lambda x: "SI" if x == 1 else "NO")

    if selected_ips:
        view["__key"] = view["IPS"].map(norm_key)
        selected_key = norm_key(selected_ips)
        if (view["__key"] == selected_key).any():
            view = pd.concat(
                [
                    view[view["__key"] == selected_key],
                    view[view["__key"] != selected_key],
                ],
                ignore_index=True,
            )
        view = view.drop(columns=["__key"])

    def yes_no_style(val: object) -> str:
        if val == "SI":
            return "background-color: #d3f9d8; color: #1b4332; font-weight: 600; text-align: center;"
        if val == "NO":
            return "background-color: #e9ecef; color: #495057; font-weight: 600; text-align: center;"
        return ""

    styled = view.style.applymap(yes_no_style, subset=dummy_cols)
    st.dataframe(styled, width='stretch', hide_index=True)
    st.caption(f"Fuente: {servicios_matrix_source}")


def render_capacity_instalada() -> None:
    section_header("Capacidad instalada por IPS", "Fuente: hoja Capacidad")
    explain_box(
        "Como se calcula",
        [
            "Se usa la hoja Capacidad del Excel Cali ANALISIS.",
            "Se normaliza la primera columna como nombre de IPS.",
            "Se muestran columnas numericas y descriptivas con fila TOTAL para capacidades.",
        ],
    )

    if capacidad_df.empty:
        st.warning("No se pudo leer la hoja Capacidad.")
        st.caption(f"Detalle: {capacidad_source}")
        return

    work = capacidad_df.copy().dropna(axis=0, how="all").dropna(axis=1, how="all")
    if work.empty:
        st.warning("La hoja Capacidad no contiene datos utiles.")
        st.caption(f"Detalle: {capacidad_source}")
        return

    cols = [str(c).strip() for c in work.columns]
    ips_col = (
        find_col(cols, ["ips"])
        or find_col(cols, ["prestador"])
        or find_col(cols, ["entidad"])
        or find_col(cols, ["clinica"])
        or find_col(cols, ["fundacion"])
        or cols[0]
    )

    view = work.rename(columns={ips_col: "IPS"}).copy()
    view["IPS"] = view["IPS"].astype(str).str.strip()
    view = view[view["IPS"].notna() & (view["IPS"] != "")]

    ordered_cols = ["IPS"] + [c for c in view.columns if c != "IPS"]
    view = view[ordered_cols]

    numeric_cols: list[str] = []
    for col in view.columns:
        if col == "IPS":
            continue
        parsed = pd.to_numeric(view[col], errors="coerce")
        if parsed.notna().any():
            view[col] = parsed
            numeric_cols.append(col)

    total_row = {"IPS": "TOTAL"}
    for col in view.columns:
        if col == "IPS":
            continue
        if col in numeric_cols:
            total_row[col] = pd.to_numeric(view[col], errors="coerce").sum(min_count=1)
        else:
            total_row[col] = ""
    view = pd.concat([view, pd.DataFrame([total_row])], ignore_index=True)

    fmt = {
        col: (lambda v: "" if pd.isna(v) else f"{v:,.0f}")
        for col in numeric_cols
    }
    st.dataframe(view.style.format(fmt), width='stretch', hide_index=True)
    st.caption(f"Fuente: {capacidad_source}")

tab_analisis, tab_ips, tab_tarifas = st.tabs(
    ["Analisis", "Analisis IPS", "Tarifas IPS"]
)

with tab_analisis:
    section_header("Clasificación por ingresos", "IPS por ingresos totales (último año)")
    explain_box(
        "Como se calcula",
        [
            "Se usa el ingreso operativo (REV) del último año disponible.",
            "Se clasifican IPS en Alta/Media/Baja según terciles.",
            "El gráfico usa ingresos totales por IPS.",
        ],
    )
    last_year = max(int(y) for y in year_cols)
    rev_last = ing_oper_df[ing_oper_df["year"] == last_year][["entity", "REV"]].dropna()
    if rev_last.empty:
        st.warning("No hay datos de Total Ingreso Operativo para clasificar.")
    else:
        q33 = rev_last["REV"].quantile(0.33)
        q66 = rev_last["REV"].quantile(0.66)

        def segment(rev: float) -> str:
            if rev >= q66:
                return "Alta"
            if rev >= q33:
                return "Media"
            return "Baja"

        rev_last = rev_last.copy()
        rev_last["Segmento"] = rev_last["REV"].map(segment)
        rev_last["IPS"] = rev_last["entity"].map(lambda x: str(x).upper())
        rev_last = rev_last.sort_values("REV", ascending=False)

        fig = px.bar(
            rev_last,
            x="REV",
            y="IPS",
            color="Segmento",
            orientation="h",
            title="Total Ingreso Operativo por IPS (último año)",
            labels={"REV": "Total Ingreso Operativo", "IPS": "IPS"},
            color_discrete_map={"Alta": "#e8590c", "Media": "#1c7ed6", "Baja": "#0f6a62"},
        )
        fig.update_layout(title_x=0.5, title_xanchor="center")
        fig = style_chart(fig)
        chart_container(fig)

        divider()
        section_header("Evolución de ingresos", "Top 5 IPS por ingresos del último año")
        explain_box(
            "Como se calcula",
            [
                "Selecciona las 5 IPS con mayores ingresos del último año.",
                "Se grafica la evolución anual del ingreso operativo.",
            ],
        )
        top_ips = rev_last.head(5)["entity"].tolist()
        rev_line = ing_oper_df[ing_oper_df["entity"].isin(top_ips)][["entity", "year", "REV"]].dropna()
        rev_line["IPS"] = rev_line["entity"].map(lambda x: str(x).upper())
        fig = px.line(
            rev_line,
            x="year",
            y="REV",
            color="IPS",
            markers=True,
            title="Total Ingreso Operativo (Top 5 IPS)",
            labels={"year": "Año", "REV": "Total Ingreso Operativo"},
        )
        fig.update_layout(title_x=0.5, title_xanchor="center")
        fig = style_chart(fig)
        chart_container(fig)

        divider()
        section_header("Ingresos vs Utilidad Neta", "Todas las IPS (ultimo ano)")
        explain_box(
            "Como se calcula",
            [
                "Se cruza el ingreso operativo con la utilidad neta del último año.",
                "Permite comparar tamaño vs rentabilidad absoluta.",
            ],
        )
        net_last = blocks_df[blocks_df["year"] == last_year][["entity", "NET_INCOME"]]
        bar_df = rev_last[["entity", "REV"]].merge(net_last, on="entity", how="left")
        bar_df["IPS"] = bar_df["entity"].map(lambda x: str(x).upper())
        long_df = bar_df.melt(
            id_vars=["IPS"],
            value_vars=["REV", "NET_INCOME"],
            var_name="Metric",
            value_name="Value",
        )
        long_df["Metric"] = long_df["Metric"].map(
            {"REV": "Ingresos", "NET_INCOME": "Utilidad Neta"}
        )
        fig = px.bar(
            long_df,
            x="Value",
            y="IPS",
            color="Metric",
            barmode="group",
            orientation="h",
            title="Ingresos vs Utilidad Neta (ultimo ano)",
            labels={"Value": "COP", "IPS": "IPS"},
        )
        fig.update_layout(title_x=0.5, title_xanchor="center")
        fig = style_chart(fig)
        chart_container(fig)

        section_header("Clasificación IPS", "Segmento por ingresos")
        explain_box(
            "Como se calcula",
            [
                "Tabla con ingresos y utilidad neta del último año.",
                "Incluye fila TOTAL para agregar el mercado IPS.",
            ],
        )
        net_income_last = blocks_df[blocks_df["year"] == last_year][
            ["entity", "NET_INCOME"]
        ].rename(columns={"NET_INCOME": "Utilidad Neta"})
        class_table = rev_last[["entity", "IPS", "Segmento", "REV"]].merge(
            net_income_last, on="entity", how="left"
        )
        class_table = class_table.drop(columns=["entity"]).rename(
            columns={"REV": f"Total Ingreso Operativo {last_year}"}
        )
        class_table = append_total_row(
            class_table,
            "IPS",
            [f"Total Ingreso Operativo {last_year}", "Utilidad Neta"],
        )
        st.dataframe(
            class_table.style.format(
                {
                    f"Total Ingreso Operativo {last_year}": fmt_currency,
                    "Utilidad Neta": fmt_currency,
                }
            ),
            width='stretch',
        )

        divider()
        section_header("Score Financiero IPS", "Último año")
        explain_box(
            "Como se calcula",
            [
                "Score basado en subscores de liquidez, solvencia, rentabilidad y eficiencia.",
                "Se aplican pesos configurables en la barra lateral.",
                "Incluye fila PROMEDIO del mercado IPS.",
            ],
        )
        if used_default_weights:
            st.warning(
                "La suma de pesos en sidebar es 0%. Se aplicaron pesos predeterminados "
                "(Liquidez 30%, Endeudamiento 30%, Rentabilidad 20%, Eficiencia 20%)."
            )
        score_last = scored_df[scored_df["year"] == last_year][
            ["entity", "score_financiero"]
        ].dropna()
        if score_last.empty:
            st.info("No hay datos para el score financiero.")
        else:
            score_last = score_last.copy()
            score_last["IPS"] = score_last["entity"].map(lambda x: str(x).upper())
            score_last = score_last.sort_values("score_financiero", ascending=False)
            score_last = score_last[["entity", "IPS", "score_financiero"]].rename(
                columns={"score_financiero": "Score Financiero"}
            )
            avg_row = {
                "entity": "PROMEDIO",
                "IPS": "PROMEDIO",
                "Score Financiero": score_last["Score Financiero"].mean(skipna=True),
            }
            score_last = pd.concat([score_last, pd.DataFrame([avg_row])], ignore_index=True)
            st.dataframe(
                score_last.style.format({"Score Financiero": "{:.1f}"}),
                width='stretch',
            )

            divider()
            section_header("Evolución Score Financiero", "Top 5 IPS por score del último año")
            explain_box(
                "Como se calcula",
                [
                    "Top 5 IPS por score del último año.",
                    "Se grafica la evolución anual del Score Financiero.",
                ],
            )
            top_ips_score = score_last.head(5)["entity"].tolist()
            score_line = scored_df[scored_df["entity"].isin(top_ips_score)][
                ["entity", "year", "score_financiero"]
            ].dropna()
            score_line["IPS"] = score_line["entity"].map(lambda x: str(x).upper())
            fig = px.line(
                score_line,
                x="year",
                y="score_financiero",
                color="IPS",
                markers=True,
                title="Score Financiero (Top 5 IPS)",
                labels={"year": "Año", "score_financiero": "Score Financiero"},
            )
            fig.update_layout(title_x=0.5, title_xanchor="center")
            fig = style_chart(fig)
            chart_container(fig)

            divider()
            section_header("Score Financiero histórico", "IPS x Año")
            explain_box(
                "Como se calcula",
                [
                    "Matriz IPS x Año con el Score Financiero.",
                    "Incluye fila PROMEDIO por año.",
                ],
            )
            score_pivot = scored_df.pivot_table(
                index="entity", columns="year", values="score_financiero", aggfunc="mean"
            )
            score_pivot["IPS"] = score_pivot.index.map(lambda x: str(x).upper())
            score_pivot = score_pivot.reset_index(drop=True)
            score_pivot = score_pivot.set_index("IPS")
            if last_year in score_pivot.columns:
                score_pivot = score_pivot.sort_values(last_year, ascending=False)
            avg_row = score_pivot.mean(skipna=True).to_frame().T
            avg_row.index = ["PROMEDIO"]
            score_pivot = pd.concat([score_pivot, avg_row])
            st.dataframe(
                score_pivot.style.format({year: "{:.1f}" for year in score_pivot.columns}),
                width='stretch',
            )

    divider()
    render_capacity_instalada()

    divider()
    render_services_matrix()
with tab_ips:
    section_header("Analisis IPS", "Vista cuantitativa y cualitativa")
    explain_box(
        "Como se calcula",
        [
            "La vista Cuantitativo mantiene estados financieros, subscores y ratios.",
            "La vista Cualitativo consolida contexto narrativo por IPS.",
            "El contenido se organiza para lectura ejecutiva y trazabilidad.",
        ],
    )
    selected_ips = st.selectbox("IPS", ips_list, index=0, format_func=lambda x: str(x).upper())
    ips_view = subsection_selector(
        ["Cuantitativo", "Cualitativo"],
        key="competencia_ips_view",
        label="Vista IPS",
    )

    if ips_view == "Cuantitativo":
        section_header("KPI historicos clave", "Margen bruto, margen neto y tamaño de mercado")
        explain_box(
            "Como se calcula",
            [
                "Margen bruto y margen neto se toman de la serie historica de ratios.",
                "Tamaño de mercado = Ingreso IPS / Ingreso total IPS por año.",
                "Vista ejecutiva de tendencia anual, sin tabla de cumplimiento.",
            ],
        )
        ips_kpis = build_ips_historical_kpis(
            ratios_df=ratios_df,
            ing_oper_df=ing_oper_df,
            selected_ips=selected_ips,
            years_universe=[int(y) for y in year_cols],
        )
        if ips_kpis.empty:
            st.info("No hay datos suficientes para construir KPI historicos de la IPS seleccionada.")
        else:
            st.dataframe(
                ips_kpis.style.format(
                    {
                        "MargenBruto": "{:.2%}",
                        "MargenNeto": "{:.2%}",
                        "TamanioMercado": "{:.2%}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            plot_df = ips_kpis.rename(columns={"Año": "Ano"}).copy()
            plot_long = plot_df.melt(
                id_vars=["Ano"],
                value_vars=["MargenBruto", "MargenNeto", "TamanioMercado"],
                var_name="KPI",
                value_name="Valor",
            ).dropna(subset=["Valor"])
            if not plot_long.empty:
                fig = px.line(
                    plot_long,
                    x="Ano",
                    y="Valor",
                    color="KPI",
                    markers=True,
                    title="KPI historicos de la IPS seleccionada",
                    labels={"Ano": "Año", "Valor": "Valor"},
                )
                fig.update_layout(yaxis_tickformat=".0%")
                fig = style_chart(fig)
                chart_container(fig)

        divider()
        section_header("Bloques financieros", "Cuentas usadas y valores por ano")
        explain_box(
            "Como se calcula",
            [
                "Bloques construidos a partir de cuentas contables especificas.",
                "Valores anuales en millones de COP.",
            ],
        )
        st.caption(f"Fuente: {ips_source}")
        st.dataframe(blocks_table(selected_ips), width='stretch')

        divider()
        section_header("Estados financieros por cuenta", "IPS seleccionada")
        explain_box(
            "Como se calcula",
            [
                "Se respetan las cuentas y el orden del Excel.",
                "Se muestran valores anuales con formato moneda.",
            ],
        )
        df_ips = ips_df[ips_df["EPS_clean"] == selected_ips]
        if df_ips.empty:
            st.info("No hay datos para la IPS seleccionada.")
        else:
            estado_df, balance_df = split_financial_statements(df_ips)

            cols = ["CUENTA"] + year_cols if "CUENTA" in df_ips.columns else year_cols

            def render_financial_table(title: str, df_view: pd.DataFrame) -> None:
                st.subheader(title)
                if df_view.empty:
                    st.info("No hay datos para esta seccion.")
                    return
                view = df_view[cols].copy()
                for year in year_cols:
                    if year in view.columns:
                        view[year] = pd.to_numeric(view[year], errors="coerce")
                st.dataframe(
                    view.style.format({year: fmt_currency for year in year_cols}),
                    width='stretch',
                )

            render_financial_table("Estado de resultados", estado_df)
            render_financial_table("Balance general", balance_df)

        divider()
        section_header("Subscores y Score Financiero", "IPS seleccionada")
        explain_box(
            "Como se calcula",
            [
                "Subscores = promedio de ratios por categoria.",
                "Score Financiero = promedio ponderado con pesos del sidebar.",
                "Incluye columna Promedio por score.",
            ],
        )
        score_df = score_table(selected_ips)
        score_fmt = {str(year): "{:.1f}" for year in year_cols}
        score_fmt["Promedio"] = "{:.1f}"
        st.dataframe(score_df.style.format(score_fmt), width='stretch')

        divider()
        section_header("Indicadores de Liquidez")
        explain_box(
            "Como se calcula",
            [
                "Ratios de corto plazo (liquidez).",
                "Se agrega columna Promedio por indicador.",
            ],
        )
        st.dataframe(ratio_table(selected_ips, liquidity_ratios), width='stretch')

        divider()
        section_header("Indicadores de Endeudamiento / Solvencia")
        explain_box(
            "Como se calcula",
            [
                "Ratios de apalancamiento y solvencia.",
                "Se agrega columna Promedio por indicador.",
            ],
        )
        st.dataframe(ratio_table(selected_ips, solvency_ratios), width='stretch')

        divider()
        section_header("Indicadores de Rentabilidad")
        explain_box(
            "Como se calcula",
            [
                "Ratios de margen y retorno sobre activos.",
                "Se agrega columna Promedio por indicador.",
            ],
        )
        st.dataframe(ratio_table(selected_ips, profitability_ratios), width='stretch')

        divider()
        section_header("Indicadores de Eficiencia / Actividad")
        explain_box(
            "Como se calcula",
            [
                "Ratios de rotacion y eficiencia de costos.",
                "Se agrega columna Promedio por indicador.",
            ],
        )
        st.dataframe(ratio_table(selected_ips, efficiency_ratios), width='stretch')

    if ips_view == "Cualitativo":
        profile = qualitative_profile(selected_ips)
        if profile is None:
            section_header("Analisis cualitativo", "Sin ficha registrada")
            text_card(
                "Cobertura actual",
                "Aun no hay ficha cualitativa documentada para esta IPS. Agrega una ficha para habilitar la vista narrativa.",
            )
        else:
            section_header("Analisis cualitativo", profile["title"])
            explain_box(
                "Como se calcula",
                [
                    "Resumen narrativo por entidad basado en fuentes referidas del proyecto.",
                    "No reemplaza analisis legal ni due diligence documental.",
                    "Se estructura por vinculos, dotacion, cadena de suministro y eventos EPS.",
                ],
            )
            text_card("Entidad", profile["title"])
            for title, bullets in profile["sections"]:
                bullet_card(title, bullets)
            references = profile.get("references", [])
            if references:
                divider()
                section_header("Documentos de referencia", profile["title"])
                for ref in references:
                    label = str(ref.get("label", "Documento"))
                    url = str(ref.get("url", "")).strip()
                    if url:
                        st.markdown(f"- [{label}]({url})")
            income_mix_2024 = profile.get("income_mix_2024", [])
            if income_mix_2024:
                divider()
                section_header("Composicion de ingresos por cliente", f"{profile['title']} (2024)")
                pie_df = pd.DataFrame(income_mix_2024)
                fig = px.pie(
                    pie_df,
                    names="Cliente",
                    values="Participacion",
                    title="Composicion de ingresos por cliente (2024)",
                )
                fig.update_traces(
                    textposition="inside",
                    texttemplate="%{label}<br>%{percent}",
                    hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
                    sort=False,
                )
                fig.update_layout(title_x=0.5, title_xanchor="center")
                fig = style_chart(fig)
                chart_container(fig)

            particular_procedure_mix_2024 = profile.get("particular_procedure_mix_2024", [])
            if particular_procedure_mix_2024:
                divider()
                section_header(
                    "Pacientes particulares por procedimiento",
                    f"{profile['title']} (2024)",
                )
                procedure_df = pd.DataFrame(particular_procedure_mix_2024)
                fig = px.pie(
                    procedure_df,
                    names="Procedimiento",
                    values="Participacion",
                    title="Composicion de procedimientos en pacientes particulares (2024)",
                )
                fig.update_traces(
                    textposition="inside",
                    texttemplate="%{label}<br>%{percent}",
                    hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
                    sort=False,
                )
                fig.update_layout(title_x=0.5, title_xanchor="center")
                fig = style_chart(fig)
                chart_container(fig)

                particular_income_2024 = profile.get("particular_income_2024")
                if particular_income_2024 is not None:
                    text_card(
                        "Ingresos por Particulares (2024)",
                        f"COP {particular_income_2024:,.0f}".replace(",", "."),
                    )

        divider()
        render_services_matrix(selected_ips=selected_ips)
with tab_tarifas:
    section_header("Tarifas IPS", "Fuente: TarifasCompetencia")
    explain_box(
        "Como se calcula",
        [
            "Tabla informativa de tarifas por IPS.",
            "No se usa para calculos del modelo.",
        ],
    )
    section_header("Incremento Cali por servicio (referencia)", "Fuente: referencia definida por negocio")
    explain_box(
        "Como se calcula",
        [
            "Tabla fija de incrementos por servicio.",
            "No se usa para calculos del modelo.",
        ],
    )
    incremento_ref_df = pd.DataFrame(
        [
            ("MDNI", 0.70),
            ("CONSULTIA", 0.15),
            ("REPRO", 0.15),
            ("HEMO", 0.08),
            ("ELECTRO", 0.08),
            ("CIRUGIA", 0.08),
            ("TERAPIAS", 0.50),
            ("VASCULAR PERIF", 2.12),
            ("BANCO DE SANGRE", 2.00),
            ("LABORATORIO", 2.00),
            ("ENDOVASCULAR", 0.08),
            ("ESTANCIA", -0.30),
            ("NEURO", 0.08),
            ("RADIOLOGIA", 2.12),
        ],
        columns=["Servicio", "Incremento Cali"],
    )
    st.dataframe(
        incremento_ref_df.style.format(
            {"Incremento Cali": lambda v: "" if pd.isna(v) else f"{v:.0%}"}
        ),
        width='stretch',
    )

    divider()
    if tarifas_comp_df.empty:
        st.warning("No se pudo leer la hoja TarifasCompetencia.")
        st.caption(f"Detalle: {tarifas_comp_source}")
    else:
        view = tarifas_comp_df.copy()
        money_cols = [
            c
            for c in view.columns
            if any(
                normalize_text(token) in normalize_text(c)
                for token in ["tarifa", "precio", "valor", "venta", "costo"]
            )
        ]
        # Avoid Arrow type errors on mixed columns (e.g., numeric + 'No manejan').
        safe_view = view.copy()

        def safe_text(value: object) -> str:
            if pd.isna(value):
                return ""
            return str(value)

        def safe_money(value: object) -> str:
            if pd.isna(value):
                return ""
            num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.notna(num):
                return fmt_currency(float(num))
            return str(value)

        for col in safe_view.columns:
            if col in money_cols:
                safe_view[col] = safe_view[col].map(safe_money)
            else:
                safe_view[col] = safe_view[col].map(safe_text)

        st.dataframe(safe_view, width='stretch')

