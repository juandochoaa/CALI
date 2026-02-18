from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


import re
import unicodedata
from datetime import date, datetime
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboards.data_loader import (
    load_cifras_eps,
    load_cifras_eps_raw,
    load_eps_financials,
    load_financials,
)
from dashboards.ui import (
    apply_theme,
    chart_container,
    divider,
    explain_box,
    page_header,
    section_header,
    style_chart,
)
from src.models.target_population import compute_target_population

st.set_page_config(page_title="Proyecciones (EEFF)", layout="wide")
apply_theme()

page_header(
    "Proyecciones (EEFF)",
    "Ingresos, costos, EBITDA, prevalencia y comparacion.",
    "Financial outlook",
)

with st.sidebar:
    st.header("Parametros")
    st.caption("Archivo: data/raw/proyecciones.xlsx + Cali ANALISIS.xlsx")


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


INTERV_PCT_MAP = {
    "mdni": 0.4602,
    "consulta": 1.3599,
    "repro": 0.4607,
    "hemo": 0.0899,
    "electro": 0.0872,
    "cirugia": 0.0915,
}


def is_date_like(value: object) -> bool:
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return True
    try:
        parsed = pd.to_datetime(value, errors="coerce")
        return pd.notna(parsed)
    except Exception:
        return False


def normalize_account(text: str) -> str:
    return normalize_text(text).replace("%", "")


def filter_valid_accounts(df: pd.DataFrame, account_col: str) -> pd.DataFrame:
    work = df.copy()
    work[account_col] = work[account_col].astype(str).str.strip()
    norm = work[account_col].map(normalize_account)
    mask = (
        work[account_col].notna()
        & (work[account_col].astype(str).str.strip() != "")
        & ~norm.str.contains("margen")
        & ~norm.str.contains("%")
        & ~norm.str.contains("ratio")
    )
    return work[mask]


def find_col_any(columns: List[str], candidates: List[List[str]]) -> str | None:
    for includes in candidates:
        col = find_col(columns, includes)
        if col:
            return col
    return None


def pick_year_col(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        norm = normalize_text(col)
        if norm == "year" or "ano" in norm:
            return col
    return None


def map_interv_pct(service: str, interv_map: dict[str, float]) -> float:
    norm = normalize_text(service or "")
    for token, pct in interv_map.items():
        if token in norm:
            return pct
    return np.nan


def map_cap_service_to_tariff(service: str) -> str | None:
    norm = normalize_text(service or "")
    mapping = {
        "hemodinamia": "HEMO",
        "electrofisiologia": "ELECTRO",
        "cirugia": "CIRUGIA",
        "vascular": "CIRUGIA",
        "neurointervencionismo": "HEMO",
    }
    for key, val in mapping.items():
        if key in norm:
            return val
    return None


def parse_tarifas_servicios(
    tarifas_df: pd.DataFrame, ciudad: str, apply_increment: bool = False
) -> pd.DataFrame:
    cols = [str(c) for c in tarifas_df.columns]
    sede_col = find_col_any(cols, [["ciudad"], ["sede"], ["departamento"]])
    servicio_col = find_col_any(cols, [["servicio"]])
    tarifa_col = find_col_any(cols, [["tarifa"], ["precio"], ["valor"]])
    pacientes_col = find_col_any(cols, [["pacientes"], ["paciente"]])
    incremento_col = find_col_any(cols, [["incremento"], ["aumento"]])

    if not all([servicio_col, tarifa_col, pacientes_col]):
        return pd.DataFrame()

    work = tarifas_df.copy()
    work[servicio_col] = work[servicio_col].astype(str)
    if sede_col:
        work[sede_col] = work[sede_col].astype(str)
        work = work[
            work[sede_col].map(normalize_text).str.contains(normalize_text(ciudad), na=False)
        ]
    work[tarifa_col] = pd.to_numeric(work[tarifa_col], errors="coerce")
    work[pacientes_col] = pd.to_numeric(work[pacientes_col], errors="coerce")
    if incremento_col:
        work[incremento_col] = pd.to_numeric(work[incremento_col], errors="coerce")

    def weighted_avg(group: pd.DataFrame, col: str) -> float:
        pac = group[pacientes_col].sum()
        if pac and pac > 0:
            return (group[col] * group[pacientes_col]).sum() / pac
        return group[col].mean()

    grouped = (
        work.groupby(servicio_col, dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "TarifaPromedio": weighted_avg(g, tarifa_col),
                    "Pacientes": g[pacientes_col].sum(),
                    "Incremento": weighted_avg(g, incremento_col)
                    if incremento_col
                    else np.nan,
                }
            )
        )
        .reset_index()
        .rename(columns={servicio_col: "ServicioTarifa"})
    )
    grouped["ServicioTarifa"] = grouped["ServicioTarifa"].astype(str).str.upper().str.strip()
    total_pac = grouped["Pacientes"].sum()
    grouped["PctPacientes"] = grouped["Pacientes"] / total_pac if total_pac else np.nan
    grouped["PctIntervenciones"] = grouped["ServicioTarifa"].map(
        lambda x: map_interv_pct(x, INTERV_PCT_MAP)
    )

    if apply_increment:
        inc = grouped["Incremento"].fillna(0)
        inc = inc.apply(lambda v: v / 100 if pd.notna(v) and v > 1 else v)
        grouped["TarifaPromedio"] = grouped["TarifaPromedio"] * (1 + inc)
    return grouped


def parse_tarifas_procedimientos(
    tarifas_df: pd.DataFrame, ciudad: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [str(c) for c in tarifas_df.columns]
    sede_col = find_col_any(cols, [["ciudad"], ["sede"], ["departamento"]])
    servicio_col = find_col_any(cols, [["servicio"]])
    procedimiento_col = find_col_any(cols, [["procedimiento"]])
    tarifa_col = find_col_any(cols, [["tarifa"], ["precio"], ["valor"]])
    pacientes_col = find_col_any(cols, [["pacientes"], ["paciente"]])

    if not all([servicio_col, procedimiento_col, tarifa_col, pacientes_col]):
        return pd.DataFrame(), pd.DataFrame()

    work = tarifas_df.copy()
    work[servicio_col] = work[servicio_col].astype(str)
    work[procedimiento_col] = work[procedimiento_col].astype(str)
    if sede_col:
        work[sede_col] = work[sede_col].astype(str)
        work = work[
            work[sede_col].map(normalize_text).str.contains(normalize_text(ciudad), na=False)
        ]
    work[tarifa_col] = pd.to_numeric(work[tarifa_col], errors="coerce")
    work[pacientes_col] = pd.to_numeric(work[pacientes_col], errors="coerce")

    def weighted_avg(group: pd.DataFrame, col: str) -> float:
        pac = group[pacientes_col].sum()
        if pac and pac > 0:
            return (group[col] * group[pacientes_col]).sum() / pac
        return group[col].mean()

    proc = (
        work.groupby([servicio_col, procedimiento_col], dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "TarifaPromedio": weighted_avg(g, tarifa_col),
                    "Pacientes": g[pacientes_col].sum(),
                }
            )
        )
        .reset_index()
        .rename(columns={servicio_col: "ServicioTarifa", procedimiento_col: "Procedimiento"})
    )
    proc["ServicioTarifa"] = proc["ServicioTarifa"].astype(str).str.upper().str.strip()
    proc["Procedimiento"] = proc["Procedimiento"].astype(str).str.upper().str.strip()
    total_pac = proc["Pacientes"].sum()
    proc["PctPacientes"] = proc["Pacientes"] / total_pac if total_pac else np.nan

    service_totals = proc.groupby("ServicioTarifa")["Pacientes"].sum()
    proc["PctIntervenciones"] = proc.apply(
        lambda row: map_interv_pct(row["ServicioTarifa"], INTERV_PCT_MAP)
        * (row["Pacientes"] / service_totals.get(row["ServicioTarifa"], np.nan))
        if service_totals.get(row["ServicioTarifa"], 0) > 0
        else np.nan,
        axis=1,
    )

    serv = (
        proc.groupby("ServicioTarifa", dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "TarifaPromedio": (
                        (g["TarifaPromedio"] * g["Pacientes"]).sum() / g["Pacientes"].sum()
                        if g["Pacientes"].sum() > 0
                        else g["TarifaPromedio"].mean()
                    ),
                    "Pacientes": g["Pacientes"].sum(),
                }
            )
        )
        .reset_index()
    )
    serv["ServicioTarifa"] = serv["ServicioTarifa"].astype(str).str.upper().str.strip()
    total_pac_serv = serv["Pacientes"].sum()
    serv["PctPacientes"] = serv["Pacientes"] / total_pac_serv if total_pac_serv else np.nan
    serv["PctIntervenciones"] = serv["ServicioTarifa"].map(
        lambda x: map_interv_pct(x, INTERV_PCT_MAP)
    )
    return proc, serv


def scenario_id_from_label(label: str) -> int:
    norm = normalize_text(label)
    if "incremento" in norm:
        return 3
    if "bogota" in norm:
        return 2
    return 1


def _filter_tarifas_by_target(
    tarifas_df: pd.DataFrame,
    target: int,
    scenario_col: str | None,
    sede_col: str | None,
) -> pd.DataFrame:
    work = tarifas_df.copy()
    if scenario_col:
        work[scenario_col] = work[scenario_col].astype(str)

        def match_scenario(value: str) -> bool:
            norm = normalize_text(value)
            nums = re.findall(r"\d+", norm)
            if nums and int(nums[0]) == target:
                return True
            if target == 1:
                return "santander" in norm
            if target == 2:
                return ("bogota" in norm) and ("incremento" not in norm)
            return ("bogota" in norm and "incremento" in norm) or (
                "cali" in norm and "incremento" in norm
            )

        return work[work[scenario_col].map(match_scenario)].copy()

    if sede_col:
        work[sede_col] = work[sede_col].astype(str)
        if target == 1:
            return work[
                work[sede_col].map(normalize_text).str.contains("santander", na=False)
            ].copy()
        if target in (2, 3):
            return work[
                work[sede_col].map(normalize_text).str.contains("bogota", na=False)
            ].copy()
    return work


def _normalize_tarifa_weights(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "PacientesValle" not in out.columns:
        out["PacientesValle"] = np.nan
    if "RatioIntervenciones" not in out.columns:
        out["RatioIntervenciones"] = np.nan
    if "IntervencionesRaw" not in out.columns:
        out["IntervencionesRaw"] = np.nan

    out["Pacientes"] = pd.to_numeric(out["Pacientes"], errors="coerce")
    out["PacientesValle"] = pd.to_numeric(out["PacientesValle"], errors="coerce")

    total_pac_valle = out["PacientesValle"].sum(min_count=1)
    total_pac = out["Pacientes"].sum(min_count=1)

    if pd.notna(total_pac_valle) and total_pac_valle > 0:
        out["PctPacientes"] = out["PacientesValle"] / total_pac_valle
    elif pd.notna(total_pac) and total_pac > 0:
        out["PctPacientes"] = out["Pacientes"] / total_pac
    else:
        pct = pd.to_numeric(out.get("PctPacientes"), errors="coerce")
        pct_sum = pct.sum(min_count=1)
        if pd.notna(pct_sum) and pct_sum > 0:
            out["PctPacientes"] = pct / pct_sum
        elif len(out) > 0:
            out["PctPacientes"] = 1.0 / len(out)
        else:
            out["PctPacientes"] = np.nan
        out["Pacientes"] = out["PctPacientes"]

    out["RatioIntervenciones"] = pd.to_numeric(
        out["RatioIntervenciones"], errors="coerce"
    )
    inter_raw = pd.to_numeric(out["IntervencionesRaw"], errors="coerce")
    pac_valle = pd.to_numeric(out["PacientesValle"], errors="coerce")
    ratio_fallback = np.divide(
        inter_raw.to_numpy(dtype=float),
        pac_valle.to_numpy(dtype=float),
        out=np.full(len(out), np.nan, dtype=float),
        where=(pac_valle.to_numpy(dtype=float) > 0),
    )
    missing_ratio = out["RatioIntervenciones"].isna()
    if missing_ratio.any():
        out.loc[missing_ratio, "RatioIntervenciones"] = ratio_fallback[missing_ratio]
    return out


def _parse_percent_value(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    text = text.replace("%", "").replace(" ", "")
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(".", "").replace(",", ".")
    try:
        out = float(text)
    except Exception:
        return np.nan
    if out > 1:
        out = out / 100.0
    return float(out)


def _build_tarifa_scenario_group(
    work: pd.DataFrame,
    servicio_col: str,
    tarifa_col: str,
    pacientes_col: str | None,
    pct_pacientes_col: str | None,
    pacientes_valle_col: str | None,
    ratio_interv_col: str | None,
    intervenciones_col: str | None,
) -> pd.DataFrame:
    if work.empty:
        return pd.DataFrame()

    local = work.copy()
    local[servicio_col] = local[servicio_col].astype(str).str.upper().str.strip()
    local[tarifa_col] = pd.to_numeric(local[tarifa_col], errors="coerce")

    if pacientes_col:
        local[pacientes_col] = pd.to_numeric(local[pacientes_col], errors="coerce")
    else:
        local["__pacientes_fallback__"] = 1.0
        pacientes_col = "__pacientes_fallback__"
    if pct_pacientes_col:
        local[pct_pacientes_col] = local[pct_pacientes_col].map(_parse_percent_value)
    if pacientes_valle_col:
        local[pacientes_valle_col] = pd.to_numeric(
            local[pacientes_valle_col], errors="coerce"
        )
    if ratio_interv_col:
        local[ratio_interv_col] = local[ratio_interv_col].map(_parse_percent_value)
    if intervenciones_col:
        local[intervenciones_col] = pd.to_numeric(local[intervenciones_col], errors="coerce")

    def weighted_avg(group: pd.DataFrame, value_col: str | None) -> float:
        if not value_col:
            return np.nan
        pac = pd.to_numeric(group[pacientes_col], errors="coerce").fillna(0)
        val = pd.to_numeric(group[value_col], errors="coerce")
        total_pac = pac.sum()
        if total_pac > 0:
            return float((val * pac).sum(min_count=1) / total_pac)
        return float(val.mean())

    grouped = (
        local.groupby(servicio_col, dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "TarifaBase": weighted_avg(g, tarifa_col),
                    "Pacientes": pd.to_numeric(g[pacientes_col], errors="coerce").sum(
                        min_count=1
                    ),
                    "PctPacientesRaw": weighted_avg(g, pct_pacientes_col),
                    "PacientesValleRaw": pd.to_numeric(
                        g[pacientes_valle_col], errors="coerce"
                    ).sum(min_count=1)
                    if pacientes_valle_col
                    else np.nan,
                    "RatioIntervencionesRaw": weighted_avg(g, ratio_interv_col),
                    "IntervencionesRaw": pd.to_numeric(g[intervenciones_col], errors="coerce").sum(
                        min_count=1
                    )
                    if intervenciones_col
                    else np.nan,
                }
            )
        )
        .reset_index()
        .rename(columns={servicio_col: "ServicioTarifa"})
    )

    grouped["TarifaPromedio"] = grouped["TarifaBase"]
    grouped["PctPacientes"] = pd.to_numeric(grouped["PctPacientesRaw"], errors="coerce")
    grouped["PacientesValle"] = pd.to_numeric(grouped["PacientesValleRaw"], errors="coerce")
    grouped["RatioIntervenciones"] = pd.to_numeric(
        grouped["RatioIntervencionesRaw"], errors="coerce"
    )
    grouped["IntervencionesRaw"] = pd.to_numeric(grouped["IntervencionesRaw"], errors="coerce")

    ratio_fallback = np.divide(
        grouped["IntervencionesRaw"].to_numpy(dtype=float),
        grouped["PacientesValle"].to_numpy(dtype=float),
        out=np.full(len(grouped), np.nan, dtype=float),
        where=(grouped["PacientesValle"].to_numpy(dtype=float) > 0),
    )
    missing_ratio = grouped["RatioIntervenciones"].isna()
    if missing_ratio.any():
        grouped.loc[missing_ratio, "RatioIntervenciones"] = ratio_fallback[missing_ratio]

    grouped["OrigenServicio"] = "Original"
    grouped = _normalize_tarifa_weights(grouped)
    return grouped


def _complete_scenario_1_from_2(
    grouped_s1: pd.DataFrame, grouped_s2: pd.DataFrame
) -> pd.DataFrame:
    if grouped_s1.empty and grouped_s2.empty:
        return pd.DataFrame()
    if grouped_s1.empty:
        out = grouped_s2.copy()
        out["OrigenServicio"] = "S2_completado"
        return _normalize_tarifa_weights(out)

    s1 = grouped_s1.copy()
    s2 = grouped_s2.copy()
    s2_services = set(s2["ServicioTarifa"])
    s1 = s1[s1["ServicioTarifa"].isin(s2_services)].copy()
    s1["OrigenServicio"] = "S1_original"
    s2["OrigenServicio"] = "S2_completado"

    missing_services = sorted(set(s2["ServicioTarifa"]) - set(s1["ServicioTarifa"]))
    if not missing_services:
        return _normalize_tarifa_weights(s1)

    missing = s2[s2["ServicioTarifa"].isin(missing_services)].copy()
    existing = s1.copy()

    w_missing = pd.to_numeric(missing["PctPacientes"], errors="coerce").sum(min_count=1)
    w_missing = 0.0 if pd.isna(w_missing) else float(np.clip(w_missing, 0.0, 1.0))

    existing["Pacientes"] = pd.to_numeric(existing["Pacientes"], errors="coerce")
    missing["Pacientes"] = pd.to_numeric(missing["Pacientes"], errors="coerce")
    existing_pac = existing["Pacientes"].sum(min_count=1)
    missing_pac = missing["Pacientes"].sum(min_count=1)

    if (
        pd.notna(existing_pac)
        and existing_pac > 0
        and pd.notna(missing_pac)
        and missing_pac > 0
        and 0 < w_missing < 1
    ):
        target_existing_pac = missing_pac * (1.0 - w_missing) / w_missing
        scale_existing = target_existing_pac / existing_pac
        existing["Pacientes"] = existing["Pacientes"] * scale_existing
    elif w_missing >= 1:
        existing["Pacientes"] = 0.0
    else:
        target_existing_weight = max(0.0, 1.0 - w_missing)
        pct_existing = pd.to_numeric(existing["PctPacientes"], errors="coerce").fillna(0.0)
        pct_sum = pct_existing.sum()
        if pct_sum > 0:
            pct_existing = pct_existing / pct_sum * target_existing_weight
        elif len(existing) > 0:
            pct_existing = pd.Series(
                target_existing_weight / len(existing), index=existing.index, dtype=float
            )
        existing["PctPacientes"] = pct_existing
        denom = (
            float(missing_pac / w_missing)
            if w_missing > 0 and pd.notna(missing_pac) and missing_pac > 0
            else max(float(existing_pac) if pd.notna(existing_pac) else 0.0, 1.0)
        )
        existing["Pacientes"] = existing["PctPacientes"] * denom
        if (pd.isna(missing_pac) or missing_pac <= 0) and w_missing > 0:
            missing["Pacientes"] = pd.to_numeric(
                missing["PctPacientes"], errors="coerce"
            ).fillna(0.0) * denom

    combined = pd.concat([existing, missing], ignore_index=True)
    return _normalize_tarifa_weights(combined)


def parse_tarifas_escenarios(
    tarifas_df: pd.DataFrame, scenario_label: str
) -> pd.DataFrame:
    if tarifas_df.empty:
        return pd.DataFrame()

    cols = [str(c) for c in tarifas_df.columns]
    scenario_col = find_col_any(cols, [["escenario"], ["scenario"], ["tipo"]])
    sede_col = find_col_any(cols, [["ciudad"], ["sede"], ["departamento"]])
    servicio_col = find_col_any(cols, [["servicio"]])
    tarifa_col = next((c for c in cols if normalize_text(c) == "tarifas"), None)
    if not tarifa_col:
        tarifa_col = find_col_any(
            cols,
            [
                ["tarifas", "promedio", "icb"],
                ["tarifa", "promedio", "icb"],
                ["tarifas", "promedio"],
                ["tarifa", "promedio"],
                ["tarifas"],
                ["tarifa"],
                ["precio"],
                ["valor"],
            ],
        )
    pacientes_valle_col = find_col_any(
        cols,
        [
            ["pacientes", "valle"],
            ["pacientes", "valle", "cauca"],
        ],
    )
    pacientes_col = next((c for c in cols if normalize_text(c) == "pacientes"), None)
    if not pacientes_col:
        pacientes_col = find_col_any(cols, [["pacientes"], ["paciente"]])
    if pacientes_col and pacientes_valle_col and pacientes_col == pacientes_valle_col:
        pacientes_col = None
    pct_pacientes_col = find_col_any(
        cols,
        [
            ["%", "pacientes"],
            ["pct", "pacientes"],
            ["porcentaje", "pacientes"],
        ],
    )
    ratio_interv_col = find_col_any(
        cols,
        [
            ["ratio", "intervenciones"],
            ["ratio", "intervencion"],
        ],
    )
    intervenciones_col = find_col_any(cols, [["intervenciones"]])
    if ratio_interv_col and intervenciones_col and ratio_interv_col == intervenciones_col:
        intervenciones_col = None

    if not servicio_col or not tarifa_col:
        return pd.DataFrame()

    target = scenario_id_from_label(scenario_label)
    work_target = _filter_tarifas_by_target(tarifas_df, target, scenario_col, sede_col)
    if work_target.empty:
        return pd.DataFrame()

    grouped = _build_tarifa_scenario_group(
        work_target,
        servicio_col=servicio_col,
        tarifa_col=tarifa_col,
        pacientes_col=pacientes_col,
        pct_pacientes_col=pct_pacientes_col,
        pacientes_valle_col=pacientes_valle_col,
        ratio_interv_col=ratio_interv_col,
        intervenciones_col=intervenciones_col,
    )

    if target == 1:
        work_s2 = _filter_tarifas_by_target(tarifas_df, 2, scenario_col, sede_col)
        if not work_s2.empty:
            grouped_s2 = _build_tarifa_scenario_group(
                work_s2,
                servicio_col=servicio_col,
                tarifa_col=tarifa_col,
                pacientes_col=pacientes_col,
                pct_pacientes_col=pct_pacientes_col,
                pacientes_valle_col=pacientes_valle_col,
                ratio_interv_col=ratio_interv_col,
                intervenciones_col=intervenciones_col,
            )
            grouped = _complete_scenario_1_from_2(grouped, grouped_s2)
        else:
            grouped["OrigenServicio"] = "S1_original"
            grouped = _normalize_tarifa_weights(grouped)
    elif target == 2:
        grouped["OrigenServicio"] = "S2_original"
    else:
        grouped["OrigenServicio"] = "S3_original"

    return grouped[
        [
            "ServicioTarifa",
            "TarifaBase",
            "TarifaPromedio",
            "Pacientes",
            "PctPacientes",
            "PacientesValle",
            "RatioIntervenciones",
            "IntervencionesRaw",
            "OrigenServicio",
        ]
    ]


def selector_escenario_tarifas(
    scenario: str,
    tarifas_esc_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    return parse_tarifas_escenarios(tarifas_esc_df, scenario), None


def build_tarifa_table(
    df: pd.DataFrame,
    sede_col: str,
    target_sede: str,
    servicio_col: str,
    procedimiento_col: str | None,
    tarifa_col: str,
    pacientes_col: str,
    posibles_valle: float | None,
    inter_col: str | None = None,
    interv_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    work = df.copy()
    work[sede_col] = work[sede_col].astype(str)
    work["sede_norm"] = work[sede_col].map(normalize_text)
    work = work[work["sede_norm"].str.contains(normalize_text(target_sede), na=False)]
    if work.empty:
        return pd.DataFrame()

    work[pacientes_col] = pd.to_numeric(work[pacientes_col], errors="coerce")
    work[tarifa_col] = pd.to_numeric(work[tarifa_col], errors="coerce")
    if inter_col:
        work[inter_col] = pd.to_numeric(work[inter_col], errors="coerce")

    total_pacientes = work[pacientes_col].sum()

    if interv_map:
        work["interv_pct_map"] = work[servicio_col].map(lambda x: map_interv_pct(x, interv_map))
        service_totals = work.groupby(servicio_col)[pacientes_col].sum().rename("service_total_pac")
        work = work.merge(service_totals, left_on=servicio_col, right_index=True, how="left")
        work["pct_interv_row"] = work["interv_pct_map"] * (
            work[pacientes_col] / work["service_total_pac"]
        )
    elif inter_col:
        total_interv = work[inter_col].sum()
        work["pct_interv_row"] = (
            work[inter_col] / total_interv if total_interv and total_interv > 0 else np.nan
        )
    else:
        work["pct_interv_row"] = np.nan

    group_col = procedimiento_col or servicio_col

    def weighted_tarifa(group: pd.DataFrame) -> float:
        pac = group[pacientes_col].sum()
        if pac and pac > 0:
            return (group[tarifa_col] * group[pacientes_col]).sum() / pac
        return group[tarifa_col].mean()

    grouped = (
        work.groupby(group_col, dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "TarifaPromedio": weighted_tarifa(g),
                    "Pacientes": g[pacientes_col].sum(),
                    "PctIntervenciones": g["pct_interv_row"].sum(min_count=1),
                }
            )
        )
        .reset_index()
        .rename(columns={group_col: "Servicio" if procedimiento_col is None else "Procedimiento"})
    )

    grouped["PctPacientes"] = (
        grouped["Pacientes"] / total_pacientes if total_pacientes and total_pacientes > 0 else np.nan
    )
    grouped["PctIntervenciones"] = grouped["PctIntervenciones"].fillna(grouped["PctPacientes"])

    if posibles_valle is not None and pd.notna(posibles_valle):
        grouped["PacientesValle"] = grouped["PctPacientes"] * posibles_valle
        grouped["Intervenciones"] = grouped["PacientesValle"] * (1 + grouped["PctIntervenciones"])
        grouped["Ventas"] = grouped["Intervenciones"] * grouped["TarifaPromedio"]
    else:
        grouped["PacientesValle"] = np.nan
        grouped["Intervenciones"] = np.nan
        grouped["Ventas"] = np.nan

    total_row = {
        ("Servicio" if procedimiento_col is None else "Procedimiento"): "TOTAL",
        "TarifaPromedio": weighted_tarifa(work),
        "Pacientes": grouped["Pacientes"].sum(min_count=1),
        "PctPacientes": grouped["PctPacientes"].sum(min_count=1),
        "PctIntervenciones": grouped["PctIntervenciones"].sum(min_count=1),
        "PacientesValle": grouped["PacientesValle"].sum(min_count=1),
        "Intervenciones": grouped["Intervenciones"].sum(min_count=1),
        "Ventas": grouped["Ventas"].sum(min_count=1),
    }
    grouped = pd.concat([grouped, pd.DataFrame([total_row])], ignore_index=True)

    return grouped


def fmt_currency(val) -> str:
    if pd.isna(val):
        return ""
    return f"${val:,.0f}"


def fmt_percent(val, digits: int = 1) -> str:
    if pd.isna(val):
        return ""
    return f"{val:.{digits}%}"


def format_currency_df(df: pd.DataFrame, columns: list[str]) -> pd.io.formats.style.Styler:
    def fmt(val):
        return fmt_currency(val)

    return df.style.format({col: fmt for col in columns if col in df.columns})


def pick_ingresos_series(annual_pivot: pd.DataFrame) -> pd.Series | None:
    targets = [
        "ingresos netos por ventas",
        "total ingreso operativo",
        "ingresos",
    ]
    norm_index = annual_pivot.index.map(normalize_account)
    for target in targets:
        mask = norm_index == target
        if mask.any():
            return annual_pivot.loc[mask].iloc[0].sort_index()
    return None


def weighted_growth(yoy: pd.Series) -> float:
    valid = yoy.dropna()
    if valid.empty:
        return float("nan")
    values = valid.to_numpy()
    weights = np.arange(1, len(values) + 1, dtype=float)
    return float(np.average(values, weights=weights))


EEFF_OUTPUT_ORDER = [
    "INGRESOS",
    "Insumos",
    "NominaAsistencial",
    "HonorariosMedicos",
    "HonorariosdeProductividad",
    "CostosDirectos",
    "CostosIndirectosdeFabricacion",
    "Depreciaciones",
    "COSTOS",
    "UTILIDADBRUTA",
    "OTROSINGRESOS",
    "NominaAdministrativa",
    "HonorariosAdministrativos",
    "DeteriorodeCartera",
    "Depreciacionesyamortizaciones",
    "OtrosGastosdeadministracion",
    "GASTOSTOTALES",
    "EBITDA",
    "UTILIDADOPERACIONAL",
    "INGRESOSFINANCIEROS",
    "GASTOSFINANCIEROS",
    "UTILIDADANTESDEIMPUESTOS",
    "IMPUESTOS",
    "UTILIDADNETA",
]

EEFF_DERIVED_ACCOUNTS = {
    "COSTOS",
    "UTILIDADBRUTA",
    "GASTOSTOTALES",
    "EBITDA",
    "UTILIDADOPERACIONAL",
    "UTILIDADANTESDEIMPUESTOS",
    "UTILIDADNETA",
}

PRICE_SCENARIOS = [
    "Santander (1)",
    "Bogotá (2)",
    "Bogotá + Incremento Cali (3)",
]

EEFF_DRIVER_ACCOUNTS = [a for a in EEFF_OUTPUT_ORDER if a not in EEFF_DERIVED_ACCOUNTS]

EEFF_EDITABLE_FACTOR_ACCOUNTS = [
    "Insumos",
    "NominaAsistencial",
    "HonorariosMedicos",
]

EEFF_ACCOUNT_ALIASES = {
    "ingresos": "INGRESOS",
    "ingresosnetosporventas": "INGRESOS",
    "totalingresooperativo": "INGRESOS",
    "insumos": "Insumos",
    "nominaasistencial": "NominaAsistencial",
    "honorariosmedicos": "HonorariosMedicos",
    "honorariosdeproductividad": "HonorariosdeProductividad",
    "costosdirectos": "CostosDirectos",
    "costosindirectosdefabricacion": "CostosIndirectosdeFabricacion",
    "depreciaciones": "Depreciaciones",
    "costos": "COSTOS",
    "utilidadbruta": "UTILIDADBRUTA",
    "otrosingresos": "OTROSINGRESOS",
    "otrosingresosoperacionales": "OTROSINGRESOS",
    "nominaadministrativa": "NominaAdministrativa",
    "honorariosadministrativos": "HonorariosAdministrativos",
    "deteriorodecartera": "DeteriorodeCartera",
    "depreciacionesyamortizaciones": "Depreciacionesyamortizaciones",
    "otrosgastosdeadministracion": "OtrosGastosdeadministracion",
    "gastostotales": "GASTOSTOTALES",
    "ebitda": "EBITDA",
    "utilidadoperacional": "UTILIDADOPERACIONAL",
    "ingresosfinancieros": "INGRESOSFINANCIEROS",
    "gastosfinancieros": "GASTOSFINANCIEROS",
    "utilidadantesdeimpuestos": "UTILIDADANTESDEIMPUESTOS",
    "impuestos": "IMPUESTOS",
    "utilidadneta": "UTILIDADNETA",
}


def account_key(text: object) -> str:
    norm = normalize_text(text)
    return re.sub(r"[^a-z0-9]", "", norm)


def canonical_desglose_account(text: object) -> str | None:
    return EEFF_ACCOUNT_ALIASES.get(account_key(text))


def extract_year_from_label(label: object) -> int | None:
    if isinstance(label, (int, np.integer)):
        year = int(label)
        return year if 1900 <= year <= 2100 else None
    if isinstance(label, float) and float(label).is_integer():
        year = int(label)
        return year if 1900 <= year <= 2100 else None
    if isinstance(label, (pd.Timestamp, datetime, date)):
        return int(pd.to_datetime(label).year)
    txt = str(label)
    match = re.search(r"(20\d{2})", txt)
    if match:
        return int(match.group(1))
    return None


def build_base_desglose_base(
    desglose_df: pd.DataFrame,
    years: tuple[int, ...] = (2024, 2025),
    source_name: str = "BOGOTA_DESGLOSE",
) -> tuple[pd.Series, list[str], str]:
    if desglose_df.empty:
        return pd.Series(dtype=float), EEFF_DRIVER_ACCOUNTS, f"No se encontro la hoja {source_name}."

    cols = [str(c) for c in desglose_df.columns]
    account_col = find_col(cols, ["cuenta"]) or cols[0]

    year_cols = []
    year_values = []
    for col in desglose_df.columns:
        if str(col) == account_col:
            continue
        y = extract_year_from_label(col)
        if y in years:
            year_cols.append(col)
            year_values.append(y)

    if not year_cols:
        return (
            pd.Series(dtype=float),
            EEFF_DRIVER_ACCOUNTS,
            f"No se encontraron columnas de 2024/2025 en {source_name}.",
        )

    work = desglose_df[[account_col] + year_cols].copy()
    work[account_col] = work[account_col].astype(str).str.strip()
    work = work[work[account_col] != ""].copy()
    work["CuentaCanon"] = work[account_col].map(canonical_desglose_account)
    work = work[work["CuentaCanon"].notna()].copy()
    if work.empty:
        return (
            pd.Series(dtype=float),
            EEFF_DRIVER_ACCOUNTS,
            f"No se reconocieron cuentas contables en {source_name}.",
        )

    for col in year_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    grouped = work.groupby("CuentaCanon", as_index=False)[year_cols].sum(min_count=1)
    grouped["MontoSantBase"] = grouped[year_cols].mean(axis=1, skipna=True)
    base = grouped.set_index("CuentaCanon")["MontoSantBase"].reindex(EEFF_OUTPUT_ORDER)

    missing = [acc for acc in EEFF_DRIVER_ACCOUNTS if pd.isna(base.get(acc))]
    detected = sorted(set(year_values))
    msg = f"Base {source_name} calculada como promedio de anos {detected}."
    return base, missing, msg


def recompute_derived_accounts(values: pd.Series) -> pd.Series:
    stmt = pd.Series(0.0, index=EEFF_OUTPUT_ORDER, dtype=float)
    base_vals = pd.to_numeric(values.reindex(EEFF_OUTPUT_ORDER), errors="coerce")
    stmt = stmt.add(base_vals, fill_value=0.0).fillna(0.0)

    stmt["COSTOS"] = (
        stmt["Insumos"]
        + stmt["NominaAsistencial"]
        + stmt["HonorariosMedicos"]
        + stmt["HonorariosdeProductividad"]
        + stmt["CostosDirectos"]
        + stmt["CostosIndirectosdeFabricacion"]
        + stmt["Depreciaciones"]
    )
    stmt["UTILIDADBRUTA"] = stmt["INGRESOS"] - stmt["COSTOS"]
    stmt["GASTOSTOTALES"] = (
        stmt["NominaAdministrativa"]
        + stmt["HonorariosAdministrativos"]
        + stmt["DeteriorodeCartera"]
        + stmt["Depreciacionesyamortizaciones"]
        + stmt["OtrosGastosdeadministracion"]
    )
    stmt["EBITDA"] = (
        stmt["UTILIDADBRUTA"]
        + stmt["OTROSINGRESOS"]
        - (stmt["GASTOSTOTALES"] - stmt["Depreciacionesyamortizaciones"])
    )
    stmt["UTILIDADOPERACIONAL"] = stmt["EBITDA"] - stmt["Depreciacionesyamortizaciones"]
    stmt["UTILIDADANTESDEIMPUESTOS"] = (
        stmt["UTILIDADOPERACIONAL"] + stmt["INGRESOSFINANCIEROS"] - stmt["GASTOSFINANCIEROS"]
    )
    stmt["UTILIDADNETA"] = stmt["UTILIDADANTESDEIMPUESTOS"] - stmt["IMPUESTOS"]
    return stmt


def build_cali_base_statement(
    base_sant: pd.Series, factor_map: dict[str, float]
) -> tuple[pd.Series, pd.Series]:
    factor_series = pd.Series(1.0, index=EEFF_OUTPUT_ORDER, dtype=float)
    for account in EEFF_EDITABLE_FACTOR_ACCOUNTS:
        factor_series[account] = float(factor_map.get(account, 1.0))

    base_numeric = pd.to_numeric(base_sant.reindex(EEFF_OUTPUT_ORDER), errors="coerce").fillna(0.0)
    adjusted = base_numeric * factor_series
    statement = recompute_derived_accounts(adjusted)
    return statement, factor_series


def build_ratio_from_statement(statement: pd.Series) -> pd.Series:
    ingresos = float(statement.get("INGRESOS", np.nan))
    if not np.isfinite(ingresos) or ingresos == 0:
        return pd.Series(np.nan, index=statement.index, dtype=float)
    return statement / ingresos


def project_statement_from_ratios(
    ratio_base: pd.Series, revenue_by_year: pd.Series, years: list[int]
) -> pd.DataFrame:
    out = pd.DataFrame(index=EEFF_OUTPUT_ORDER)
    for year in years:
        ingresos_year = pd.to_numeric(revenue_by_year.get(year, np.nan), errors="coerce")
        if pd.isna(ingresos_year):
            out[year] = np.nan
            continue
        drivers = pd.Series(0.0, index=EEFF_OUTPUT_ORDER, dtype=float)
        drivers["INGRESOS"] = float(ingresos_year)
        for account in EEFF_DRIVER_ACCOUNTS:
            if account == "INGRESOS":
                continue
            ratio_val = pd.to_numeric(ratio_base.get(account, np.nan), errors="coerce")
            if pd.notna(ratio_val):
                drivers[account] = float(ratio_val) * float(ingresos_year)
        out[year] = recompute_derived_accounts(drivers)
    return out


def build_constructor_scenarios(
    proj_statement: pd.DataFrame,
    years: list[int],
    fixed_payment: float,
    pct_revenue: float,
    pct_utility: float,
) -> pd.DataFrame:
    ingresos = pd.to_numeric(proj_statement.loc["INGRESOS", years], errors="coerce")
    utilidad_base = pd.to_numeric(proj_statement.loc["UTILIDADNETA", years], errors="coerce")

    pago_fijo = pd.Series(float(fixed_payment), index=years, dtype=float)
    pago_pct_ingresos = ingresos * float(pct_revenue)
    # Si la utilidad base es negativa, se asume pago cero para este esquema.
    pago_pct_utilidad = utilidad_base.clip(lower=0) * float(pct_utility)

    out = pd.DataFrame(
        {
            "Ano": years,
            "Ingresos": ingresos.values,
            "UtilidadNeta_Base": utilidad_base.values,
            "Pago_Fijo": pago_fijo.values,
            "Pago_PctIngresos": pago_pct_ingresos.values,
            "Pago_PctUtilidad": pago_pct_utilidad.values,
        }
    )
    out["UtilidadPost_Fijo"] = out["UtilidadNeta_Base"] - out["Pago_Fijo"]
    out["UtilidadPost_PctIngresos"] = out["UtilidadNeta_Base"] - out["Pago_PctIngresos"]
    out["UtilidadPost_PctUtilidad"] = out["UtilidadNeta_Base"] - out["Pago_PctUtilidad"]
    return out


def perpetuity_pv(flow_by_year: pd.Series, discount_rate: float, growth_rate: float) -> tuple[float, float, float]:
    flow = pd.to_numeric(flow_by_year, errors="coerce").fillna(0.0)
    if flow.empty:
        return np.nan, np.nan, np.nan
    if discount_rate <= growth_rate:
        return np.nan, np.nan, np.nan

    years = sorted(flow.index.tolist())
    y0 = int(years[0])
    y_last = int(years[-1])

    pv_finite = 0.0
    for y in years:
        t = int(y) - y0 + 1
        pv_finite += float(flow.loc[y]) / ((1.0 + discount_rate) ** t)

    terminal_base = float(flow.loc[y_last])
    terminal_at_last = terminal_base * (1.0 + growth_rate) / (discount_rate - growth_rate)
    t_last = y_last - y0 + 1
    pv_terminal = terminal_at_last / ((1.0 + discount_rate) ** t_last)
    pv_total = pv_finite + pv_terminal
    return pv_finite, pv_terminal, pv_total


def _format_es_number(value: float, decimals: int = 1) -> str:
    text = f"{float(value):,.{decimals}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def _format_mm_es(value_cop: float) -> str:
    return f"{(float(value_cop) / 1_000_000):.1f}".replace(".", ",") + " MM"


def build_rent_reference_table(area_m2: float) -> pd.DataFrame:
    scenarios = [
        ("Minimo", 25_606.0),
        ("Promedio", 43_349.0),
        ("Maximo", 75_336.0),
    ]
    rows: list[dict[str, object]] = []
    for scenario, rate in scenarios:
        monthly = float(rate) * float(area_m2)
        rows.append(
            {
                "Escenario": scenario,
                "Formula": f"{_format_es_number(rate, 0)} x {_format_es_number(area_m2, 1)}",
                "Arriendo total mensual (COP)": monthly,
                "Arriendo total mensual (COP, redondeado)": _format_mm_es(monthly),
            }
        )
    return pd.DataFrame(rows)


def compute_fixed_rent_base(area_m2: float, rate_m2: float) -> tuple[float, float]:
    monthly = float(area_m2) * float(rate_m2)
    annual = monthly * 12.0
    return monthly, annual


def calibrate_constructor_percentages(
    constructor_df: pd.DataFrame,
    discount_rate: float,
    perpetuity_growth: float,
    fixed_annual_payment: float,
) -> dict:
    warnings: list[str] = []
    result: dict[str, object] = {
        "vp_target": np.nan,
        "vp_ingresos_base": np.nan,
        "vp_utilidad_pos_base": np.nan,
        "pct_revenue_auto": np.nan,
        "pct_utility_auto": np.nan,
        "pct_revenue_auto_pct": np.nan,
        "pct_utility_auto_pct": np.nan,
        "warnings": warnings,
    }

    if constructor_df.empty:
        warnings.append("No hay flujo base para calibrar porcentajes del constructor.")
        return result

    if discount_rate <= perpetuity_growth:
        warnings.append("La tasa de descuento debe ser mayor que el crecimiento perpetuo.")
        return result

    year_index = pd.to_numeric(constructor_df.get("Ano"), errors="coerce")
    ingresos = pd.to_numeric(constructor_df.get("Ingresos"), errors="coerce")
    utilidad_base = pd.to_numeric(constructor_df.get("UtilidadNeta_Base"), errors="coerce")
    valid = year_index.notna()
    if not valid.any():
        warnings.append("No se encontraron anos validos para calibrar VP.")
        return result

    year_index = year_index.loc[valid].astype(int)
    ingresos = ingresos.loc[valid]
    utilidad_pos = utilidad_base.loc[valid].clip(lower=0.0)

    fixed_series = pd.Series(float(fixed_annual_payment), index=year_index, dtype=float)
    _, _, vp_target = perpetuity_pv(fixed_series, discount_rate, perpetuity_growth)
    result["vp_target"] = vp_target

    if not np.isfinite(vp_target):
        warnings.append("No fue posible calcular el VP objetivo del pago fijo base.")
        return result

    ingresos_series = pd.Series(ingresos.to_numpy(dtype=float), index=year_index)
    utilidad_pos_series = pd.Series(utilidad_pos.to_numpy(dtype=float), index=year_index)

    _, _, vp_ingresos_base = perpetuity_pv(ingresos_series, discount_rate, perpetuity_growth)
    _, _, vp_utilidad_pos_base = perpetuity_pv(utilidad_pos_series, discount_rate, perpetuity_growth)
    result["vp_ingresos_base"] = vp_ingresos_base
    result["vp_utilidad_pos_base"] = vp_utilidad_pos_base

    if np.isfinite(vp_ingresos_base) and vp_ingresos_base > 0:
        pct_revenue_auto = float(vp_target / vp_ingresos_base)
        result["pct_revenue_auto"] = pct_revenue_auto
        result["pct_revenue_auto_pct"] = pct_revenue_auto * 100.0
    else:
        warnings.append("No se pudo calibrar % ingresos: VP de ingresos no valido o no positivo.")

    if np.isfinite(vp_utilidad_pos_base) and vp_utilidad_pos_base > 0:
        pct_utility_auto = float(vp_target / vp_utilidad_pos_base)
        result["pct_utility_auto"] = pct_utility_auto
        result["pct_utility_auto_pct"] = pct_utility_auto * 100.0
        if pct_utility_auto > 1.0:
            warnings.append(
                "El % utilidades calibrado supera 100% para igualar VP."
            )
    else:
        warnings.append("No se pudo calibrar % utilidades: VP de utilidad positiva no valido o no positivo.")

    return result


def _parse_monetary_value(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    text = re.sub(r"[^0-9,.\-]", "", text)
    if not text:
        return np.nan
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        tail = text.split(",")[-1]
        if len(tail) <= 2:
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    else:
        text = text.replace(",", "")
    try:
        return float(text)
    except Exception:
        return np.nan


def build_salary_reference_table() -> pd.DataFrame:
    rows = [
        ("Auxiliar de enfermeria", 2_091_628, 2_024_540),
        ("Psicologo", 4_680_000, 3_753_866),
        ("AUDITOR MEDICO", 6_500_000, 7_303_342),
        ("MEDICO HOSPITALARIO", 5_361_000, 5_313_344),
        ("Bacteriologo", 3_376_920, 3_208_579),
        ("Camillero", 1_757_457, 1_757_457),
        ("Coordinador asistencial", 8_500_000, 7_509_070),
        ("JEFE DE ENFERMERIA", 4_000_000, 3_419_816),
        ("ENFERMERO (A) AUDITOR", 4_164_848, 4_415_224),
        ("Fisioterapeuta", 2_500_000, 2_934_013),
        ("INSTRUMENTADORA QUIRURGICA", 3_909_399, 3_747_403),
        ("NUTRICIONISTA", 2_750_000, 2_892_208),
        ("REGENTE DE FARMACIA", 2_704_198, 2_404_182),
        ("TECNOLOGO EN RADIOLOGIA", 2_119_342, 1_750_905),
        ("Jefe de Cirugia", 3_692_000, 4_269_721),
    ]
    return pd.DataFrame(rows, columns=["Cargo", "Cali", "Nuestra"])


def _apply_salary_result(
    result: dict[str, object],
    raw_table: pd.DataFrame,
    role_col: str,
    salary_col_cali: str,
    salary_col_ours: str,
    source_label: str,
) -> bool:
    out = raw_table[[role_col, salary_col_cali, salary_col_ours]].copy()
    out["Salario_Cali"] = out[salary_col_cali].map(_parse_monetary_value)
    out["Salario_Nosotros"] = out[salary_col_ours].map(_parse_monetary_value)
    out = out[(out["Salario_Cali"] > 0) & (out["Salario_Nosotros"] > 0)].copy()
    if out.empty:
        return False

    out["Factor_Cali_vs_Nosotros"] = out["Salario_Cali"] / out["Salario_Nosotros"]
    out["Aumento_Cali_vs_Nosotros"] = out["Factor_Cali_vs_Nosotros"] - 1.0

    result["table"] = out.reset_index(drop=True)
    result["salary_col_cali"] = salary_col_cali
    result["salary_col_ours"] = salary_col_ours
    result["role_col"] = role_col
    result["mean_increase"] = float(out["Aumento_Cali_vs_Nosotros"].mean())
    result["mean_factor"] = float(out["Factor_Cali_vs_Nosotros"].mean())
    result["source_label"] = source_label
    return True


def parse_salary_comparison_table(th_raw_df: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {
        "table": pd.DataFrame(),
        "salary_col_cali": None,
        "salary_col_ours": None,
        "role_col": None,
        "mean_increase": np.nan,
        "mean_factor": np.nan,
        "source_label": "Hoja TH",
        "warnings": [],
    }
    warnings = result["warnings"]
    ref_table = build_salary_reference_table()

    if th_raw_df.empty:
        warnings.append("No se encontro informacion en la hoja TH para comparacion salarial.")
        _apply_salary_result(
            result=result,
            raw_table=ref_table,
            role_col="Cargo",
            salary_col_cali="Cali",
            salary_col_ours="Nuestra",
            source_label="Tabla salarial de referencia (Cali vs nuestra)",
        )
        return result

    work = th_raw_df.copy().dropna(axis=0, how="all").dropna(axis=1, how="all")
    if work.empty:
        warnings.append("La hoja TH no contiene filas utiles para comparacion salarial.")
        _apply_salary_result(
            result=result,
            raw_table=ref_table,
            role_col="Cargo",
            salary_col_cali="Cali",
            salary_col_ours="Nuestra",
            source_label="Tabla salarial de referencia (Cali vs nuestra)",
        )
        return result

    header_idx = None
    for idx, row in work.iterrows():
        cells = [normalize_text(v) for v in row.tolist() if pd.notna(v) and str(v).strip()]
        if len(cells) < 2:
            continue
        has_cali = any("cali" in c for c in cells)
        has_ours = any(
            any(token in c for token in ["nuestro", "nosotros", "icb", "foscal", "actual", "interno"])
            for c in cells
        )
        has_salary = any(any(token in c for token in ["salario", "sueldo", "nomina", "remuneracion"]) for c in cells)
        if has_cali and has_ours and (has_salary or len(cells) >= 3):
            header_idx = idx
            break

    if header_idx is None:
        warnings.append(
            "No se detecto una tabla de comparacion salarial Cali vs nosotros en TH."
        )
        _apply_salary_result(
            result=result,
            raw_table=ref_table,
            role_col="Cargo",
            salary_col_cali="Cali",
            salary_col_ours="Nuestra",
            source_label="Tabla salarial de referencia (Cali vs nuestra)",
        )
        return result

    header_raw = work.loc[header_idx].tolist()
    headers: list[str] = []
    used: dict[str, int] = {}
    for pos, value in enumerate(header_raw):
        label = str(value).strip() if pd.notna(value) and str(value).strip() else f"Col_{pos+1}"
        count = used.get(label, 0)
        headers.append(label if count == 0 else f"{label}_{count+1}")
        used[label] = count + 1

    table = work.loc[header_idx + 1 :].copy()
    table = table.iloc[:, : len(headers)]
    table.columns = headers
    table = table.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if table.empty:
        warnings.append("La tabla salarial en TH no contiene filas de datos.")
        _apply_salary_result(
            result=result,
            raw_table=ref_table,
            role_col="Cargo",
            salary_col_cali="Cali",
            salary_col_ours="Nuestra",
            source_label="Tabla salarial de referencia (Cali vs nuestra)",
        )
        return result

    cols = [str(c) for c in table.columns]
    salary_col_cali = find_col_any(cols, [["salario", "cali"], ["cali"], ["mercado", "cali"]])
    salary_col_ours = find_col_any(
        cols,
        [
            ["salario", "nuestro"],
            ["salario", "nosotros"],
            ["icb"],
            ["foscal"],
            ["actual"],
            ["interno"],
        ],
    )

    if salary_col_cali is None or salary_col_ours is None:
        numeric_candidates: list[str] = []
        for col in cols:
            parsed = table[col].map(_parse_monetary_value)
            if parsed.notna().sum() >= 1:
                numeric_candidates.append(col)
        if salary_col_cali is None and numeric_candidates:
            salary_col_cali = numeric_candidates[0]
        if salary_col_ours is None and len(numeric_candidates) >= 2:
            salary_col_ours = numeric_candidates[1]

    if salary_col_cali is None or salary_col_ours is None:
        warnings.append(
            "No se pudieron identificar columnas salariales de Cali y Nosotros en TH."
        )
        _apply_salary_result(
            result=result,
            raw_table=ref_table,
            role_col="Cargo",
            salary_col_cali="Cali",
            salary_col_ours="Nuestra",
            source_label="Tabla salarial de referencia (Cali vs nuestra)",
        )
        return result

    role_col = find_col_any(
        cols,
        [["cargo"], ["perfil"], ["rol"], ["especialidad"], ["servicio"], ["area"]],
    )
    if role_col is None:
        role_col = next((c for c in cols if c not in [salary_col_cali, salary_col_ours]), cols[0])

    ok = _apply_salary_result(
        result=result,
        raw_table=table,
        role_col=role_col,
        salary_col_cali=salary_col_cali,
        salary_col_ours=salary_col_ours,
        source_label="Hoja TH",
    )
    if not ok:
        warnings.append("No hay filas salariales validas para calcular aumento promedio.")
        _apply_salary_result(
            result=result,
            raw_table=ref_table,
            role_col="Cargo",
            salary_col_cali="Cali",
            salary_col_ours="Nuestra",
            source_label="Tabla salarial de referencia (Cali vs nuestra)",
        )
    return result


def _is_valid_target_population_snapshot(snapshot: object) -> bool:
    return isinstance(snapshot, dict) and isinstance(snapshot.get("summary_metrics"), dict)


def _resolve_target_population_snapshot(
    comp_df: pd.DataFrame,
    edad_df: pd.DataFrame,
    prev_raw: pd.DataFrame,
    comp_source: str,
    edad_source: str,
    prev_source: str,
) -> tuple[dict, float, bool]:
    snapshot = st.session_state.get("target_population_snapshot_v1")
    autogen = False
    if not _is_valid_target_population_snapshot(snapshot):
        snapshot = compute_target_population(
            comparacion_df=comp_df,
            eps_edad_df=edad_df,
            prevalencia_df_raw=prev_raw,
        )
        autogen = True

    metadata = snapshot.setdefault("metadata", {})
    metadata["sources"] = {
        "comparacion": comp_source,
        "eps_edad": edad_source,
        "prevalencia": prev_source,
    }

    objetivo = pd.to_numeric(
        snapshot.get("summary_metrics", {}).get("posibles_atendidos_valle"),
        errors="coerce",
    )

    st.session_state["target_population_snapshot_v1"] = snapshot
    st.session_state["objetivo_valle_pacientes"] = objetivo
    return snapshot, float(objetivo) if pd.notna(objetivo) else np.nan, autogen


proj, proj_source = load_financials()
prev_raw, prev_source = load_cifras_eps_raw("Prevalencia", header=None)
th_raw_df, th_source = load_cifras_eps_raw("TH", header=None)
comp_df, comp_source = load_cifras_eps("Comparacion")
edad_df, edad_source = load_cifras_eps("EPS_Edad")
tarifas_esc_df, tarifas_esc_source = load_cifras_eps("Tarifas_Escenarios")
if tarifas_esc_df.empty:
    alt_tarifas_esc_df, alt_tarifas_esc_source = load_cifras_eps("Tarifas Escenarios")
    if not alt_tarifas_esc_df.empty:
        tarifas_esc_df, tarifas_esc_source = alt_tarifas_esc_df, alt_tarifas_esc_source
sant_df, sant_source = load_cifras_eps("SANTANDER")
if sant_df.empty:
    alt_sant_df, alt_sant_source = load_cifras_eps("EEFF_Santander")
    if not alt_sant_df.empty:
        sant_df, sant_source = alt_sant_df, alt_sant_source
base_desglose_df, base_desglose_source = load_cifras_eps("BOGOTA_DESGLOSE")
salary_comp = parse_salary_comparison_table(th_raw_df)

target_snapshot, objetivo_valle, target_snapshot_autogen = _resolve_target_population_snapshot(
    comp_df=comp_df,
    edad_df=edad_df,
    prev_raw=prev_raw,
    comp_source=comp_source,
    edad_source=edad_source,
    prev_source=prev_source,
)
if target_snapshot_autogen:
    st.info(
        "Se autogenero el calculo de poblacion objetivo en esta pagina. "
        "Al abrir Contexto y Demanda se reutilizara el mismo snapshot."
    )

proj_totals = pd.DataFrame()
total_year1 = np.nan

tab_eeff, tab_prev, tab_comp, tab_tar, tab_sant, tab_share = st.tabs(
    ["EEFF", "Prevalencia", "Comparacion", "Tarifas", "Santander", "Market Share"]
)

with tab_eeff:
    st.selectbox(
        "Escenario de precios",
        PRICE_SCENARIOS,
        index=0,
        key="precio_scenario",
    )

    # Santander ratios + growth (para EEFF y proyecciones)
    ratio_df = None
    growth_avg = np.nan
    if not sant_df.empty:
        account_col = sant_df.columns[0]
        month_cols = [c for c in sant_df.columns[1:] if is_date_like(c)]
        if month_cols:
            sant_work = filter_valid_accounts(sant_df, account_col)
            long_df = sant_work[[account_col] + month_cols].melt(
                id_vars=account_col, var_name="Month", value_name="Valor"
            )
            long_df["Valor"] = pd.to_numeric(long_df["Valor"], errors="coerce")
            long_df["Year"] = pd.to_datetime(long_df["Month"], errors="coerce").dt.year
            long_df = long_df.dropna(subset=["Year"])

            annual_df = (
                long_df.groupby([account_col, "Year"], dropna=False)["Valor"]
                .sum(min_count=1)
                .reset_index()
                .rename(columns={account_col: "Cuenta"})
            )
            annual_pivot = annual_df.pivot_table(
                index="Cuenta", columns="Year", values="Valor", aggfunc="sum"
            ).sort_index()
            ingresos_series = pick_ingresos_series(annual_pivot)
            if ingresos_series is not None:
                yoy = ingresos_series / ingresos_series.shift(1) - 1
                growth_avg = weighted_growth(yoy)
                ratio_df = annual_pivot.div(ingresos_series, axis=1).replace(
                    [np.inf, -np.inf], np.nan
                )
                ratio_df["Promedio"] = ratio_df.mean(axis=1, skipna=True)

    has_proj_kpi = False
    if proj.empty:
        pass
    else:
        required_cols = {
            "year",
            "revenue_cop_bn",
            "costs_cop_bn",
            "ebitda_cop_bn",
            "cashflow_cop_bn",
            "capex_cop_bn",
        }
        if not required_cols.issubset(set(proj.columns)):
            st.warning("La hoja de proyecciones no tiene las columnas requeridas.")
        else:
            has_proj_kpi = True
            section_header("KPIs financieros")
            explain_box(
                "Como se calcula",
                [
                    "Fuente: proyecciones.xlsx (si está disponible).",
                    "KPIs principales: ingresos, costos, EBITDA y flujo de caja.",
                    "Se muestran métricas del año 2030 como referencia.",
                ],
            )

            col1, col2, col3 = st.columns(3)
            col1.metric("EBITDA 2030", f"{proj.loc[proj['year'] == 2030, 'ebitda_cop_bn'].iat[0]:.1f} bn")
            col2.metric("Margen EBITDA", f"{proj['ebitda_cop_bn'].iloc[-1] / proj['revenue_cop_bn'].iloc[-1]:.0%}")
            col3.metric("Capex total", f"{proj['capex_cop_bn'].sum():.1f} bn")

            left, right = st.columns([1.1, 0.9])

            with left:
                plot_df = proj.melt(
                    id_vars="year",
                    value_vars=["revenue_cop_bn", "costs_cop_bn", "ebitda_cop_bn"],
                    var_name="metric",
                    value_name="value",
                )
                fig = px.line(
                    plot_df,
                    x="year",
                    y="value",
                    color="metric",
                    markers=True,
                    title="Ingresos, costos y EBITDA (COP bn)",
                )
                fig = style_chart(fig)
                chart_container(fig)
                st.caption(f"Fuente: {proj_source}")

            with right:
                fig = px.bar(
                    proj,
                    x="year",
                    y="cashflow_cop_bn",
                    title="Flujo de caja (COP bn)",
                    text="cashflow_cop_bn",
                )
                fig = style_chart(fig)
                chart_container(fig)
                st.caption(f"Fuente: {proj_source}")

            section_header("Detalle anual")
            explain_box(
                "Como se calcula",
                [
                    "Detalle anual de ingresos, costos, EBITDA, flujo de caja y capex.",
                    "Cifras en COP billones según el archivo de proyecciones.",
                ],
            )
            money_cols = [
                "revenue_cop_bn",
                "costs_cop_bn",
                "ebitda_cop_bn",
                "cashflow_cop_bn",
                "capex_cop_bn",
            ]
            st.dataframe(
                proj.style.format({col: (lambda v: "" if pd.isna(v) else f"${v:,.1f}") for col in money_cols}),
                width='stretch',
            )

    divider()
    section_header("Ventas Año 1 por objetivo de pacientes", "Objetivo Valle + mix del escenario")
    explain_box(
        "Como se calcula",
        [
            "Año 1 parte del objetivo de pacientes del Valle.",
            "Ese total se distribuye por servicio según %Pacientes del escenario de precios.",
            "Tarifa operativa por servicio: TARIFAS (valor final del escenario).",
            "Intervenciones = RatioIntervenciones x PacientesValle.",
            "Desde Año 2 en adelante se usa el promedio histórico de crecimiento de ventas.",
        ],
    )
    if pd.isna(objetivo_valle):
        st.warning("No se pudo calcular el objetivo Valle (posibles atendidos).")
    else:
        scenario = st.session_state.get("precio_scenario", PRICE_SCENARIOS[0])
        scenario_id = scenario_id_from_label(scenario)

        tariffs_serv, _ = selector_escenario_tarifas(
            scenario, tarifas_esc_df
        )
        tarifas_s1, _ = selector_escenario_tarifas(PRICE_SCENARIOS[0], tarifas_esc_df)
        tarifas_s2, _ = selector_escenario_tarifas(PRICE_SCENARIOS[1], tarifas_esc_df)

        if tariffs_serv.empty:
            tariffs_serv = tarifas_s1 if not tarifas_s1.empty else tarifas_s2

        st.caption(f"Escenario activo: {scenario}.")
        col1, col2 = st.columns(2)
        col1.metric("Objetivo Valle (pacientes)", f"{objetivo_valle:,.0f}")
        col2.metric("Escenario de precios", scenario)

        st.caption(
            "Escenario 1 se completa con servicios del escenario 2 y se reescalan ponderaciones."
        )

        if tariffs_serv.empty:
            st.warning("No se encontraron tarifas para el escenario seleccionado.")
        else:
            tariffs = tariffs_serv.copy()
            tariffs["PctPacientes"] = pd.to_numeric(
                tariffs.get("PctPacientes"), errors="coerce"
            )
            if tariffs["PctPacientes"].isna().all():
                st.warning(
                    "No se pudo calcular el % de pacientes del escenario; se reparte en partes iguales."
                )
                tariffs["PctPacientes"] = 1 / len(tariffs) if len(tariffs) else np.nan
            else:
                tariffs["PctPacientes"] = tariffs["PctPacientes"].fillna(0.0)
                total_mix = tariffs["PctPacientes"].sum()
                if total_mix and total_mix > 0:
                    tariffs["PctPacientes"] = tariffs["PctPacientes"] / total_mix

            tariffs["Pacientes_Ano1"] = objetivo_valle * tariffs["PctPacientes"]
            tariffs["Intervenciones_Ano1"] = tariffs.apply(
                lambda row: row["Pacientes_Ano1"] * (1 + row["PctIntervenciones"])
                if pd.notna(row.get("PctIntervenciones"))
                else row["Pacientes_Ano1"],
                axis=1,
            )
            tariffs["Ventas_Ano1"] = tariffs["Intervenciones_Ano1"] * tariffs["TarifaPromedio"]

            show_cols = [
                "ServicioTarifa",
                "OrigenServicio",
                "PctPacientes",
                "Pacientes_Ano1",
                "TarifaPromedio",
                "PctIntervenciones",
                "Ventas_Ano1",
            ]
            if scenario_id == 3:
                show_cols = [
                    "ServicioTarifa",
                    "OrigenServicio",
                    "PctPacientes",
                    "Pacientes_Ano1",
                    "TarifaPromedio",
                    "INCREMENTO CALI",
                    "TARIFAS CON INCREMENTO",
                    "PctIntervenciones",
                    "Ventas_Ano1",
                ]
            show_df = tariffs[show_cols].rename(columns={"ServicioTarifa": "Servicio"})

            total_row = {
                "Servicio": "TOTAL",
                "OrigenServicio": "",
                "PctPacientes": show_df["PctPacientes"].sum(min_count=1),
                "Pacientes_Ano1": show_df["Pacientes_Ano1"].sum(min_count=1),
                "TarifaPromedio": np.nan,
                "PctIntervenciones": np.nan,
                "Ventas_Ano1": show_df["Ventas_Ano1"].sum(min_count=1),
            }
            show_df = pd.concat([show_df, pd.DataFrame([total_row])], ignore_index=True)

            styled = show_df.style.format(
                {
                    "PctPacientes": lambda v: "" if pd.isna(v) else f"{v:.1%}",
                    "Pacientes_Ano1": "{:,.1f}",
                    "TarifaPromedio": lambda v: "" if pd.isna(v) else f"${v:,.0f}",
                    "INCREMENTO CALI": lambda v: "" if pd.isna(v) else f"{v:.2%}",
                    "TARIFAS CON INCREMENTO": lambda v: "" if pd.isna(v) else f"${v:,.0f}",
                    "PctIntervenciones": lambda v: "" if pd.isna(v) else f"{v:.1%}",
                    "Ventas_Ano1": lambda v: "" if pd.isna(v) else f"${v:,.0f}",
                }
            )
            st.dataframe(styled, width='stretch')

            divider()
            section_header("Proyección anual por servicio", "Mix del escenario activo desde Año 2")
            explain_box(
                "Como se calcula",
                [
                    "Año 1 usa el objetivo de pacientes del Valle distribuido por mix de servicios.",
                    "Desde Año 2 se mantiene el mix (%Pacientes) del escenario activo.",
                    "Crecimiento anual fijo igual al promedio histórico de ventas.",
                ],
            )
            total_year1 = show_df.loc[show_df["Servicio"] != "TOTAL", "Pacientes_Ano1"].sum()

            c_obj1, c_obj2 = st.columns(2)
            c_obj1.metric("Objetivo Valle (pacientes)", f"{objetivo_valle:,.0f}")
            c_obj2.metric("Pacientes Año 1 proyectados", f"{total_year1:,.0f}")

            growth_default = growth_avg
            if pd.isna(growth_default):
                st.warning("No se pudo calcular crecimiento historico; se asume 0%.")
                growth_default = 0.0
            growth_rate = float(growth_default)
            st.caption(f"Tasa predeterminada aplicada: {growth_default:.2%}.")
            st.caption(
                f"Crecimiento histórico calculado: {growth_default:.2%}. "
                "Esta misma tasa se usa para proyectar desde Año 2."
            )

            proj_years = list(range(2026, 2031))
            service_rows = []
            for year in proj_years:
                if year == 2026:
                    patients_by_service = tariffs["Pacientes_Ano1"]
                else:
                    total_year = total_year1 * (1 + growth_rate) ** (year - 2026)
                    patients_by_service = total_year * tariffs["PctPacientes"]
                interventions = patients_by_service * (1 + tariffs["PctIntervenciones"].fillna(0))
                sales = interventions * tariffs["TarifaPromedio"]
                for idx, svc in enumerate(tariffs["ServicioTarifa"]):
                    service_rows.append(
                        {
                            "Servicio": svc,
                            "Ano": year,
                            "Pacientes": patients_by_service.iloc[idx],
                            "Intervenciones": interventions.iloc[idx],
                            "Ventas": sales.iloc[idx],
                        }
                    )
            service_proj = pd.DataFrame(service_rows)

            # tabla completa por servicio (wide)
            def pivot_metric(metric: str) -> pd.DataFrame:
                pivot = service_proj.pivot_table(
                    index="Servicio", columns="Ano", values=metric, aggfunc="sum"
                )
                pivot.columns = [f"{col} {metric}" for col in pivot.columns]
                return pivot

            full_table = pivot_metric("Pacientes").join(
                pivot_metric("Intervenciones")
            ).join(pivot_metric("Ventas"))
            full_table = full_table.reset_index()
            st.dataframe(
                full_table.style.format(
                    {
                        **{c: "{:,.0f}" for c in full_table.columns if "Pacientes" in c},
                        **{c: "{:,.0f}" for c in full_table.columns if "Intervenciones" in c},
                        **{
                            c: (lambda v: "" if pd.isna(v) else f"${v:,.0f}")
                            for c in full_table.columns
                            if "Ventas" in c
                        },
                    }
                ),
                width='stretch',
            )

            # totales por ano
            proj_totals = (
                service_proj.groupby("Ano")[["Pacientes", "Ventas"]]
                .sum()
                .reset_index()
            )
            st.session_state["proj_totals"] = proj_totals
            st.session_state["total_year1"] = total_year1
            st.session_state["proj_years"] = proj_years
            st.dataframe(
                proj_totals.style.format(
                    {
                        "Pacientes": "{:,.0f}",
                        "Ventas": lambda v: "" if pd.isna(v) else f"${v:,.0f}",
                    }
                ),
                width='stretch',
            )

            if not has_proj_kpi:
                divider()
                section_header("KPIs financieros (proxy tarifas)")
                explain_box(
                    "Como se calcula",
                    [
                        "Proxy construido con ventas por tarifas + ratios Santander.",
                        "Se estima EBITDA y margen a partir de proporciones históricas.",
                    ],
                )
                revenue_by_year = proj_totals.set_index("Ano")["Ventas"].sort_index()
                rev_2030 = revenue_by_year.get(2030, np.nan)
                ebitda_2030 = np.nan
                if ratio_df is not None:
                    ebitda_mask = ratio_df.index.map(
                        lambda x: normalize_account(x) == "ebitda"
                    )
                    if ebitda_mask.any():
                        ebitda_ratio = ratio_df.loc[ebitda_mask, "Promedio"].iloc[0]
                        ebitda_2030 = ebitda_ratio * rev_2030 if pd.notna(rev_2030) else np.nan
                margen_ebitda = (
                    ebitda_2030 / rev_2030 if pd.notna(ebitda_2030) and pd.notna(rev_2030) else np.nan
                )
                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Ingresos 2030",
                    f"${rev_2030:,.0f}" if pd.notna(rev_2030) else "NA",
                )
                col2.metric(
                    "EBITDA 2030",
                    f"${ebitda_2030:,.0f}" if pd.notna(ebitda_2030) else "NA",
                )
                col3.metric(
                    "Margen EBITDA 2030",
                    f"{margen_ebitda:.1%}" if pd.notna(margen_ebitda) else "NA",
                )

            divider()
            section_header("Estructura de proyeccion anual (2026-2030)")
            explain_box(
                "Como se calcula",
                [
                    "Base Bogotá: promedio monetario 2024-2025 desde BOGOTA_DESGLOSE.",
                    "Ajuste Cali: factor editable por cuenta aplicado sobre montos base.",
                    "Con el EEFF base Cali se obtienen proporciones y se aplican a ingresos proyectados.",
                    "Las filas de utilidades/totales siempre se recalculan por formula contable.",
                ],
            )
            if base_desglose_df.empty:
                st.warning("No se pudo leer BOGOTA_DESGLOSE para proyectar EEFF.")
                st.caption(f"Detalle: {base_desglose_source}")
            else:
                base_desglose, missing_accounts, base_msg = build_base_desglose_base(
                    base_desglose_df,
                    years=(2024, 2025),
                    source_name="BOGOTA_DESGLOSE",
                )
                st.caption(f"{base_msg} Fuente: {base_desglose_source}")
                if missing_accounts:
                    st.warning(
                        "Faltan cuentas en BOGOTA_DESGLOSE; se imputan en 0 para proyeccion: "
                        + ", ".join(missing_accounts)
                    )

                salary_source_label = str(salary_comp.get("source_label", th_source))
                section_header("Comparacion salarial Cali vs nosotros", f"Fuente: {salary_source_label}")
                salary_table = salary_comp.get("table", pd.DataFrame())
                salary_warnings = salary_comp.get("warnings", [])
                for warning_msg in salary_warnings:
                    st.warning(str(warning_msg))
                if salary_source_label != "Hoja TH":
                    st.info(
                        "Se esta usando la tabla salarial de referencia para calcular el aumento promedio "
                        "de Nomina Asistencial."
                    )

                avg_salary_increase = pd.to_numeric(
                    salary_comp.get("mean_increase"), errors="coerce"
                )
                nomina_factor_auto = (
                    1.0 + float(avg_salary_increase) if pd.notna(avg_salary_increase) else 1.0
                )

                if isinstance(salary_table, pd.DataFrame) and not salary_table.empty:
                    role_col = str(salary_comp.get("role_col", salary_table.columns[0]))
                    salary_cali_col = str(
                        salary_comp.get("salary_col_cali", salary_table.columns[1])
                    )
                    salary_ours_col = str(
                        salary_comp.get("salary_col_ours", salary_table.columns[2])
                    )
                    salary_view = salary_table[
                        [
                            role_col,
                            salary_cali_col,
                            salary_ours_col,
                            "Salario_Cali",
                            "Salario_Nosotros",
                            "Aumento_Cali_vs_Nosotros",
                            "Factor_Cali_vs_Nosotros",
                        ]
                    ].copy()
                    st.dataframe(
                        salary_view.style.format(
                            {
                                salary_cali_col: "{}",
                                salary_ours_col: "{}",
                                "Salario_Cali": lambda v: "" if pd.isna(v) else f"${v:,.0f}",
                                "Salario_Nosotros": lambda v: "" if pd.isna(v) else f"${v:,.0f}",
                                "Aumento_Cali_vs_Nosotros": lambda v: "" if pd.isna(v) else f"{v:.2%}",
                                "Factor_Cali_vs_Nosotros": lambda v: "" if pd.isna(v) else f"{v:.4f}",
                            }
                        ),
                        width="stretch",
                    )
                m1, m2 = st.columns(2)
                m1.metric(
                    "Promedio aumento Cali vs nosotros",
                    f"{float(avg_salary_increase):.2%}" if pd.notna(avg_salary_increase) else "NA",
                )
                m2.metric(
                    "Factor automatico Nomina Asistencial",
                    f"{nomina_factor_auto:.4f}",
                )
                st.caption(
                    "La tasa promedio de aumento salarial (Cali vs nosotros) se aplica como "
                    "factor de NominaAsistencial = 1 + tasa."
                )

                nomina_signature = (
                    round(float(avg_salary_increase), 8) if pd.notna(avg_salary_increase) else None,
                    int(len(salary_table)) if isinstance(salary_table, pd.DataFrame) else 0,
                )
                if st.session_state.get("eeff_nomina_factor_auto_signature_v1") != nomina_signature:
                    st.session_state["eeff_factor_nomina_asistencial"] = float(nomina_factor_auto)
                    st.session_state["eeff_nomina_factor_auto_signature_v1"] = nomina_signature

                if "eeff_factor_insumos" not in st.session_state:
                    st.session_state["eeff_factor_insumos"] = 1.0
                if "eeff_factor_honorarios_medicos" not in st.session_state:
                    st.session_state["eeff_factor_honorarios_medicos"] = 1.0
                if "eeff_factor_nomina_asistencial" not in st.session_state:
                    st.session_state["eeff_factor_nomina_asistencial"] = float(nomina_factor_auto)

                section_header("Factores Cali (aplicados a monto base)")
                c1, c2, c3 = st.columns(3)
                factor_map = {
                    "Insumos": c1.number_input(
                        "Factor Cali - Insumos",
                        min_value=0.0,
                        value=float(st.session_state["eeff_factor_insumos"]),
                        step=0.05,
                        key="eeff_factor_insumos",
                    ),
                    "NominaAsistencial": c2.number_input(
                        "Factor Cali - Nomina Asistencial",
                        min_value=0.0,
                        value=float(st.session_state["eeff_factor_nomina_asistencial"]),
                        step=0.05,
                        key="eeff_factor_nomina_asistencial",
                    ),
                    "HonorariosMedicos": c3.number_input(
                        "Factor Cali - Honorarios Medicos",
                        min_value=0.0,
                        value=float(st.session_state["eeff_factor_honorarios_medicos"]),
                        step=0.05,
                        key="eeff_factor_honorarios_medicos",
                    ),
                }

                base_cali_stmt, factor_series = build_cali_base_statement(
                    base_desglose, factor_map
                )
                ratio_cali_base = build_ratio_from_statement(base_cali_stmt)

                base_table = pd.DataFrame(
                    {
                        "Cuenta": EEFF_OUTPUT_ORDER,
                        "MontoBaseDesglose": [
                            base_desglose.get(acc, np.nan) for acc in EEFF_OUTPUT_ORDER
                        ],
                        "FactorCali": [factor_series.get(acc, 1.0) for acc in EEFF_OUTPUT_ORDER],
                        "MontoCaliBase": [base_cali_stmt.get(acc, np.nan) for acc in EEFF_OUTPUT_ORDER],
                        "RatioCaliBase": [ratio_cali_base.get(acc, np.nan) for acc in EEFF_OUTPUT_ORDER],
                        "FilaDerivada": [acc in EEFF_DERIVED_ACCOUNTS for acc in EEFF_OUTPUT_ORDER],
                    }
                )

                section_header("Base Bogotá -> Base Cali (monto)")
                st.dataframe(
                    base_table.style.format(
                        {
                            "MontoBaseDesglose": lambda v: "" if pd.isna(v) else f"${v:,.0f}",
                            "FactorCali": "{:,.2f}",
                            "MontoCaliBase": lambda v: "" if pd.isna(v) else f"${v:,.0f}",
                            "RatioCaliBase": lambda v: "" if pd.isna(v) else f"{v:.2%}",
                        }
                    ),
                    width='stretch',
                )

                section_header("Ratios base Cali (aplicados hacia 2026-2030)")
                ratio_view = base_table[["Cuenta", "RatioCaliBase"]].copy()
                st.dataframe(
                    ratio_view.style.format(
                        {"RatioCaliBase": lambda v: "" if pd.isna(v) else f"{v:.2%}"}
                    ),
                    width='stretch',
                )

                ingresos_base = float(base_cali_stmt.get("INGRESOS", np.nan))
                if not np.isfinite(ingresos_base) or ingresos_base <= 0:
                    st.warning(
                        "INGRESOS base Cali es invalido (<=0). No se puede construir la proyeccion EEFF."
                    )
                else:
                    st.caption(
                        "Proyeccion EEFF: primero se ajusta monto base por factor Cali, luego se "
                        "calculan proporciones y finalmente se aplican a ingresos proyectados por año."
                    )
                    year_col_proj = pick_year_col(proj_totals)
                    if year_col_proj is None:
                        st.warning(
                            "No se pudo detectar la columna de ano en proyecciones de ventas."
                        )
                        st.stop()
                    revenue_by_year = proj_totals.set_index(year_col_proj)["Ventas"].sort_index()
                    proj_statement = project_statement_from_ratios(
                        ratio_base=ratio_cali_base,
                        revenue_by_year=revenue_by_year,
                        years=proj_years,
                    )
                    proj_table = proj_statement.reset_index().rename(columns={"index": "Cuenta"})
                    year_cols = [c for c in proj_table.columns if isinstance(c, int)]
                    st.dataframe(
                        proj_table.style.format(
                            {col: (lambda v: "" if pd.isna(v) else f"${v:,.0f}") for col in year_cols}
                        ),
                        width='stretch',
                    )

                    divider()
                    section_header("Escenarios de pago al constructor")
                    explain_box(
                        "Como se calcula",
                        [
                            "Referencia de lote: minimo/promedio/maximo en COP por m2.",
                            "Caso base obligatorio: escenario Promedio (arriendo fijo mensual).",
                            "Se calibra automaticamente % ingresos y % utilidades para igualar VP del pago fijo base.",
                            "Si utilidad neta base es negativa, el pago por % utilidades se toma en 0.",
                        ],
                    )

                    area_m2 = 4_619.2
                    monthly_rent_base, fixed_payment = compute_fixed_rent_base(
                        area_m2=area_m2,
                        rate_m2=43_349.0,
                    )
                    rent_ref_df = build_rent_reference_table(area_m2=area_m2)

                    st.dataframe(
                        rent_ref_df.style.format(
                            {
                                "Arriendo total mensual (COP)": lambda v: f"${v:,.1f}",
                            }
                        ),
                        width="stretch",
                    )
                    c_base_1, c_base_2 = st.columns(2)
                    c_base_1.metric(
                        "Arriendo mensual base (Promedio)",
                        f"${monthly_rent_base:,.1f}",
                    )
                    c_base_2.metric(
                        "Arriendo anual base (modelo)",
                        f"${fixed_payment:,.1f}",
                    )

                    p1, p2 = st.columns(2)
                    discount_rate = p1.number_input(
                        "Tasa descuento anual (%)",
                        min_value=0.01,
                        max_value=100.0,
                        value=15.0,
                        step=0.5,
                        format="%.2f",
                        key="constructor_discount_rate",
                    ) / 100.0
                    perpetuity_growth = p2.number_input(
                        "Crecimiento perpetuo (%)",
                        min_value=-20.0,
                        max_value=50.0,
                        value=3.0,
                        step=0.5,
                        format="%.2f",
                        key="constructor_perpetuity_growth",
                    ) / 100.0

                    constructor_base_df = build_constructor_scenarios(
                        proj_statement=proj_statement,
                        years=proj_years,
                        fixed_payment=float(fixed_payment),
                        pct_revenue=0.0,
                        pct_utility=0.0,
                    )
                    calibration = calibrate_constructor_percentages(
                        constructor_df=constructor_base_df,
                        discount_rate=float(discount_rate),
                        perpetuity_growth=float(perpetuity_growth),
                        fixed_annual_payment=float(fixed_payment),
                    )

                    auto_pct_revenue = pd.to_numeric(
                        calibration.get("pct_revenue_auto_pct"), errors="coerce"
                    )
                    auto_pct_utility = pd.to_numeric(
                        calibration.get("pct_utility_auto_pct"), errors="coerce"
                    )
                    flow_signature = (
                        tuple(pd.to_numeric(constructor_base_df["Ano"], errors="coerce").fillna(-1).astype(int).tolist()),
                        round(float(discount_rate), 10),
                        round(float(perpetuity_growth), 10),
                        round(float(fixed_payment), 2),
                        tuple(
                            np.round(
                                pd.to_numeric(
                                    constructor_base_df["Ingresos"], errors="coerce"
                                ).fillna(0.0).to_numpy(dtype=float),
                                2,
                            ).tolist()
                        ),
                        tuple(
                            np.round(
                                pd.to_numeric(
                                    constructor_base_df["UtilidadNeta_Base"], errors="coerce"
                                ).clip(lower=0.0).fillna(0.0).to_numpy(dtype=float),
                                2,
                            ).tolist()
                        ),
                    )
                    prev_signature = st.session_state.get("constructor_calibration_signature_v1")
                    if prev_signature != flow_signature:
                        if pd.notna(auto_pct_revenue):
                            st.session_state["constructor_pct_revenue"] = float(auto_pct_revenue)
                        if pd.notna(auto_pct_utility):
                            st.session_state["constructor_pct_utility"] = float(auto_pct_utility)
                        st.session_state["constructor_calibration_signature_v1"] = flow_signature

                    if "constructor_pct_revenue" not in st.session_state:
                        st.session_state["constructor_pct_revenue"] = (
                            float(auto_pct_revenue) if pd.notna(auto_pct_revenue) else 0.0
                        )
                    if "constructor_pct_utility" not in st.session_state:
                        st.session_state["constructor_pct_utility"] = (
                            float(auto_pct_utility) if pd.notna(auto_pct_utility) else 0.0
                        )

                    s1, s2, s3 = st.columns(3)
                    s1.number_input(
                        "Monto fijo anual base (COP)",
                        min_value=0.0,
                        value=float(fixed_payment),
                        step=50_000_000.0,
                        format="%.1f",
                        key="constructor_fixed_payment_base",
                        disabled=True,
                    )
                    pct_revenue_input = s2.number_input(
                        "% sobre ingresos (editable)",
                        min_value=0.0,
                        value=float(st.session_state["constructor_pct_revenue"]),
                        step=0.1,
                        format="%.4f",
                        key="constructor_pct_revenue",
                    )
                    pct_utility_input = s3.number_input(
                        "% sobre utilidades (editable)",
                        min_value=0.0,
                        value=float(st.session_state["constructor_pct_utility"]),
                        step=0.1,
                        format="%.4f",
                        key="constructor_pct_utility",
                    )
                    pct_revenue = float(pct_revenue_input) / 100.0
                    pct_utility = float(pct_utility_input) / 100.0

                    if discount_rate <= perpetuity_growth:
                        st.warning(
                            "La tasa de descuento debe ser mayor que el crecimiento perpetuo para calcular VP."
                        )
                    for warning_msg in calibration.get("warnings", []):
                        st.warning(str(warning_msg))

                    constructor_df = build_constructor_scenarios(
                        proj_statement=proj_statement,
                        years=proj_years,
                        fixed_payment=float(fixed_payment),
                        pct_revenue=float(pct_revenue),
                        pct_utility=float(pct_utility),
                    )

                    section_header("Comparacion anual de pagos y utilidad post-pago")
                    yearly_view = constructor_df[
                        [
                            "Ano",
                            "Ingresos",
                            "UtilidadNeta_Base",
                            "Pago_Fijo",
                            "Pago_PctIngresos",
                            "Pago_PctUtilidad",
                            "UtilidadPost_Fijo",
                            "UtilidadPost_PctIngresos",
                            "UtilidadPost_PctUtilidad",
                        ]
                    ].copy()
                    st.dataframe(
                        yearly_view.style.format(
                            {
                                "Ingresos": lambda v: "" if pd.isna(v) else f"${v:,.0f}",
                                "UtilidadNeta_Base": lambda v: "" if pd.isna(v) else f"${v:,.0f}",
                                "Pago_Fijo": lambda v: "" if pd.isna(v) else f"${v:,.0f}",
                                "Pago_PctIngresos": lambda v: "" if pd.isna(v) else f"${v:,.0f}",
                                "Pago_PctUtilidad": lambda v: "" if pd.isna(v) else f"${v:,.0f}",
                                "UtilidadPost_Fijo": lambda v: "" if pd.isna(v) else f"${v:,.0f}",
                                "UtilidadPost_PctIngresos": lambda v: "" if pd.isna(v) else f"${v:,.0f}",
                                "UtilidadPost_PctUtilidad": lambda v: "" if pd.isna(v) else f"${v:,.0f}",
                            }
                        ),
                        width="stretch",
                    )

                    util_plot = constructor_df[
                        ["Ano", "UtilidadPost_Fijo", "UtilidadPost_PctIngresos", "UtilidadPost_PctUtilidad"]
                    ].melt(
                        id_vars="Ano",
                        var_name="Escenario",
                        value_name="UtilidadPost",
                    )
                    util_plot["Escenario"] = util_plot["Escenario"].replace(
                        {
                            "UtilidadPost_Fijo": "Monto fijo",
                            "UtilidadPost_PctIngresos": "% ingresos",
                            "UtilidadPost_PctUtilidad": "% utilidades",
                        }
                    )
                    fig_util = px.line(
                        util_plot,
                        x="Ano",
                        y="UtilidadPost",
                        color="Escenario",
                        markers=True,
                        title="Utilidad neta post-pago por escenario",
                    )
                    fig_util = style_chart(fig_util)
                    chart_container(fig_util)

                    scenario_defs = [
                        ("Monto fijo", "Pago_Fijo", "UtilidadPost_Fijo"),
                        ("% ingresos", "Pago_PctIngresos", "UtilidadPost_PctIngresos"),
                        ("% utilidades", "Pago_PctUtilidad", "UtilidadPost_PctUtilidad"),
                    ]

                    year_index = constructor_df["Ano"].astype(int)
                    ingresos_series = pd.Series(constructor_df["Ingresos"].to_numpy(), index=year_index)
                    year_last = int(year_index.max())
                    pay_last_col = f"Pago_{year_last}"
                    util_last_col = f"UtilidadPost_{year_last}"
                    margin_last_col = f"MargenNetoPost_{year_last}"
                    summary_rows = []
                    for scenario_name, pay_col, util_col in scenario_defs:
                        payment_series = pd.Series(constructor_df[pay_col].to_numpy(), index=year_index)
                        util_series = pd.Series(constructor_df[util_col].to_numpy(), index=year_index)

                        pv_pay_fin, pv_pay_term, pv_pay_total = perpetuity_pv(
                            payment_series, discount_rate, perpetuity_growth
                        )
                        pv_util_fin, pv_util_term, pv_util_total = perpetuity_pv(
                            util_series, discount_rate, perpetuity_growth
                        )
                        ingreso_last = float(ingresos_series.get(year_last, np.nan))
                        util_last = float(util_series.get(year_last, np.nan))
                        margin_last = (
                            util_last / ingreso_last
                            if np.isfinite(ingreso_last) and ingreso_last != 0
                            else np.nan
                        )

                        summary_rows.append(
                            {
                                "Escenario": scenario_name,
                                pay_last_col: float(payment_series.get(year_last, np.nan)),
                                util_last_col: util_last,
                                margin_last_col: margin_last,
                                "VP_Pagos_Finito": pv_pay_fin,
                                "VP_Pagos_Terminal": pv_pay_term,
                                "VP_Pagos_Total": pv_pay_total,
                                "VP_Utilidad_Finita": pv_util_fin,
                                "VP_Utilidad_Terminal": pv_util_term,
                                "VP_Utilidad_Total": pv_util_total,
                            }
                        )

                    summary_df = pd.DataFrame(summary_rows)
                    summary_df["Ranking_Mas_Utilidad"] = summary_df["VP_Utilidad_Total"].rank(
                        ascending=False, method="dense"
                    )
                    summary_df["Ranking_Menor_Pago"] = summary_df["VP_Pagos_Total"].rank(
                        ascending=True, method="dense"
                    )
                    summary_df = summary_df.sort_values(
                        ["Ranking_Mas_Utilidad", "Ranking_Menor_Pago", "Escenario"]
                    ).reset_index(drop=True)

                    vp_target = pd.to_numeric(calibration.get("vp_target"), errors="coerce")
                    vp_pct_ingresos = pd.to_numeric(
                        summary_df.loc[summary_df["Escenario"] == "% ingresos", "VP_Pagos_Total"],
                        errors="coerce",
                    )
                    vp_pct_utilidades = pd.to_numeric(
                        summary_df.loc[summary_df["Escenario"] == "% utilidades", "VP_Pagos_Total"],
                        errors="coerce",
                    )
                    vp_pct_ingresos = (
                        float(vp_pct_ingresos.iloc[0]) if len(vp_pct_ingresos) else np.nan
                    )
                    vp_pct_utilidades = (
                        float(vp_pct_utilidades.iloc[0]) if len(vp_pct_utilidades) else np.nan
                    )
                    gap_vp_ingresos = (
                        vp_pct_ingresos - float(vp_target)
                        if pd.notna(vp_target) and np.isfinite(vp_pct_ingresos)
                        else np.nan
                    )
                    gap_vp_utilidades = (
                        vp_pct_utilidades - float(vp_target)
                        if pd.notna(vp_target) and np.isfinite(vp_pct_utilidades)
                        else np.nan
                    )

                    d1, d2, d3, d4, d5 = st.columns(5)
                    d1.metric(
                        "VP objetivo (Pago fijo base)",
                        f"${float(vp_target):,.0f}" if pd.notna(vp_target) else "NA",
                    )
                    d2.metric(
                        "% ingresos calibrado",
                        f"{float(auto_pct_revenue):.4f}%" if pd.notna(auto_pct_revenue) else "NA",
                    )
                    d3.metric(
                        "% utilidades calibrado",
                        f"{float(auto_pct_utility):.4f}%" if pd.notna(auto_pct_utility) else "NA",
                    )
                    d4.metric(
                        "Brecha VP % ingresos vs objetivo",
                        f"${gap_vp_ingresos:,.0f}" if np.isfinite(gap_vp_ingresos) else "NA",
                    )
                    d5.metric(
                        "Brecha VP % utilidades vs objetivo",
                        f"${gap_vp_utilidades:,.0f}" if np.isfinite(gap_vp_utilidades) else "NA",
                    )

                    if pd.notna(auto_pct_revenue) and not np.isclose(
                        float(pct_revenue_input),
                        float(auto_pct_revenue),
                        rtol=0.0,
                        atol=1e-9,
                    ):
                        st.info(
                            "El % ingresos aplicado difiere del calibrado automatico. "
                            "Revisa la brecha de VP mostrada arriba."
                        )
                    if pd.notna(auto_pct_utility) and not np.isclose(
                        float(pct_utility_input),
                        float(auto_pct_utility),
                        rtol=0.0,
                        atol=1e-9,
                    ):
                        st.info(
                            "El % utilidades aplicado difiere del calibrado automatico. "
                            "Revisa la brecha de VP mostrada arriba."
                        )

                    section_header("Resumen economico por escenario")
                    st.dataframe(
                        summary_df.style.format(
                            {
                                pay_last_col: lambda v: "" if pd.isna(v) else f"${v:,.0f}",
                                util_last_col: lambda v: "" if pd.isna(v) else f"${v:,.0f}",
                                margin_last_col: lambda v: "" if pd.isna(v) else f"{v:.2%}",
                                "VP_Pagos_Finito": lambda v: "" if pd.isna(v) else f"${v:,.0f}",
                                "VP_Pagos_Terminal": lambda v: "" if pd.isna(v) else f"${v:,.0f}",
                                "VP_Pagos_Total": lambda v: "" if pd.isna(v) else f"${v:,.0f}",
                                "VP_Utilidad_Finita": lambda v: "" if pd.isna(v) else f"${v:,.0f}",
                                "VP_Utilidad_Terminal": lambda v: "" if pd.isna(v) else f"${v:,.0f}",
                                "VP_Utilidad_Total": lambda v: "" if pd.isna(v) else f"${v:,.0f}",
                                "Ranking_Mas_Utilidad": "{:.0f}",
                                "Ranking_Menor_Pago": "{:.0f}",
                            }
                        ),
                        width='stretch',
                    )
with tab_prev:
    section_header("Prevalencia por rango de edad", "Fuente: Prevalencia")
    explain_box(
        "Como se calcula",
        [
            "Se usa la misma tabla de prevalencia del calculo central de poblacion objetivo.",
            "Los porcentajes se convierten a proporcion (0-1) si vienen en %.",
            "Se grafica prevalencia por grupo etario.",
        ],
    )
    prev_df = target_snapshot.get("metadata", {}).get("prevalencia_df", pd.DataFrame())
    if not isinstance(prev_df, pd.DataFrame) or prev_df.empty:
        st.warning("No se pudo leer la tabla de prevalencia.")
        st.caption(f"Detalle: {prev_source}")
    else:
        fig = px.bar(
            prev_df,
            x="GrupoEdad",
            y="Prevalencia",
            text="Prevalencia",
            title="Prevalencia de enfermedades cardiovasculares por edad",
        )
        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        fig.update_layout(title_x=0.5, title_xanchor="center")
        fig = style_chart(fig)
        chart_container(fig)
        st.dataframe(prev_df, width='stretch')

with tab_comp:
    section_header("Comparacion Santander vs Valle", "Fuente centralizada")
    explain_box(
        "Como se calcula",
        [
            "Se reutiliza el calculo central de Contexto y Demanda.",
            "Fallback tecnico: si no existe snapshot en sesion, se autogenera aqui.",
            "Objetivo Valle y tablas de edad se alimentan de la misma fuente unica.",
        ],
    )

    summary = target_snapshot.get("summary_metrics", {})
    eps_view = target_snapshot.get("eps_view", pd.DataFrame())
    edad_view = target_snapshot.get("edad_view", pd.DataFrame())
    formula_df = target_snapshot.get("formula_view", pd.DataFrame())
    edad_chart_df = target_snapshot.get("edad_chart_df", pd.DataFrame())
    warnings = target_snapshot.get("warnings", [])
    target_sources = target_snapshot.get("metadata", {}).get("sources", {})

    for msg in warnings:
        st.warning(msg)

    pct_atendido = pd.to_numeric(summary.get("pct_atendido_santander"), errors="coerce")
    atendidos_total = pd.to_numeric(summary.get("atendidos_santander"), errors="coerce")
    posibles_valle = pd.to_numeric(summary.get("posibles_atendidos_valle"), errors="coerce")
    valle_total = pd.to_numeric(summary.get("afiliados_valle_total"), errors="coerce")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("% atendido Santander", f"{pct_atendido:.2%}" if pd.notna(pct_atendido) else "NA")
    col2.metric("Atendidos Santander", f"{atendidos_total:,.0f}" if pd.notna(atendidos_total) else "NA")
    col3.metric("Posibles atendidos Valle", f"{posibles_valle:,.0f}" if pd.notna(posibles_valle) else "NA")
    col4.metric("Afiliados Valle del Cauca", f"{valle_total:,.0f}" if pd.notna(valle_total) else "NA")

    if isinstance(eps_view, pd.DataFrame) and not eps_view.empty:
        st.dataframe(eps_view, width='stretch')
        st.caption("Atendidos = ICB atendidos + Grupo Foscal atendidos.")
    else:
        st.warning("No hay tabla comparativa EPS disponible.")

    divider()
    section_header("Distribucion por edad", "Calculado con EPS_Edad + Prevalencia")
    if isinstance(edad_view, pd.DataFrame) and not edad_view.empty:
        st.dataframe(edad_view, width='stretch')
    else:
        st.warning("No hay distribucion por edad disponible.")

    divider()
    section_header("Formulas de calculo", "Guia de interpretacion")
    if isinstance(formula_df, pd.DataFrame) and not formula_df.empty:
        st.dataframe(formula_df, width='stretch')
    else:
        st.warning("No hay tabla de formulas disponible.")

    if isinstance(edad_chart_df, pd.DataFrame) and not edad_chart_df.empty:
        chart_df = edad_chart_df.copy()
        chart_df["PacientesPorEdad"] = pd.to_numeric(chart_df["PacientesPorEdad"], errors="coerce")
        chart_df = chart_df.dropna(subset=["PacientesPorEdad"])
        if not chart_df.empty:
            fig = px.bar(
                chart_df,
                x="GrupoEdad",
                y="PacientesPorEdad",
                title="Pacientes por edad (Valle del Cauca)",
                labels={"GrupoEdad": "Grupo de edad", "PacientesPorEdad": "Pacientes"},
            )
            fig.update_layout(title_x=0.5, title_xanchor="center")
            fig = style_chart(fig)
            chart_container(fig)

    st.caption(
        " | ".join(
            [
                f"Comparacion: {target_sources.get('comparacion', comp_source)}",
                f"EPS_Edad: {target_sources.get('eps_edad', edad_source)}",
                f"Prevalencia: {target_sources.get('prevalencia', prev_source)}",
            ]
        )
    )

with tab_tar:
    section_header("Tarifas por escenario", "Fuente: Tarifas_Escenarios")
    explain_box(
        "Como se calcula",
        [
            "Se usa la hoja Tarifas_Escenarios.",
            "TarifaPromedio se calcula desde TARIFAS CON INCREMENTO en los 3 escenarios.",
            "En escenario 1 se completan servicios faltantes desde escenario 2 y se reescalan ponderaciones.",
        ],
    )

    if tarifas_esc_df.empty:
        st.warning("No se pudo leer la hoja Tarifas_Escenarios.")
        st.caption(f"Detalle: {tarifas_esc_source}")
    else:
        scenario = st.session_state.get("precio_scenario", PRICE_SCENARIOS[0])
        scenario_id = scenario_id_from_label(scenario)
        st.caption(f"Escenario activo: {scenario}")

        scenario_df, _ = selector_escenario_tarifas(scenario, tarifas_esc_df)
        if scenario_df.empty:
            st.warning("No se encontraron tarifas para el escenario seleccionado.")
            st.caption(f"Columnas disponibles: {[str(c) for c in tarifas_esc_df.columns]}")
        else:
            show_cols = [
                "ServicioTarifa",
                "OrigenServicio",
                "TarifaBase",
                "TARIFAS CON INCREMENTO",
                "TarifaPromedio",
                "Pacientes",
                "PctPacientes",
                "PctIntervenciones",
            ]
            rename_map = {
                "ServicioTarifa": "Servicio",
                "TarifaBase": "Tarifas",
            }
            if scenario_id == 3:
                show_cols = [
                    "ServicioTarifa",
                    "OrigenServicio",
                    "TarifaBase",
                    "INCREMENTO CALI",
                    "TARIFAS CON INCREMENTO",
                    "TarifaPromedio",
                    "Pacientes",
                    "PctPacientes",
                    "PctIntervenciones",
                ]
            view = scenario_df[show_cols].rename(columns=rename_map)

            total_row = {
                "Servicio": "TOTAL",
                "OrigenServicio": "",
                "Tarifas": np.nan,
                "TarifaPromedio": np.nan,
                "Pacientes": pd.to_numeric(view.get("Pacientes"), errors="coerce").sum(min_count=1),
                "PctPacientes": pd.to_numeric(view.get("PctPacientes"), errors="coerce").sum(min_count=1),
                "PctIntervenciones": pd.to_numeric(view.get("PctIntervenciones"), errors="coerce").sum(min_count=1),
            }
            total_row["INCREMENTO CALI"] = pd.to_numeric(view.get("INCREMENTO CALI"), errors="coerce").mean()
            total_row["TARIFAS CON INCREMENTO"] = np.nan
            view = pd.concat([view, pd.DataFrame([total_row])], ignore_index=True)

            st.dataframe(
                view.style.format(
                    {
                        "Tarifas": lambda v: "" if pd.isna(v) else f"${v:,.0f}",
                        "TarifaPromedio": lambda v: "" if pd.isna(v) else f"${v:,.0f}",
                        "INCREMENTO CALI": lambda v: "" if pd.isna(v) else f"{v:.2%}",
                        "TARIFAS CON INCREMENTO": lambda v: "" if pd.isna(v) else f"${v:,.0f}",
                        "Pacientes": "{:,.0f}",
                        "PctPacientes": fmt_percent,
                        "PctIntervenciones": fmt_percent,
                    }
                ),
                width='stretch',
            )

with tab_sant:
    section_header("Sede Santander (Estados de resultados)")
    explain_box(
        "Como se calcula",
        [
            "Se anualizan los estados mensuales de la hoja SANTANDER.",
            "Se filtran cuentas no contables (% y ratios).",
        ],
    )
    if sant_df.empty:
        st.warning("No se pudo leer la hoja SANTANDER.")
        st.caption(f"Detalle: {sant_source}")
        st.stop()

    account_col = sant_df.columns[0]
    month_cols = [c for c in sant_df.columns[1:] if is_date_like(c)]
    if not month_cols:
        st.warning("No se detectaron columnas mensuales con fechas en SANTANDER.")
        st.caption(f"Columnas: {[str(c) for c in sant_df.columns]}")
        st.stop()

    sant_work = filter_valid_accounts(sant_df, account_col)
    long_df = sant_work[[account_col] + month_cols].melt(
        id_vars=account_col, var_name="Month", value_name="Valor"
    )
    long_df["Valor"] = pd.to_numeric(long_df["Valor"], errors="coerce")
    long_df["Year"] = pd.to_datetime(long_df["Month"], errors="coerce").dt.year
    long_df = long_df.dropna(subset=["Year"])

    annual_df = (
        long_df.groupby([account_col, "Year"], dropna=False)["Valor"]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={account_col: "Cuenta"})
    )

    if annual_df.empty:
        st.warning("No se pudo construir la tabla anual de Santander.")
        st.stop()

    annual_pivot = annual_df.pivot_table(
        index="Cuenta", columns="Year", values="Valor", aggfunc="sum"
    ).sort_index()

    ingresos_series = pick_ingresos_series(annual_pivot)
    if ingresos_series is None:
        st.warning("No se encontro la cuenta 'Ingresos' en Santander.")
        st.stop()

    ingresos_series = ingresos_series.sort_index()

    yoy = ingresos_series / ingresos_series.shift(1) - 1
    growth_avg = weighted_growth(yoy)

    section_header("Ingresos anuales y crecimiento YoY")
    explain_box(
        "Como se calcula",
        [
            "Ingresos anuales a partir de suma mensual.",
            "YoY = crecimiento anual de ingresos.",
        ],
    )
    ingresos_view = pd.DataFrame(
        {"Ingresos": ingresos_series, "YoY": yoy}
    ).reset_index().rename(columns={"Year": "Año"})
    st.dataframe(
        ingresos_view.style.format({"Ingresos": fmt_currency, "YoY": fmt_percent}),
        width='stretch',
    )
    st.caption("Cifras en millones. YoY = crecimiento anual de ingresos.")

    section_header("Proporciones vs ingresos (historico)")
    explain_box(
        "Como se calcula",
        [
            "Cada cuenta dividida por ingresos del año.",
            "Se calcula promedio histórico como referencia.",
        ],
    )
    ratio_df = annual_pivot.div(ingresos_series, axis=1).replace([np.inf, -np.inf], np.nan)
    ratio_df["Promedio"] = ratio_df.mean(axis=1, skipna=True)
    st.dataframe(ratio_df.reset_index(), width='stretch')

    st.caption("La estructura de proyeccion anual se muestra en el tab EEFF.")

with tab_share:
    section_header("Market share proyectado", "IPS + proyecto (2026-2030)")
    explain_box(
        "Como se calcula",
        [
            "IPS crecen con tasa de mercado; proyecto con tarifas y crecimiento.",
            "Market share = ingresos entidad / (IPS total + proyecto).",
        ],
    )

    proj_totals_share = st.session_state.get("proj_totals", pd.DataFrame())
    total_year1_share = st.session_state.get("total_year1", np.nan)
    proj_years_share = st.session_state.get("proj_years")

    if proj_totals_share.empty:
        st.warning("No hay proyecciones de ventas por tarifas disponibles.")
        st.stop()

    year_col = pick_year_col(proj_totals_share)
    if year_col is None:
        st.warning("No se pudo detectar la columna de anio en proyecciones.")
        st.stop()

    if proj_years_share is None:
        proj_years_share = sorted(proj_totals_share[year_col].dropna().astype(int).unique())

    market_growth = st.slider(
        "Crecimiento mercado",
        0.02,
        0.12,
        0.06,
        key="market_growth_share",
    )

    target_year = st.selectbox(
        "Anio objetivo para llegar al potencial del Valle",
        proj_years_share,
        index=len(proj_years_share) - 1,
        key="target_year_share",
    )

    growth_required = None
    if (
        pd.notna(objetivo_valle)
        and pd.notna(total_year1_share)
        and total_year1_share > 0
        and objetivo_valle > 0
    ):
        steps_to_target = int(target_year) - int(proj_years_share[0])
        if steps_to_target > 0:
            growth_required = (objetivo_valle / total_year1_share) ** (1 / steps_to_target) - 1
    if growth_required is not None and pd.notna(growth_required):
        st.caption(
            f"Tasa requerida para llegar al objetivo en {target_year}: {growth_required:.2%}."
        )
        if st.button(f"Ajustar crecimiento para llegar al objetivo en {target_year}"):
            st.session_state["pending_growth_rate"] = float(growth_required)
            st.rerun()
    else:
        st.caption("No se pudo calcular la tasa requerida para el objetivo.")

    our_rev_mn = proj_totals_share.set_index(year_col)["Ventas"] / 1_000_000

    ips_df, ips_source = load_eps_financials("IPS_EEFF")
    if ips_df.empty:
        st.warning("No se encontro IPS_EEFF.")
        st.caption(f"Detalle: {ips_source}")
        st.stop()

    ips_year_cols = [c for c in ips_df.columns if str(c).strip().isdigit()]
    ips_year_cols = sorted(ips_year_cols, key=lambda x: int(x))
    base_year = 2024 if 2024 in ips_year_cols else (int(ips_year_cols[-1]) if ips_year_cols else None)
    if base_year is None:
        st.warning("No hay columnas de anio en IPS_EEFF.")
        st.stop()

    if "IPS" in ips_df.columns and "EPS_clean" not in ips_df.columns:
        ips_df["EPS_clean"] = ips_df["IPS"].astype(str).str.replace(".xlsx", "", regex=False).str.strip()

    ips_df["CUENTA_norm"] = ips_df["CUENTA"].astype(str).map(normalize_account)
    income_accounts = ["ingresos netos por ventas", "total ingreso operativo"]
    work = ips_df[ips_df["CUENTA_norm"].isin(income_accounts)].copy()
    if work.empty:
        st.warning("No se encontraron cuentas de ingresos en IPS_EEFF.")
        st.stop()

    long_df = work.melt(
        id_vars=["EPS_clean", "CUENTA_norm"],
        value_vars=ips_year_cols,
        var_name="year",
        value_name="REV",
    )
    long_df["REV"] = pd.to_numeric(long_df["REV"], errors="coerce")
    long_df["year"] = long_df["year"].astype(str).str.strip().astype(int)

    order_map = {acc: idx for idx, acc in enumerate(income_accounts)}
    long_df["order"] = long_df["CUENTA_norm"].map(order_map)
    grouped = (
        long_df[long_df["year"] == base_year]
        .groupby(["EPS_clean", "year", "CUENTA_norm"], dropna=False)["REV"]
        .sum(min_count=1)
        .reset_index()
    )
    grouped["order"] = grouped["CUENTA_norm"].map(order_map)
    grouped = (
        grouped.sort_values("order")
        .groupby(["EPS_clean", "year"], as_index=False)
        .first()
    )
    rev_base = grouped[["EPS_clean", "REV"]].rename(columns={"EPS_clean": "IPS"})
    rev_base = rev_base.dropna(subset=["REV"])

    ips_proj = rev_base.copy()
    for year in proj_years_share:
        ips_proj[year] = ips_proj["REV"] * (1 + market_growth) ** (year - base_year)
    ips_proj = ips_proj.drop(columns=["REV"])

    ips_total_by_year = ips_proj[proj_years_share].sum()
    market_total = ips_total_by_year + our_rev_mn.reindex(proj_years_share).fillna(0)

    share_df = ips_proj.copy()
    for year in proj_years_share:
        share_df[year] = share_df[year] / market_total[year]

    our_row = {"IPS": "NUESTRO PROYECTO"}
    for year in proj_years_share:
        our_row[year] = our_rev_mn.get(year, 0.0) / market_total[year]
    share_df = pd.concat([share_df, pd.DataFrame([our_row])], ignore_index=True)

    year_view = st.selectbox(
        "Anio para grafica de market share",
        proj_years_share,
        index=0,
        key="market_share_year_view",
    )

    chart_df = share_df[["IPS", year_view]].copy()
    chart_df["Tipo"] = np.where(
        chart_df["IPS"] == "NUESTRO PROYECTO", "Proyecto", "IPS"
    )
    chart_df = chart_df.sort_values(year_view, ascending=True)
    fig = px.bar(
        chart_df,
        x=year_view,
        y="IPS",
        color="Tipo",
        orientation="h",
        text=year_view,
        title=f"Market share {year_view}",
        color_discrete_map={"Proyecto": "#c25416", "IPS": "#0f6a62"},
    )
    fig.update_xaxes(tickformat=".0%")
    fig = style_chart(fig)
    chart_container(fig)

    section_header("Tabla Market Share (IPS + proyecto)")
    explain_box(
        "Como se calcula",
        [
            "Tabla IPS x Año con participación %.",
            "Incluye fila TOTAL para validar ~100% por año.",
        ],
    )
    total_row = {"IPS": "TOTAL"}
    for year in proj_years_share:
        total_row[year] = share_df[year].sum(skipna=True)
    share_df = pd.concat([share_df, pd.DataFrame([total_row])], ignore_index=True)
    st.dataframe(
        share_df.style.format({year: "{:.2%}" for year in proj_years_share}),
        width='stretch',
    )



