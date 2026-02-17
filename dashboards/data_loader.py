from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
CALI_ANALISIS_FILE = "Cali ANALISIS.xlsx"


def _sheet_key(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch not in {" ", "_"})


def _resolve_sheet_name(path: Path, preferred_sheet: str | int | None) -> str | int | None:
    xls = pd.ExcelFile(path)
    names = xls.sheet_names
    if preferred_sheet is None:
        return 0
    if isinstance(preferred_sheet, int):
        return preferred_sheet if 0 <= preferred_sheet < len(names) else 0
    if preferred_sheet in names:
        return preferred_sheet
    candidates = [
        preferred_sheet,
        preferred_sheet.replace(" ", "_"),
        preferred_sheet.replace("_", " "),
    ]
    for candidate in candidates:
        if candidate in names:
            return candidate
    preferred_key = _sheet_key(preferred_sheet)
    for name in names:
        if _sheet_key(name) == preferred_key:
            return name
    return None


def _read_excel(path: Path, preferred_sheet: str | int | None = None) -> Tuple[pd.DataFrame, str | int]:
    xls = pd.ExcelFile(path)
    sheet = _resolve_sheet_name(path, preferred_sheet)
    if sheet is None:
        sheet = 0 if xls.sheet_names else 0
    df = xls.parse(sheet)
    return df, sheet


def _load_excel(filename: str, preferred_sheet: str | int | None = None) -> Tuple[pd.DataFrame, str]:
    path = RAW_DIR / filename
    if not path.exists():
        alt = ROOT_DIR / filename
        if alt.exists():
            path = alt
    if not path.exists():
        return pd.DataFrame(), "Archivo no encontrado"
    try:
        df, sheet_used = _read_excel(path, preferred_sheet)
        sheet_label = f"{sheet_used}" if isinstance(sheet_used, int) else sheet_used
        return df, f"Excel: {path.name} (hoja {sheet_label})"
    except Exception:
        return pd.DataFrame(), f"Error leyendo {path.name}"


def load_client_mix() -> Tuple[pd.DataFrame, str]:
    return _load_excel("clientes.xlsx", preferred_sheet="mix")


def load_client_funnel() -> Tuple[pd.DataFrame, str]:
    return _load_excel("clientes.xlsx", preferred_sheet="funnel")


def load_competitors() -> Tuple[pd.DataFrame, str]:
    return _load_excel("competencia.xlsx", preferred_sheet="competencia")


def load_financials() -> Tuple[pd.DataFrame, str]:
    return _load_excel("proyecciones.xlsx", preferred_sheet="proyecciones")


def load_market() -> Tuple[pd.DataFrame, str]:
    return _load_excel("mercado.xlsx", preferred_sheet="mercado")


def load_cifras_eps(sheet: str) -> Tuple[pd.DataFrame, str]:
    filename = CALI_ANALISIS_FILE
    path = RAW_DIR / filename
    if not path.exists():
        alt = ROOT_DIR / filename
        if alt.exists():
            path = alt
    if not path.exists():
        return pd.DataFrame(), "Archivo no encontrado"
    try:
        df = pd.read_excel(path, sheet_name=sheet)
        df.columns = [str(c).strip() for c in df.columns]
        return df, f"Excel: {path.name} (hoja {sheet})"
    except Exception as exc:
        return pd.DataFrame(), f"Error leyendo {path.name}: {exc}"


def load_cifras_eps_raw(sheet: str, header: int | None = None) -> Tuple[pd.DataFrame, str]:
    filename = CALI_ANALISIS_FILE
    path = RAW_DIR / filename
    if not path.exists():
        alt = ROOT_DIR / filename
        if alt.exists():
            path = alt
    if not path.exists():
        return pd.DataFrame(), "Archivo no encontrado"
    try:
        df = pd.read_excel(path, sheet_name=sheet, header=header)
        df.columns = [str(c).strip() for c in df.columns]
        return df, f"Excel: {path.name} (hoja {sheet})"
    except Exception as exc:
        return pd.DataFrame(), f"Error leyendo {path.name}: {exc}"


def load_eps_financials(sheet: str = "EPS_EEFF") -> Tuple[pd.DataFrame, str]:
    filename = CALI_ANALISIS_FILE
    path = RAW_DIR / filename
    if not path.exists():
        alt = ROOT_DIR / filename
        if alt.exists():
            path = alt
    if not path.exists():
        return pd.DataFrame(), "Archivo no encontrado"
    try:
        sheet_used = _resolve_sheet_name(path, sheet)
        if sheet_used is None:
            xls = pd.ExcelFile(path)
            return pd.DataFrame(), (
                f"Hoja no encontrada: {sheet}. Hojas disponibles: {', '.join(xls.sheet_names)}"
            )

        df_raw = pd.read_excel(path, sheet_name=sheet_used, header=None)

        def clean_cell(value: object) -> str:
            if value is None:
                return ""
            return str(value).strip()

        header_row = None
        for idx, row in df_raw.iterrows():
            first = clean_cell(row.iloc[0]) if len(row) > 0 else ""
            second = clean_cell(row.iloc[1]) if len(row) > 1 else ""
            if second == "CUENTA" and first in {"EPS", "IPS"}:
                header_row = idx
                break

        if header_row is None:
            header_row = 0

        df = pd.read_excel(path, sheet_name=sheet_used, header=header_row)
        df.columns = [str(c).strip() for c in df.columns]

        def normalize_text(series: pd.Series) -> pd.Series:
            return series.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

        if "EPS" in df.columns:
            df["EPS"] = normalize_text(df["EPS"])
            df["EPS_clean"] = (
                df["EPS"]
                .str.replace(".xlsx", "", regex=False)
                .str.strip()
                .replace({"SANITAS": "EPS SANITAS"})
            )
        if "IPS" in df.columns:
            df["IPS"] = normalize_text(df["IPS"])
            df["EPS_clean"] = df["IPS"].str.replace(".xlsx", "", regex=False).str.strip()
        if "CUENTA" in df.columns:
            df["CUENTA"] = normalize_text(df["CUENTA"])
        # Trim all string cells to remove indentation
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).str.strip()
        return df, f"Excel: {path.name} (hoja {sheet_used})"
    except Exception as exc:
        return pd.DataFrame(), f"Error leyendo {path.name}: {exc}"


def load_capacidad_objetivo() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Equipo": [
                "Angiografos",
                "Quirofanos",
                "Camas Hospitalizacion",
                "Camas UCI",
                "Camas Intermedia",
            ],
            "Cantidad": [2, 6, 80, 40, 40],
        }
    )


def load_capacidad_dotacion() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Servicio": [
                "HEMODINAMIA",
                "ELECTROFISIOLOGIA",
                "VASCULAR",
                "NEUROINTERVENCIONISMO",
                "CIRUGIA",
            ],
            "Modalidad": ["INTRAMURAL"] * 5,
            "Complejidad": ["ALTA"] * 5,
            "Equipo": [
                "ANGIOGRAFO",
                "ANGIOGRAFO",
                "VASCULAR",
                "NEUROINTERVENCIONISMO",
                "DOTACION DE QUIROFANO",
            ],
            "Cantidad_equipo": [1, 1, 1, 1, 1],
            "Pacientes_por_dia": [10.0, 7.0, 1.5, 2.0, 1.5],
            "Dias_semana": [4.0, 1.0, 2.0, 1.0, 5.0],
            "Capacidad_semanal": [40.0, 6.7, 3.0, 2.0, 7.5],
            "Capacidad_mensual": [160.0, 26.7, 12.0, 8.0, 30.0],
            "Pacientes_atendidos_mensual": [84.0, 24.0, 16.0, 11.0, 15.0],
            "Proporcion_uso": [0.525, 0.9, 1.333, 1.375, 0.5],
        }
    )
