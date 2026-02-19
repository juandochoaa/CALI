from __future__ import annotations

import unicodedata
from itertools import combinations
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

EPS_OBJ_DEFAULT: List[str] = [
    "ASMETSALUD EPS",
    "COMFENALCO VALLE EPS",
    "COOSALUD EPS",
    "EMSSANAR EPS",
    "EPS SANITAS",
    "EPS SOS",
    "EPS SURA",
    "FAMISANAR",
    "FERRONALES - EAS",
    "MALLAMAS EPSI",
    "NUEVA EPS",
    "SALUD TOTAL EPS",
]

PROBABILITY_COLUMNS: List[str] = [
    "P_avg_CM_ratio_lt_1",
    "P_avg_PA_ratio_lt_1",
    "P_avg_RI_ratio_lt_1",
    "P_avg_CashDays_lt_15",
    "P_avg_PayDays_out_20_70",
]


def _norm_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.lower().split()).strip()


def _norm_compact(value: Any) -> str:
    return _norm_text(value).replace(" ", "")


def _canonicalize_eps(series: pd.Series, eps_obj: Sequence[str]) -> pd.Series:
    mapping = {_norm_text(eps): eps for eps in eps_obj}
    return (
        series.astype(str)
        .str.strip()
        .map(lambda x: mapping.get(_norm_text(x), str(x).strip()))
    )


def _canonicalize_eps_value(value: Any, eps_obj: Sequence[str]) -> str:
    text = str(value).strip()
    if not text:
        return text

    base_norm = _norm_text(text)
    base_compact = base_norm.replace(" ", "")
    variants = {
        base_norm,
        base_compact,
        _norm_text(base_norm.replace(" eps", "")),
        _norm_text(base_norm.replace("eps ", "")),
        _norm_text(base_norm + " eps"),
        _norm_text(("eps " + base_norm).strip()),
        _norm_text(base_norm.replace("-", " ")),
        _norm_text(base_norm.replace(".", " ")),
    }
    variants = {v for v in variants if v}

    mapping: Dict[str, str] = {}
    for eps in eps_obj:
        eps_norm = _norm_text(eps)
        eps_compact = eps_norm.replace(" ", "")
        keys = {
            eps_norm,
            eps_compact,
            _norm_text(eps_norm.replace(" eps", "")),
            _norm_text(eps_norm.replace("eps ", "")),
        }
        for key in keys:
            if key:
                mapping[key] = eps

    for key in variants:
        if key in mapping:
            return mapping[key]
    return text


def _to_long_year(df: pd.DataFrame, id_vars: List[str], value_name: str) -> pd.DataFrame:
    year_cols = [c for c in df.columns if str(c).strip().isdigit()]
    out = df.melt(
        id_vars=id_vars,
        value_vars=year_cols,
        var_name="Año",
        value_name=value_name,
    )
    out["Año"] = pd.to_numeric(out["Año"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["Año"]).copy()
    out["Año"] = out["Año"].astype(int)
    return out


def _resolve_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    norm_map = {_norm_compact(c): c for c in df.columns}
    for candidate in candidates:
        col = norm_map.get(_norm_compact(candidate))
        if col is not None:
            return col
    return None


def _resolve_year_column(df: pd.DataFrame) -> str:
    for candidate in ["Año", "AÃ±o", "AÃƒÂ±o", "Ano", "year", "Year"]:
        if candidate in df.columns:
            return candidate
    for col in df.columns:
        norm = _norm_text(col)
        if norm in {"ano", "year"}:
            return col
    for col in df.columns:
        if str(col).strip().lower() == "eps":
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        valid = series.dropna()
        if valid.empty:
            continue
        if valid.between(1900, 2100).all() and valid.nunique() >= 2:
            return col
    raise KeyError("No se encontro columna de ano.")


def _numeric_series(df: pd.DataFrame, candidates: Iterable[str], default: float = np.nan) -> pd.Series:
    col = _resolve_column(df, candidates)
    if col is None:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _rnorm_trunc(
    rng: np.random.Generator,
    mean: float,
    std: float,
    low: float,
    high: float,
    size: int,
) -> np.ndarray:
    if not np.isfinite(std) or std == 0:
        x = np.full(size, mean, dtype=float)
    else:
        x = rng.normal(mean, std, size=size)
    return np.clip(x, low, high)


def _safe_div(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    return np.divide(
        numer,
        denom,
        out=np.full_like(numer, np.nan, dtype=float),
        where=denom != 0,
    )


def _build_capmin_map(eps_list: Sequence[str]) -> Dict[str, float]:
    capmin_map = {str(eps): 19500.0 for eps in eps_list}
    if "ASMETSALUD EPS" in capmin_map:
        capmin_map["ASMETSALUD EPS"] = 17800.0
    if "EMSSANAR EPS" in capmin_map:
        capmin_map["EMSSANAR EPS"] = 17800.0
    return capmin_map


def _truncate_wide_year_columns(df: pd.DataFrame, max_year: int) -> pd.DataFrame:
    keep_cols: List[Any] = []
    for col in df.columns:
        col_str = str(col).strip()
        if col_str.isdigit():
            if int(col_str) <= int(max_year):
                keep_cols.append(col)
        else:
            keep_cols.append(col)
    return df[keep_cols].copy()


def _compute_eps_actual_breaches(
    eps_eeff_df: pd.DataFrame,
    eps_obj: Sequence[str],
    cash_thresholds: Sequence[int] = (15,),
    paydays_range: tuple[float, float] = (20.0, 70.0),
) -> pd.DataFrame:
    eeff_wide = _build_eps_eeff_wide(eps_eeff_df=eps_eeff_df, eps_obj=eps_obj)
    if eeff_wide.empty:
        return pd.DataFrame(
            columns=[
                "EPS",
                "AÃ±o",
                "Obs_CM_ratio_lt_1",
                "Obs_PA_ratio_lt_1",
                "Obs_RI_ratio_lt_1",
                "Obs_PayDays_out_20_70",
            ]
            + [f"Obs_CashDays_lt_{int(thr)}" for thr in cash_thresholds]
        )

    year_col = _resolve_year_column(eeff_wide)
    work = eeff_wide.copy()
    work["Ingresos"] = _numeric_series(work, ["TotalIngresoOperativo", "Ingresosnetosporventas"])
    work["Costo"] = _numeric_series(work, ["Otroscostospornaturaleza"])
    work["GAdmin"] = _numeric_series(work, ["Gastosadministrativos"])
    work["Cash"] = _numeric_series(work, ["EfectivooEquivalentes"])
    work["Equity"] = _numeric_series(work, ["Totaldepatrimonio"])
    work["ReservasProxy"] = _numeric_series(work, ["Provisionesparaotrospasivosygastos"])
    work["InvLiquidas"] = _numeric_series(work, ["Activosfinancierosdecortoplazo"], default=0.0).fillna(0.0)

    opex_component_map = {
        "Gastosporbeneficiosdelosempleado": [
            "Gastosporbeneficiosdelosempleado",
            "Gastosporbeneficiosdelosempleados",
        ],
        "Costosdetransporte": ["Costosdetransporte"],
        "Impuestoycontribuciones": ["Impuestoycontribuciones"],
        "Otrosgastos": ["Otrosgastos"],
    }
    for new_col, candidates in opex_component_map.items():
        work[new_col] = _numeric_series(work, candidates, default=0.0).fillna(0.0)

    work["OPEX_other"] = work[list(opex_component_map.keys())].sum(axis=1)
    work["OpExCash"] = work["Costo"] + work["GAdmin"] + work["OPEX_other"]
    work["CxP_Comercial"] = _commercial_payables_series(work)
    work["LR"] = _safe_div(work["Costo"].to_numpy(dtype=float), work["Ingresos"].to_numpy(dtype=float))

    low = float(paydays_range[0])
    high = float(paydays_range[1])
    if low > high:
        low, high = high, low

    capmin_map = _build_capmin_map(work["EPS"].dropna().astype(str).unique().tolist())
    work["CapMinReq"] = work["EPS"].map(capmin_map).astype(float)

    work["CM_ratio"] = _safe_div(
        work["Equity"].to_numpy(dtype=float),
        work["CapMinReq"].to_numpy(dtype=float),
    )
    pa_req = 0.08 * work["Ingresos"].to_numpy(dtype=float) * work["LR"].to_numpy(dtype=float)
    work["PA_ratio"] = _safe_div(work["Equity"].to_numpy(dtype=float), pa_req)
    work["RI_ratio"] = _safe_div(
        (work["Cash"] + work["InvLiquidas"]).to_numpy(dtype=float),
        work["ReservasProxy"].to_numpy(dtype=float),
    )
    work["CashDays"] = 365.0 * _safe_div(
        work["Cash"].to_numpy(dtype=float),
        work["OpExCash"].to_numpy(dtype=float),
    )
    work["PayDays"] = 365.0 * _safe_div(
        work["CxP_Comercial"].to_numpy(dtype=float),
        work["OpExCash"].to_numpy(dtype=float),
    )

    out = pd.DataFrame(
        {
            "EPS": work["EPS"].astype(str),
            "AÃ±o": pd.to_numeric(work[year_col], errors="coerce").astype("Int64"),
            "Obs_CM_ratio_lt_1": (work["CM_ratio"] < 1.0).fillna(False).astype(float),
            "Obs_PA_ratio_lt_1": (work["PA_ratio"] < 1.0).fillna(False).astype(float),
            "Obs_RI_ratio_lt_1": (work["RI_ratio"] < 1.0).fillna(False).astype(float),
            f"Obs_PayDays_out_{int(low)}_{int(high)}": (
                ((work["PayDays"] < low) | (work["PayDays"] > high))
                .fillna(False)
                .astype(float)
            ),
        }
    )
    for thr in cash_thresholds:
        out[f"Obs_CashDays_lt_{int(thr)}"] = (work["CashDays"] < float(thr)).fillna(False).astype(float)

    out = out.dropna(subset=["AÃ±o"]).copy()
    out["AÃ±o"] = out["AÃ±o"].astype(int)
    out = out.sort_values(["EPS", "AÃ±o"]).reset_index(drop=True)
    return out


def _build_eps_eeff_wide(
    eps_eeff_df: pd.DataFrame,
    eps_obj: Sequence[str],
) -> pd.DataFrame:
    eps_eeff = eps_eeff_df.copy()
    eps_eeff.columns = [str(c).strip() for c in eps_eeff.columns]
    eps_eeff["EPS"] = _canonicalize_eps(eps_eeff["EPS"], eps_obj)
    eps_eeff = eps_eeff[eps_eeff["EPS"].isin(eps_obj)].copy()
    eeff_long = _to_long_year(eps_eeff, id_vars=["EPS", "CUENTA"], value_name="Valor")
    eeff_long["EPS"] = eeff_long["EPS"].astype(str).str.strip()
    eeff_long["CUENTA"] = eeff_long["CUENTA"].astype(str).str.strip()
    eeff_long["Valor"] = pd.to_numeric(eeff_long["Valor"], errors="coerce")

    # Resolve year column dynamically to handle encoding variants of year labels.
    year_col = _resolve_year_column(eeff_long)
    eeff_wide = (
        eeff_long.pivot_table(
            index=["EPS", year_col],
            columns="CUENTA",
            values="Valor",
            aggfunc="sum",
        )
        .reset_index()
        .sort_values(["EPS", year_col])
        .reset_index(drop=True)
    )
    return eeff_wide


def _commercial_payables_series(df: pd.DataFrame) -> pd.Series:
    commercial_primary = _numeric_series(df, ["Comercialesyotrascuentasapagar"])
    if commercial_primary.notna().any():
        return commercial_primary
    return _numeric_series(
        df,
        [
            "CuentasComercialesporpagar",
            "Cuentascomercialesporpagar",
        ],
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def _recent_mean(series: pd.Series, recent_years: int = 3) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.tail(recent_years).mean())


def _build_mean_reversion_params(
    hist_df: pd.DataFrame,
    eps_list: Sequence[str],
    recent_years: int = 3,
) -> pd.DataFrame:
    year_col = _resolve_year_column(hist_df)
    rows: List[Dict[str, Any]] = []
    for eps in eps_list:
        eps_hist = hist_df[hist_df["EPS"] == eps].copy()
        eps_hist = eps_hist.sort_values(year_col)

        mu_long_g = _safe_float(pd.to_numeric(eps_hist["g"], errors="coerce").mean(), default=0.0)
        mu_long_lr = _safe_float(pd.to_numeric(eps_hist["LR"], errors="coerce").mean(), default=0.0)
        mu_recent_g = _safe_float(_recent_mean(eps_hist["g"], recent_years=recent_years), default=mu_long_g)
        mu_recent_lr = _safe_float(
            _recent_mean(eps_hist["LR"], recent_years=recent_years),
            default=mu_long_lr,
        )

        sigma_long_g = _safe_float(pd.to_numeric(eps_hist["g"], errors="coerce").std(ddof=1), default=0.0)
        sigma_long_lr = _safe_float(pd.to_numeric(eps_hist["LR"], errors="coerce").std(ddof=1), default=0.0)

        lambda_g = float(
            np.clip(
                0.05 + 0.10 * (abs(mu_recent_g - mu_long_g) / (sigma_long_g + 1e-6)),
                0.05,
                0.25,
            )
        )
        lambda_lr = float(
            np.clip(
                0.05 + 0.10 * (abs(mu_recent_lr - mu_long_lr) / (sigma_long_lr + 1e-6)),
                0.05,
                0.25,
            )
        )

        rows.append(
            {
                "EPS": eps,
                "mu_long_g": mu_long_g,
                "mu_recent_g": mu_recent_g,
                "sigma_long_g": sigma_long_g,
                "lambda_g": lambda_g,
                "mu_long_lr": mu_long_lr,
                "mu_recent_lr": mu_recent_lr,
                "sigma_long_lr": sigma_long_lr,
                "lambda_lr": lambda_lr,
            }
        )

    return pd.DataFrame(rows).sort_values("EPS").reset_index(drop=True)


def _percentile_score_low_is_better(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    score = pd.Series(np.nan, index=series.index, dtype=float)
    valid = values.dropna()
    n = len(valid)
    if n == 0:
        return score
    if n == 1:
        score.loc[valid.index] = 100.0
        return score
    rank_pos = valid.rank(method="average", ascending=True) - 1.0
    score.loc[valid.index] = (1.0 - (rank_pos / (n - 1.0))) * 100.0
    return score


def _percentile_score_high_is_better(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    score = pd.Series(np.nan, index=series.index, dtype=float)
    valid = values.dropna()
    n = len(valid)
    if n == 0:
        return score
    if n == 1:
        score.loc[valid.index] = 100.0
        return score
    rank_pos = valid.rank(method="average", ascending=True) - 1.0
    score.loc[valid.index] = (rank_pos / (n - 1.0)) * 100.0
    return score


def _build_upc_projection_by_segment(
    upc_long_df: pd.DataFrame,
    eps_mix_df: pd.DataFrame,
    end_year: int,
    optimism_factor: float,
    growth_start_year: int,
    growth_end_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seg_hist = upc_long_df[["Regimen", "GrupoEdad_UPC", "Año", "UPC"]].copy()
    seg_hist["Año"] = pd.to_numeric(seg_hist["Año"], errors="coerce")
    seg_hist["UPC"] = pd.to_numeric(seg_hist["UPC"], errors="coerce")
    seg_hist = seg_hist.dropna(subset=["Año", "UPC"]).copy()
    seg_hist["Año"] = seg_hist["Año"].astype(int)
    seg_hist = seg_hist.sort_values(["Regimen", "GrupoEdad_UPC", "Año"]).reset_index(drop=True)

    seg_rows: List[dict[str, Any]] = []
    growth_rows: List[dict[str, Any]] = []

    for (regimen, grupo_edad), gdf in seg_hist.groupby(["Regimen", "GrupoEdad_UPC"], dropna=False):
        gdf = gdf.sort_values("Año").copy()
        years = sorted(gdf["Año"].unique().tolist())
        if not years:
            continue

        y0 = growth_start_year if growth_start_year in years else int(min(years))
        y1 = growth_end_year if growth_end_year in years else int(max(years))

        if y1 > y0:
            v0 = float(gdf.loc[gdf["Año"] == y0, "UPC"].iloc[0])
            v1 = float(gdf.loc[gdf["Año"] == y1, "UPC"].iloc[0])
            g_base = (v1 / v0) ** (1.0 / (y1 - y0)) - 1.0 if v0 > 0 else 0.0
        else:
            g_base = 0.0

        g_adj = g_base * optimism_factor
        gf = max(0.01, 1.0 + g_adj)

        growth_rows.append(
            {
                "Regimen": regimen,
                "GrupoEdad_UPC": grupo_edad,
                "g_base_pct": g_base * 100.0,
                "g_adj_pct": g_adj * 100.0,
                "factor_anual": gf,
            }
        )

        hist_rows = gdf[["Regimen", "GrupoEdad_UPC", "Año", "UPC"]].rename(
            columns={"UPC": "UPC_segmento"}
        )
        seg_rows.extend(hist_rows.to_dict("records"))

        last_year = int(max(years))
        last_val = float(gdf.loc[gdf["Año"] == last_year, "UPC"].iloc[-1])
        for year in range(last_year + 1, end_year + 1):
            last_val = last_val * gf
            seg_rows.append(
                {
                    "Regimen": regimen,
                    "GrupoEdad_UPC": grupo_edad,
                    "Año": year,
                    "UPC_segmento": last_val,
                }
            )

    upc_segment_full = (
        pd.DataFrame(seg_rows)
        .drop_duplicates(["Regimen", "GrupoEdad_UPC", "Año"], keep="last")
        .sort_values(["Regimen", "GrupoEdad_UPC", "Año"])
        .reset_index(drop=True)
    )

    eps_weights = eps_mix_df[["EPS", "Regimen", "GrupoEdad_UPC", "Peso"]].copy()
    upc_eps_full = eps_weights.merge(
        upc_segment_full,
        on=["Regimen", "GrupoEdad_UPC"],
        how="left",
    )
    upc_eps_full = upc_eps_full[upc_eps_full["UPC_segmento"].notna()].copy()
    upc_eps_full["UPC_pond"] = upc_eps_full["Peso"] * upc_eps_full["UPC_segmento"]

    upc_mix_full = (
        upc_eps_full.groupby(["EPS", "Año"], as_index=False)["UPC_pond"]
        .sum()
        .rename(columns={"UPC_pond": "UPC_mix"})
        .sort_values(["EPS", "Año"])
        .reset_index(drop=True)
    )

    growth_df = pd.DataFrame(growth_rows).sort_values(["Regimen", "GrupoEdad_UPC"]).reset_index(drop=True)
    return upc_mix_full, growth_df


def _prepare_base_eps(
    upc_df: pd.DataFrame,
    eps_eeff_df: pd.DataFrame,
    eps_edad_df: pd.DataFrame,
    eps_afiliados_hist_df: pd.DataFrame,
    eps_obj: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    upc = upc_df.copy()
    eps_eeff = eps_eeff_df.copy()
    eps_edad = eps_edad_df.copy()
    eps_years = eps_afiliados_hist_df.copy()

    for df in [upc, eps_eeff, eps_edad, eps_years]:
        df.columns = [str(c).strip() for c in df.columns]

    eps_eeff["EPS"] = _canonicalize_eps(eps_eeff["EPS"], eps_obj)
    eps_years["EPS"] = _canonicalize_eps(eps_years["EPS"], eps_obj)
    eps_edad["EPS"] = _canonicalize_eps(eps_edad["EPS"], eps_obj)

    eps_eeff = eps_eeff[eps_eeff["EPS"].isin(eps_obj)].copy()
    eps_years = eps_years[eps_years["EPS"].isin(eps_obj)].copy()
    eps_edad = eps_edad[eps_edad["EPS"].isin(eps_obj)].copy()

    upc_long = _to_long_year(upc, id_vars=["Regimen", "GrupoEdad"], value_name="UPC")
    upc_long["Regimen"] = upc_long["Regimen"].astype(str).str.strip()
    upc_long["GrupoEdad_UPC"] = upc_long["GrupoEdad"].astype(str).str.strip()
    upc_long["UPC"] = pd.to_numeric(upc_long["UPC"], errors="coerce")
    upc_long = upc_long.drop(columns=["GrupoEdad"])

    eeff_long = _to_long_year(eps_eeff, id_vars=["EPS", "CUENTA"], value_name="Valor")
    eeff_long["EPS"] = eeff_long["EPS"].astype(str).str.strip()
    eeff_long["CUENTA"] = eeff_long["CUENTA"].astype(str).str.strip()
    eeff_long["Valor"] = pd.to_numeric(eeff_long["Valor"], errors="coerce")

    eeff_wide = (
        eeff_long.pivot_table(
            index=["EPS", "Año"],
            columns="CUENTA",
            values="Valor",
            aggfunc="sum",
        )
        .reset_index()
        .sort_values(["EPS", "Año"])
        .reset_index(drop=True)
    )

    years_long = _to_long_year(eps_years, id_vars=["EPS"], value_name="Afiliados")
    years_long["EPS"] = years_long["EPS"].astype(str).str.strip()
    years_long["Afiliados"] = pd.to_numeric(years_long["Afiliados"], errors="coerce")
    years_long = (
        years_long.groupby(["EPS", "Año"], as_index=False)["Afiliados"]
        .sum(min_count=1)
        .sort_values(["EPS", "Año"])
        .reset_index(drop=True)
    )

    edad_map = {
        "De 0 a 4 años": "1-4 años",
        "De 05 a 09 años": "5-14 años",
        "De 10 a 14 años": "5-14 años",
        "De 15 a 19 años": "15-18 años",
        "De 20 a 24 años": "19-44 años",
        "De 25 a 29 años": "19-44 años",
        "De 30 a 34 años": "19-44 años",
        "De 35 a 39 años": "19-44 años",
        "De 40 a 44 años": "19-44 años",
        "De 45 a 49 años": "45-49 años",
        "De 50 a 54 años": "50-54 años",
        "De 55 a 59 años": "55-59 años",
        "De 60 a 64 años": "60-64 años",
        "De 65 a 69 años": "65-69 años",
        "De 70 a 74 años": "70-74 años",
        "De 75 a 79 años": "75 años y mayores",
        "De 80 años o más": "75 años y mayores",
    }

    eps_edad_work = eps_edad[["EPS", "REGIMEN", "GRUPOEDAD", "TOTAL AFILIADOS"]].copy()
    eps_edad_work = eps_edad_work.rename(
        columns={
            "REGIMEN": "Regimen",
            "GRUPOEDAD": "GrupoEdad_EPS",
            "TOTAL AFILIADOS": "Afiliados",
        }
    )
    eps_edad_work["Regimen"] = eps_edad_work["Regimen"].astype(str).str.strip()
    eps_edad_work["GrupoEdad_EPS"] = eps_edad_work["GrupoEdad_EPS"].astype(str).str.strip()
    eps_edad_work["Afiliados"] = pd.to_numeric(eps_edad_work["Afiliados"], errors="coerce")
    eps_edad_work["GrupoEdad_UPC"] = eps_edad_work["GrupoEdad_EPS"].map(edad_map)
    eps_edad_work = eps_edad_work.dropna(subset=["GrupoEdad_UPC", "Afiliados"]).copy()

    eps_mix = (
        eps_edad_work.groupby(["EPS", "Regimen", "GrupoEdad_UPC"], as_index=False)["Afiliados"]
        .sum(min_count=1)
        .sort_values(["EPS", "Regimen", "GrupoEdad_UPC"])
        .reset_index(drop=True)
    )
    eps_total = (
        eps_mix.groupby("EPS", as_index=False)["Afiliados"]
        .sum(min_count=1)
        .rename(columns={"Afiliados": "Afiliados_EPS"})
    )
    eps_mix = eps_mix.merge(eps_total, on="EPS", how="left")
    eps_mix["Peso"] = eps_mix["Afiliados"] / eps_mix["Afiliados_EPS"]

    upc_mix_hist = (
        eps_mix.merge(upc_long, on=["Regimen", "GrupoEdad_UPC"], how="left")
        .assign(UPC_pond=lambda d: d["Peso"] * d["UPC"])
        .groupby(["EPS", "Año"], as_index=False)["UPC_pond"]
        .sum(min_count=1)
        .rename(columns={"UPC_pond": "UPC_mix"})
        .sort_values(["EPS", "Año"])
        .reset_index(drop=True)
    )

    base_eps = (
        eeff_wide.merge(years_long, on=["EPS", "Año"], how="left")
        .merge(upc_mix_hist, on=["EPS", "Año"], how="left")
        .sort_values(["EPS", "Año"])
        .reset_index(drop=True)
    )

    diagnostics = {
        "eps_universe": sorted(base_eps["EPS"].dropna().unique().tolist()),
        "missing_upc_mix_rows": int(base_eps["UPC_mix"].isna().sum()),
        "missing_afiliados_rows": int(base_eps["Afiliados"].isna().sum()),
    }
    return base_eps, upc_mix_hist, eps_mix, diagnostics


def run_eps_montecarlo(
    upc_df: pd.DataFrame,
    eps_eeff_df: pd.DataFrame,
    eps_edad_df: pd.DataFrame,
    eps_afiliados_hist_df: pd.DataFrame,
    eps_obj: Sequence[str] | None = None,
    n_sim: int = 10_000,
    horizon_end: int = 2030,
    cash_thresholds: Sequence[int] = (15,),
    paydays_range: tuple[float, float] = (20.0, 70.0),
    upc_optimism_factor: float = 1.2,
    upc_growth_start_year: int = 2019,
    upc_growth_end_year: int = 2026,
    scenarios: Mapping[str, Mapping[str, float]] | None = None,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    eps_obj = list(eps_obj or EPS_OBJ_DEFAULT)
    scenarios = scenarios or {
        "BASE": {"LR_shift": 0.0, "g_shift": 0.0},
        "STRESS_LR": {"LR_shift": 0.05, "g_shift": 0.0},
        "STRESS_MIX": {"LR_shift": 0.05, "g_shift": -0.03},
    }

    base_eps, _upc_mix_hist, eps_mix, prep_diag = _prepare_base_eps(
        upc_df=upc_df,
        eps_eeff_df=eps_eeff_df,
        eps_edad_df=eps_edad_df,
        eps_afiliados_hist_df=eps_afiliados_hist_df,
        eps_obj=eps_obj,
    )

    df = base_eps.copy().sort_values(["EPS", "Año"]).reset_index(drop=True)
    anchor_year = int(df["Año"].max())
    horizon_end = max(int(horizon_end), anchor_year + 1)
    years_sim = list(range(anchor_year + 1, horizon_end + 1))
    alpha = 1.0
    beta = 1.0

    df["Ingresos"] = _numeric_series(df, ["TotalIngresoOperativo", "Ingresosnetosporventas"])
    df["Costo"] = _numeric_series(df, ["Otroscostospornaturaleza"])
    df["GAdmin"] = _numeric_series(df, ["Gastosadministrativos"])
    df["Cash"] = _numeric_series(df, ["EfectivooEquivalentes"])
    df["Equity"] = _numeric_series(df, ["Totaldepatrimonio"])
    df["ReservasProxy"] = _numeric_series(df, ["Provisionesparaotrospasivosygastos"])

    opex_component_map = {
        "Gastosporbeneficiosdelosempleado": [
            "Gastosporbeneficiosdelosempleado",
            "Gastosporbeneficiosdelosempleados",
        ],
        "Costosdetransporte": ["Costosdetransporte"],
        "Impuestoycontribuciones": ["Impuestoycontribuciones"],
        "Otrosgastos": ["Otrosgastos"],
    }
    for new_col, candidates in opex_component_map.items():
        df[new_col] = _numeric_series(df, candidates, default=0.0).fillna(0.0)

    df["OPEX_other"] = df[list(opex_component_map.keys())].sum(axis=1)
    df["OpExCash"] = df["Costo"] + df["GAdmin"] + df["OPEX_other"]
    df["CxP_Comercial"] = _commercial_payables_series(df)
    df["ap_to_rev"] = _safe_div(
        df["CxP_Comercial"].to_numpy(dtype=float),
        df["Ingresos"].to_numpy(dtype=float),
    )
    df["PayDays"] = 365.0 * _safe_div(
        df["CxP_Comercial"].to_numpy(dtype=float),
        df["OpExCash"].to_numpy(dtype=float),
    )
    df["LR"] = _safe_div(df["Costo"].to_numpy(dtype=float), df["Ingresos"].to_numpy(dtype=float))
    df["AR"] = _safe_div(df["GAdmin"].to_numpy(dtype=float), df["Ingresos"].to_numpy(dtype=float))
    df["OER"] = _safe_div(df["OPEX_other"].to_numpy(dtype=float), df["Ingresos"].to_numpy(dtype=float))
    df["rho_res"] = _safe_div(df["ReservasProxy"].to_numpy(dtype=float), df["Ingresos"].to_numpy(dtype=float))
    df["g"] = df.groupby("EPS")["Afiliados"].pct_change()

    baseline = df[df["Año"] == anchor_year][
        ["EPS", "Afiliados", "Ingresos", "UPC_mix", "Cash", "Equity", "ReservasProxy"]
    ].copy()
    baseline["RevPerAff"] = _safe_div(
        baseline["Ingresos"].to_numpy(dtype=float),
        baseline["Afiliados"].to_numpy(dtype=float),
    )
    baseline["k_e"] = _safe_div(
        baseline["RevPerAff"].to_numpy(dtype=float),
        baseline["UPC_mix"].to_numpy(dtype=float),
    )
    baseline = baseline.rename(
        columns={
            "Afiliados": "Afiliados_0",
            "Cash": "Cash_0",
            "Equity": "Equity_0",
            "ReservasProxy": "ReservasProxy_0",
            "Ingresos": "Ingresos_0",
        }
    )
    baseline = baseline.drop_duplicates(subset=["EPS"], keep="last").reset_index(drop=True)

    hist = df[(df["Año"] >= 2019) & (df["Año"] <= anchor_year)].copy()
    params_eps = (
        hist.groupby("EPS")[["g", "LR", "AR", "OER", "rho_res", "ap_to_rev"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    params_eps.columns = ["EPS"] + [f"{v}_{stat}" for (v, stat) in params_eps.columns[1:]]
    ap_series_hist = pd.to_numeric(hist["ap_to_rev"], errors="coerce")
    global_ap_mean = float(ap_series_hist.mean()) if ap_series_hist.notna().any() else np.nan
    global_ap_median = float(ap_series_hist.median()) if ap_series_hist.notna().any() else np.nan
    global_ap_std = float(ap_series_hist.std(ddof=1)) if ap_series_hist.notna().any() else np.nan
    if not np.isfinite(global_ap_mean):
        global_ap_mean = global_ap_median if np.isfinite(global_ap_median) else 0.10
    if not np.isfinite(global_ap_std) or global_ap_std <= 0:
        global_ap_std = 0.05

    eps_list = baseline["EPS"].dropna().unique().tolist()
    capmin_map = _build_capmin_map(eps_list)
    capmin_2025 = pd.DataFrame({"EPS": eps_list})
    capmin_2025["CapMinReq"] = capmin_2025["EPS"].map(capmin_map)
    mean_reversion_params = _build_mean_reversion_params(hist_df=hist, eps_list=eps_list, recent_years=3)
    mean_reversion_map: Dict[str, Dict[str, float]] = {}
    for row in mean_reversion_params.to_dict("records"):
        mean_reversion_map[str(row["EPS"])] = {
            "mu_long_g": _safe_float(row.get("mu_long_g"), 0.0),
            "mu_recent_g": _safe_float(row.get("mu_recent_g"), 0.0),
            "sigma_long_g": _safe_float(row.get("sigma_long_g"), 0.0),
            "lambda_g": _safe_float(row.get("lambda_g"), 0.05),
            "mu_long_lr": _safe_float(row.get("mu_long_lr"), 0.0),
            "mu_recent_lr": _safe_float(row.get("mu_recent_lr"), 0.0),
            "sigma_long_lr": _safe_float(row.get("sigma_long_lr"), 0.0),
            "lambda_lr": _safe_float(row.get("lambda_lr"), 0.05),
        }

    upc_long = _to_long_year(upc_df.copy(), id_vars=["Regimen", "GrupoEdad"], value_name="UPC")
    upc_long["Regimen"] = upc_long["Regimen"].astype(str).str.strip()
    upc_long["GrupoEdad_UPC"] = upc_long["GrupoEdad"].astype(str).str.strip()
    upc_long = upc_long.drop(columns=["GrupoEdad"])
    upc_mix_full, upc_growth_segments = _build_upc_projection_by_segment(
        upc_long_df=upc_long,
        eps_mix_df=eps_mix,
        end_year=horizon_end,
        optimism_factor=float(upc_optimism_factor),
        growth_start_year=int(upc_growth_start_year),
        growth_end_year=int(upc_growth_end_year),
    )
    upc_dict = {
        (row["EPS"], int(row["Año"])): float(row["UPC_mix"])
        for row in upc_mix_full.to_dict("records")
    }

    inputs = (
        baseline[["EPS", "Afiliados_0", "Cash_0", "Equity_0", "k_e", "Ingresos_0"]]
        .merge(params_eps, on="EPS", how="left")
        .merge(capmin_2025, on="EPS", how="left")
    )
    inputs = inputs.drop_duplicates(subset=["EPS"], keep="last").reset_index(drop=True)

    df["InvLiquidas"] = _numeric_series(df, ["Activosfinancierosdecortoplazo"], default=0.0).fillna(0.0)
    inv_ratio_map: Dict[str, float] = {}
    for eps in eps_list:
        inv0 = df[(df["EPS"] == eps) & (df["Año"] == anchor_year)]["InvLiquidas"]
        ing0 = df[(df["EPS"] == eps) & (df["Año"] == anchor_year)]["Ingresos"]
        if len(inv0) and len(ing0) and float(ing0.iloc[0]) != 0:
            inv_ratio_map[eps] = float(inv0.iloc[0]) / float(ing0.iloc[0])
        else:
            inv_ratio_map[eps] = 0.0

    rng = np.random.default_rng(random_seed)
    results: List[dict[str, Any]] = []
    thresholds = [int(x) for x in cash_thresholds]
    paydays_low = float(paydays_range[0])
    paydays_high = float(paydays_range[1])
    if paydays_low > paydays_high:
        paydays_low, paydays_high = paydays_high, paydays_low

    for scenario_name, shock in scenarios.items():
        lr_shift = float(shock.get("LR_shift", 0.0))
        g_shift = float(shock.get("g_shift", 0.0))

        for row in inputs.itertuples(index=False):
            eps = row.EPS

            afiliados = np.full(n_sim, float(row.Afiliados_0), dtype=float)
            cash = np.full(n_sim, float(row.Cash_0), dtype=float)
            equity = np.full(n_sim, float(row.Equity_0), dtype=float)

            k_e = float(row.k_e)
            capmin = float(row.CapMinReq)
            inv_ratio0 = float(inv_ratio_map.get(eps, 0.0))

            g_m, g_s = float(row.g_mean), float(row.g_std)
            lr_m, lr_s = float(row.LR_mean), float(row.LR_std)
            ar_m, ar_s = float(row.AR_mean), float(row.AR_std)
            oer_m, oer_s = float(row.OER_mean), float(row.OER_std)
            rr_m, rr_s = float(row.rho_res_mean), float(row.rho_res_std)
            ap_m = _safe_float(row.ap_to_rev_mean, default=global_ap_mean)
            ap_s = _safe_float(row.ap_to_rev_std, default=global_ap_std)
            if ap_s <= 0:
                ap_s = global_ap_std
            mr_cfg = mean_reversion_map.get(eps, {})
            mu_long_g = _safe_float(mr_cfg.get("mu_long_g"), g_m)
            mu_recent_g = _safe_float(mr_cfg.get("mu_recent_g"), g_m)
            sigma_long_g = _safe_float(mr_cfg.get("sigma_long_g"), g_s)
            lambda_g = float(np.clip(_safe_float(mr_cfg.get("lambda_g"), 0.05), 0.05, 0.25))
            mu_long_lr = _safe_float(mr_cfg.get("mu_long_lr"), lr_m)
            mu_recent_lr = _safe_float(mr_cfg.get("mu_recent_lr"), lr_m)
            sigma_long_lr = _safe_float(mr_cfg.get("sigma_long_lr"), lr_s)
            lambda_lr = float(np.clip(_safe_float(mr_cfg.get("lambda_lr"), 0.05), 0.05, 0.25))

            g_sigma = sigma_long_g if sigma_long_g > 0 else _safe_float(g_s, 0.0)
            lr_sigma = sigma_long_lr if sigma_long_lr > 0 else _safe_float(lr_s, 0.0)
            mean_g = mu_recent_g + g_shift
            mean_lr = mu_recent_lr + lr_shift
            target_g = mu_long_g + g_shift
            target_lr = mu_long_lr + lr_shift

            avg_cash_counts = {thr: np.zeros(n_sim, dtype=float) for thr in thresholds}
            avg_cm_count = np.zeros(n_sim, dtype=float)
            avg_pa_count = np.zeros(n_sim, dtype=float)
            avg_ri_count = np.zeros(n_sim, dtype=float)
            avg_paydays_out_count = np.zeros(n_sim, dtype=float)

            for year in years_sim:
                upc_mix_y = upc_dict.get((eps, year), upc_dict.get((eps, year - 1), 0.0))

                mean_g = mean_g + lambda_g * (target_g - mean_g)
                mean_lr = mean_lr + lambda_lr * (target_lr - mean_lr)

                g = _rnorm_trunc(rng, mean_g, g_sigma, -0.50, 0.50, n_sim)
                lr = _rnorm_trunc(rng, mean_lr, lr_sigma, 0.01, 2.00, n_sim)
                ar = _rnorm_trunc(rng, ar_m, ar_s, 0.00, 1.00, n_sim)
                oer = _rnorm_trunc(rng, oer_m, oer_s, 0.00, 1.00, n_sim)
                rho_res = _rnorm_trunc(rng, rr_m, rr_s, 0.00, 2.00, n_sim)
                ap_to_rev = _rnorm_trunc(rng, ap_m, ap_s, 0.00, 3.00, n_sim)

                afiliados = afiliados * (1.0 + g)
                ingresos = afiliados * upc_mix_y * k_e
                opex_cash = ingresos * (lr + ar + oer)
                margin = ingresos - opex_cash
                cash = np.maximum(0.0, cash + alpha * margin)
                equity = equity + beta * margin

                reservas = ingresos * rho_res
                inversiones = ingresos * inv_ratio0
                cxp_comercial = ingresos * ap_to_rev

                cashdays = 365.0 * _safe_div(cash, opex_cash)
                paydays = 365.0 * _safe_div(cxp_comercial, opex_cash)
                cm_ratio = _safe_div(equity, np.full(n_sim, capmin, dtype=float))
                pa_req = 0.08 * ingresos * lr
                pa_ratio = _safe_div(equity, pa_req)
                ri_ratio = _safe_div((cash + inversiones), reservas)

                for thr in thresholds:
                    cash_breach = cashdays < thr
                    avg_cash_counts[thr] += cash_breach.astype(float)

                avg_cm_count += (cm_ratio < 1.0).astype(float)
                avg_pa_count += (pa_ratio < 1.0).astype(float)
                avg_ri_count += (ri_ratio < 1.0).astype(float)
                avg_paydays_out_count += (
                    (paydays < paydays_low) | (paydays > paydays_high)
                ).astype(float)

            n_years = max(len(years_sim), 1)
            row_out: dict[str, Any] = {
                "EPS": eps,
                "Escenario": scenario_name,
                "anchor_year": anchor_year,
                "alpha_fijo": alpha,
                "beta_fijo": beta,
                "P_avg_CM_ratio_lt_1": float((avg_cm_count / n_years).mean()),
                "P_avg_PA_ratio_lt_1": float((avg_pa_count / n_years).mean()),
                "P_avg_RI_ratio_lt_1": float((avg_ri_count / n_years).mean()),
                f"P_avg_PayDays_out_{int(paydays_low)}_{int(paydays_high)}": float(
                    (avg_paydays_out_count / n_years).mean()
                ),
            }
            for thr in thresholds:
                row_out[f"P_avg_CashDays_lt_{thr}"] = float((avg_cash_counts[thr] / n_years).mean())
            results.append(row_out)

    results_df = pd.DataFrame(results).sort_values(["Escenario", "EPS"]).reset_index(drop=True)
    diagnostics = {
        "anchor_year": anchor_year,
        "base_eps": base_eps,
        "upc_growth_segments": upc_growth_segments,
        "mean_reversion_params": mean_reversion_params,
        "paydays_range": (paydays_low, paydays_high),
        "scenarios": dict(scenarios),
        "eps_obj": list(eps_obj),
        "prep": prep_diag,
    }
    return results_df, diagnostics


def run_eps_montecarlo_backtesting(
    upc_df: pd.DataFrame,
    eps_eeff_df: pd.DataFrame,
    eps_edad_df: pd.DataFrame,
    eps_afiliados_hist_df: pd.DataFrame,
    eps_obj: Sequence[str] | None = None,
    n_sim: int = 3_000,
    cash_thresholds: Sequence[int] = (15,),
    paydays_range: tuple[float, float] = (20.0, 70.0),
    upc_optimism_factor: float = 1.2,
    upc_growth_start_year: int = 2019,
    upc_growth_end_year: int = 2026,
    start_anchor_year: int = 2021,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    eps_obj = list(eps_obj or EPS_OBJ_DEFAULT)
    eeff_wide_full = _build_eps_eeff_wide(eps_eeff_df=eps_eeff_df, eps_obj=eps_obj)
    if eeff_wide_full.empty:
        empty_detail = pd.DataFrame(
            columns=[
                "EPS",
                "AnchorYear",
                "TargetYear",
                "Metric",
                "PredictedProb",
                "ObservedBreach",
                "AbsError",
                "Brier",
            ]
        )
        empty_summary = pd.DataFrame(
            columns=[
                "Metric",
                "N",
                "PredictedMean",
                "ObservedRate",
                "MAE",
                "Brier",
            ]
        )
        return empty_detail, empty_summary, {
            "anchors_used": [],
            "reason": "sin_eeff_para_backtesting",
        }

    year_col = _resolve_year_column(eeff_wide_full)
    years = sorted(pd.to_numeric(eeff_wide_full[year_col], errors="coerce").dropna().astype(int).unique().tolist())
    if len(years) < 3:
        empty_detail = pd.DataFrame(
            columns=[
                "EPS",
                "AnchorYear",
                "TargetYear",
                "Metric",
                "PredictedProb",
                "ObservedBreach",
                "AbsError",
                "Brier",
            ]
        )
        empty_summary = pd.DataFrame(
            columns=[
                "Metric",
                "N",
                "PredictedMean",
                "ObservedRate",
                "MAE",
                "Brier",
            ]
        )
        return empty_detail, empty_summary, {
            "anchors_used": [],
            "reason": "historia_insuficiente",
            "years_detected": years,
        }

    anchor_min = max(int(start_anchor_year), years[0] + 1)
    anchor_max = years[-2]
    anchors = [y for y in years if anchor_min <= y <= anchor_max]
    if not anchors:
        empty_detail = pd.DataFrame(
            columns=[
                "EPS",
                "AnchorYear",
                "TargetYear",
                "Metric",
                "PredictedProb",
                "ObservedBreach",
                "AbsError",
                "Brier",
            ]
        )
        empty_summary = pd.DataFrame(
            columns=[
                "Metric",
                "N",
                "PredictedMean",
                "ObservedRate",
                "MAE",
                "Brier",
            ]
        )
        return empty_detail, empty_summary, {
            "anchors_used": [],
            "reason": "sin_ventanas_backtest",
            "anchor_min": anchor_min,
            "anchor_max": anchor_max,
        }

    actual_breaches = _compute_eps_actual_breaches(
        eps_eeff_df=eps_eeff_df,
        eps_obj=eps_obj,
        cash_thresholds=cash_thresholds,
        paydays_range=paydays_range,
    )

    low, high = paydays_range
    if low > high:
        low, high = high, low
    metric_map = {
        "P_avg_CM_ratio_lt_1": "Obs_CM_ratio_lt_1",
        "P_avg_PA_ratio_lt_1": "Obs_PA_ratio_lt_1",
        "P_avg_RI_ratio_lt_1": "Obs_RI_ratio_lt_1",
        f"P_avg_PayDays_out_{int(low)}_{int(high)}": f"Obs_PayDays_out_{int(low)}_{int(high)}",
    }
    for thr in cash_thresholds:
        metric_map[f"P_avg_CashDays_lt_{int(thr)}"] = f"Obs_CashDays_lt_{int(thr)}"

    detail_rows: List[Dict[str, Any]] = []

    for anchor_year in anchors:
        target_year = int(anchor_year) + 1
        eeff_cut = _truncate_wide_year_columns(eps_eeff_df, max_year=anchor_year)
        afiliados_cut = _truncate_wide_year_columns(eps_afiliados_hist_df, max_year=anchor_year)
        upc_cut = _truncate_wide_year_columns(upc_df, max_year=anchor_year)

        sim_results, _ = run_eps_montecarlo(
            upc_df=upc_cut,
            eps_eeff_df=eeff_cut,
            eps_edad_df=eps_edad_df,
            eps_afiliados_hist_df=afiliados_cut,
            eps_obj=eps_obj,
            n_sim=int(n_sim),
            horizon_end=target_year,
            cash_thresholds=tuple(int(x) for x in cash_thresholds),
            paydays_range=(float(low), float(high)),
            upc_optimism_factor=float(upc_optimism_factor),
            upc_growth_start_year=int(upc_growth_start_year),
            upc_growth_end_year=int(upc_growth_end_year),
            scenarios={"BASE": {"LR_shift": 0.0, "g_shift": 0.0}},
            random_seed=int(random_seed) + int(anchor_year),
        )
        sim_results = sim_results[sim_results["Escenario"] == "BASE"].copy()
        if sim_results.empty:
            continue

        actual_target = actual_breaches[actual_breaches["AÃ±o"] == target_year].copy()
        if actual_target.empty:
            continue

        merged = sim_results.merge(
            actual_target,
            on="EPS",
            how="inner",
        )
        if merged.empty:
            continue

        for prob_col, obs_col in metric_map.items():
            if prob_col not in merged.columns or obs_col not in merged.columns:
                continue
            pred = pd.to_numeric(merged[prob_col], errors="coerce")
            obs = pd.to_numeric(merged[obs_col], errors="coerce")
            valid = pred.notna() & obs.notna()
            if not valid.any():
                continue
            valid_idx = merged.index[valid]
            for idx in valid_idx:
                eps_name = str(merged.at[idx, "EPS"])
                pred_val = float(pred.at[idx])
                obs_val = float(obs.at[idx])
                abs_err = abs(pred_val - obs_val)
                brier = (pred_val - obs_val) ** 2
                detail_rows.append(
                    {
                        "EPS": eps_name,
                        "AnchorYear": int(anchor_year),
                        "TargetYear": int(target_year),
                        "Metric": prob_col,
                        "PredictedProb": pred_val,
                        "ObservedBreach": obs_val,
                        "AbsError": abs_err,
                        "Brier": brier,
                    }
                )

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        summary_df = pd.DataFrame(
            columns=[
                "Metric",
                "N",
                "PredictedMean",
                "ObservedRate",
                "MAE",
                "Brier",
            ]
        )
    else:
        summary_df = (
            detail_df.groupby("Metric", as_index=False)
            .agg(
                N=("EPS", "count"),
                PredictedMean=("PredictedProb", "mean"),
                ObservedRate=("ObservedBreach", "mean"),
                MAE=("AbsError", "mean"),
                Brier=("Brier", "mean"),
            )
            .sort_values("Metric")
            .reset_index(drop=True)
        )

    diagnostics = {
        "anchors_used": anchors,
        "n_anchors": len(anchors),
        "n_obs": int(len(detail_df)),
        "metric_map": metric_map,
        "start_anchor_year": int(start_anchor_year),
    }
    return detail_df, summary_df, diagnostics


def run_eps_seed_stability(
    upc_df: pd.DataFrame,
    eps_eeff_df: pd.DataFrame,
    eps_edad_df: pd.DataFrame,
    eps_afiliados_hist_df: pd.DataFrame,
    market_share_df: pd.DataFrame,
    reclamos_score_df: pd.DataFrame | None = None,
    cxp_score_df: pd.DataFrame | None = None,
    eps_obj: Sequence[str] | None = None,
    seeds: Sequence[int] = (7, 42, 77, 123, 2026),
    n_sim: int = 3_000,
    horizon_end: int = 2030,
    cash_thresholds: Sequence[int] = (15,),
    paydays_range: tuple[float, float] = (20.0, 70.0),
    upc_optimism_factor: float = 1.2,
    upc_growth_start_year: int = 2019,
    upc_growth_end_year: int = 2026,
    scenarios: Mapping[str, Mapping[str, float]] | None = None,
    risk_weight: float = 0.55,
    market_weight: float = 0.05,
    complaints_weight: float = 0.2,
    cxp_rev_weight: float = 0.2,
) -> Dict[str, pd.DataFrame]:
    eps_obj = list(eps_obj or EPS_OBJ_DEFAULT)
    scenarios = scenarios or {
        "BASE": {"LR_shift": 0.0, "g_shift": 0.0},
        "STRESS_LR": {"LR_shift": 0.05, "g_shift": 0.0},
        "STRESS_MIX": {"LR_shift": 0.05, "g_shift": -0.03},
    }
    probability_columns = [f"P_avg_CashDays_lt_{int(thr)}" for thr in cash_thresholds] + [
        "P_avg_CM_ratio_lt_1",
        "P_avg_PA_ratio_lt_1",
        "P_avg_RI_ratio_lt_1",
        f"P_avg_PayDays_out_{int(min(paydays_range))}_{int(max(paydays_range))}",
    ]
    probability_columns = [c for c in probability_columns if c in PROBABILITY_COLUMNS or c.startswith("P_avg_")]

    per_seed_global: List[pd.DataFrame] = []
    per_seed_escenario: List[pd.DataFrame] = []

    for seed in seeds:
        results_df, _ = run_eps_montecarlo(
            upc_df=upc_df,
            eps_eeff_df=eps_eeff_df,
            eps_edad_df=eps_edad_df,
            eps_afiliados_hist_df=eps_afiliados_hist_df,
            eps_obj=eps_obj,
            n_sim=int(n_sim),
            horizon_end=int(horizon_end),
            cash_thresholds=tuple(int(x) for x in cash_thresholds),
            paydays_range=paydays_range,
            upc_optimism_factor=float(upc_optimism_factor),
            upc_growth_start_year=int(upc_growth_start_year),
            upc_growth_end_year=int(upc_growth_end_year),
            scenarios=scenarios,
            random_seed=int(seed),
        )
        results_imputed = impute_missing_probabilities(
            results_df=results_df,
            eps_obj=eps_obj,
            scenarios=list(scenarios.keys()),
            probability_columns=PROBABILITY_COLUMNS,
        )
        results_scored = score_risk_percentiles(
            df=results_imputed,
            probability_columns=PROBABILITY_COLUMNS,
        )
        _, ranking_escenario, ranking_global = build_composite_ranking(
            scored_df=results_scored,
            market_share_df=market_share_df,
            reclamos_score_df=reclamos_score_df,
            cxp_score_df=cxp_score_df,
            risk_weight=float(risk_weight),
            market_weight=float(market_weight),
            complaints_weight=float(complaints_weight),
            cxp_rev_weight=float(cxp_rev_weight),
        )

        g = ranking_global[["EPS", "Score_Final", "Ranking_Global_Final"]].copy()
        g["Seed"] = int(seed)
        per_seed_global.append(g)

        s = ranking_escenario[
            ["Escenario", "EPS", "Score_Final", "Ranking_Escenario_Final"]
        ].copy()
        s["Seed"] = int(seed)
        per_seed_escenario.append(s)

    global_df = pd.concat(per_seed_global, ignore_index=True) if per_seed_global else pd.DataFrame()
    escenario_df = pd.concat(per_seed_escenario, ignore_index=True) if per_seed_escenario else pd.DataFrame()

    pair_rows_global: List[Dict[str, Any]] = []
    pair_rows_esc: List[Dict[str, Any]] = []

    seed_values = sorted(pd.to_numeric(pd.Series(list(seeds)), errors="coerce").dropna().astype(int).unique().tolist())
    for seed_a, seed_b in combinations(seed_values, 2):
        ga = global_df[global_df["Seed"] == seed_a][["EPS", "Score_Final", "Ranking_Global_Final"]].rename(
            columns={
                "Score_Final": "Score_A",
                "Ranking_Global_Final": "Rank_A",
            }
        )
        gb = global_df[global_df["Seed"] == seed_b][["EPS", "Score_Final", "Ranking_Global_Final"]].rename(
            columns={
                "Score_Final": "Score_B",
                "Ranking_Global_Final": "Rank_B",
            }
        )
        gm = ga.merge(gb, on="EPS", how="inner")
        if gm.empty:
            continue
        pair_rows_global.append(
            {
                "Seed_A": int(seed_a),
                "Seed_B": int(seed_b),
                "N_EPS": int(len(gm)),
                "Spearman_Score_Final": float(gm["Score_A"].corr(gm["Score_B"], method="spearman")),
                "Spearman_Ranking_Global": float(gm["Rank_A"].corr(gm["Rank_B"], method="spearman")),
            }
        )

        for scen in sorted(escenario_df["Escenario"].dropna().unique().tolist()):
            sa = escenario_df[
                (escenario_df["Seed"] == seed_a) & (escenario_df["Escenario"] == scen)
            ][["EPS", "Score_Final", "Ranking_Escenario_Final"]].rename(
                columns={
                    "Score_Final": "Score_A",
                    "Ranking_Escenario_Final": "Rank_A",
                }
            )
            sb = escenario_df[
                (escenario_df["Seed"] == seed_b) & (escenario_df["Escenario"] == scen)
            ][["EPS", "Score_Final", "Ranking_Escenario_Final"]].rename(
                columns={
                    "Score_Final": "Score_B",
                    "Ranking_Escenario_Final": "Rank_B",
                }
            )
            sm = sa.merge(sb, on="EPS", how="inner")
            if sm.empty:
                continue
            pair_rows_esc.append(
                {
                    "Escenario": scen,
                    "Seed_A": int(seed_a),
                    "Seed_B": int(seed_b),
                    "N_EPS": int(len(sm)),
                    "Spearman_Score_Final": float(sm["Score_A"].corr(sm["Score_B"], method="spearman")),
                    "Spearman_Ranking_Escenario": float(sm["Rank_A"].corr(sm["Rank_B"], method="spearman")),
                }
            )

    if pair_rows_global:
        pairwise_global_df = (
            pd.DataFrame(pair_rows_global)
            .sort_values(["Seed_A", "Seed_B"])
            .reset_index(drop=True)
        )
    else:
        pairwise_global_df = pd.DataFrame(
            columns=[
                "Seed_A",
                "Seed_B",
                "N_EPS",
                "Spearman_Score_Final",
                "Spearman_Ranking_Global",
            ]
        )
    if pair_rows_esc:
        pairwise_esc_df = (
            pd.DataFrame(pair_rows_esc)
            .sort_values(["Escenario", "Seed_A", "Seed_B"])
            .reset_index(drop=True)
        )
    else:
        pairwise_esc_df = pd.DataFrame(
            columns=[
                "Escenario",
                "Seed_A",
                "Seed_B",
                "N_EPS",
                "Spearman_Score_Final",
                "Spearman_Ranking_Escenario",
            ]
        )

    if pairwise_global_df.empty:
        global_summary_df = pd.DataFrame(
            columns=[
                "Metric",
                "Mean",
                "Min",
                "Max",
            ]
        )
    else:
        global_summary_df = pd.DataFrame(
            [
                {
                    "Metric": "Spearman_Score_Final",
                    "Mean": float(pairwise_global_df["Spearman_Score_Final"].mean()),
                    "Min": float(pairwise_global_df["Spearman_Score_Final"].min()),
                    "Max": float(pairwise_global_df["Spearman_Score_Final"].max()),
                },
                {
                    "Metric": "Spearman_Ranking_Global",
                    "Mean": float(pairwise_global_df["Spearman_Ranking_Global"].mean()),
                    "Min": float(pairwise_global_df["Spearman_Ranking_Global"].min()),
                    "Max": float(pairwise_global_df["Spearman_Ranking_Global"].max()),
                },
            ]
        )

    if pairwise_esc_df.empty:
        escenario_summary_df = pd.DataFrame(
            columns=["Escenario", "Metric", "Mean", "Min", "Max"]
        )
    else:
        rows: List[Dict[str, Any]] = []
        for scen, gdf in pairwise_esc_df.groupby("Escenario", dropna=False):
            rows.append(
                {
                    "Escenario": scen,
                    "Metric": "Spearman_Score_Final",
                    "Mean": float(gdf["Spearman_Score_Final"].mean()),
                    "Min": float(gdf["Spearman_Score_Final"].min()),
                    "Max": float(gdf["Spearman_Score_Final"].max()),
                }
            )
            rows.append(
                {
                    "Escenario": scen,
                    "Metric": "Spearman_Ranking_Escenario",
                    "Mean": float(gdf["Spearman_Ranking_Escenario"].mean()),
                    "Min": float(gdf["Spearman_Ranking_Escenario"].min()),
                    "Max": float(gdf["Spearman_Ranking_Escenario"].max()),
                }
            )
        escenario_summary_df = pd.DataFrame(rows).sort_values(["Escenario", "Metric"]).reset_index(drop=True)

    return {
        "per_seed_global": global_df,
        "per_seed_escenario": escenario_df,
        "pairwise_global": pairwise_global_df,
        "pairwise_escenario": pairwise_esc_df,
        "summary_global": global_summary_df,
        "summary_escenario": escenario_summary_df,
    }


def compute_market_share_valle(
    eps_afiliados_df: pd.DataFrame,
    eps_obj: Sequence[str] | None = None,
) -> pd.DataFrame:
    eps_obj = list(eps_obj or EPS_OBJ_DEFAULT)
    df = eps_afiliados_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    dep_col = _resolve_column(df, ["Departamento"])
    eps_col = _resolve_column(df, ["EPS"])
    total_col = _resolve_column(df, ["TOTAL AFILIADOS"])
    if dep_col is None or eps_col is None or total_col is None:
        out = pd.DataFrame({"EPS": eps_obj})
        out["Afiliados_Valle"] = 0.0
        out["MarketShare_Valle"] = 0.0
        return out

    work = df[[dep_col, eps_col, total_col]].copy()
    work = work[work[dep_col].astype(str).str.contains("valle del cauca", case=False, na=False)]
    work["EPS"] = _canonicalize_eps(work[eps_col], eps_obj)
    work["Afiliados"] = pd.to_numeric(work[total_col], errors="coerce").fillna(0.0)

    total_valle = float(work["Afiliados"].sum())
    grouped = work.groupby("EPS", as_index=False)["Afiliados"].sum()
    grouped = grouped.rename(columns={"Afiliados": "Afiliados_Valle"})
    grouped["MarketShare_Valle"] = grouped["Afiliados_Valle"] / total_valle if total_valle > 0 else 0.0

    universe = pd.DataFrame({"EPS": eps_obj})
    out = universe.merge(grouped, on="EPS", how="left")
    out["Afiliados_Valle"] = out["Afiliados_Valle"].fillna(0.0)
    out["MarketShare_Valle"] = out["MarketShare_Valle"].fillna(0.0)
    out = out.sort_values("MarketShare_Valle", ascending=False).reset_index(drop=True)
    return out


def compute_reclamos_score(
    reclamos_df: pd.DataFrame,
    eps_obj: Sequence[str] | None = None,
) -> pd.DataFrame:
    eps_obj = list(eps_obj or EPS_OBJ_DEFAULT)
    universe = pd.DataFrame({"EPS": eps_obj})
    if reclamos_df.empty:
        out = universe.copy()
        out["Tasa_Reclamos_10k"] = np.nan
        out["Reclamos"] = np.nan
        out["Score_Reclamos"] = np.nan
        out["reclamos_imputado"] = True
        out["motivo_imputacion_reclamos"] = "sin_datos_reclamos"
        return out

    df = reclamos_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    eps_col = _resolve_column(df, ["EPS"])
    tasa_col = _resolve_column(
        df,
        [
            "TASA X CADA 10.000 AFILIADOS",
            "TASA X CADA 10000 AFILIADOS",
            "TASA CADA 10.000 AFILIADOS",
            "TASA CADA 10000 AFILIADOS",
            "TASA",
        ],
    )
    reclamos_col = _resolve_column(df, ["RECLAMOS"])
    if eps_col is None or tasa_col is None:
        out = universe.copy()
        out["Tasa_Reclamos_10k"] = np.nan
        out["Reclamos"] = np.nan
        out["Score_Reclamos"] = np.nan
        out["reclamos_imputado"] = True
        out["motivo_imputacion_reclamos"] = "columnas_reclamos_incompletas"
        return out

    work_cols = [eps_col, tasa_col] + ([reclamos_col] if reclamos_col else [])
    work = df[work_cols].copy()
    work["EPS"] = work[eps_col].map(lambda x: _canonicalize_eps_value(x, eps_obj))
    work["Tasa_Reclamos_10k"] = pd.to_numeric(work[tasa_col], errors="coerce")
    if reclamos_col:
        work["Reclamos"] = pd.to_numeric(work[reclamos_col], errors="coerce")
    else:
        work["Reclamos"] = np.nan

    work = work[work["EPS"].isin(eps_obj)].copy()

    grouped = (
        work.groupby("EPS", as_index=False)
        .agg(
            Tasa_Reclamos_10k=("Tasa_Reclamos_10k", "mean"),
            Reclamos=("Reclamos", "sum"),
        )
        .reset_index(drop=True)
    )

    out = universe.merge(grouped, on="EPS", how="left")
    median_tasa = pd.to_numeric(out["Tasa_Reclamos_10k"], errors="coerce").median()
    if pd.isna(median_tasa):
        median_tasa = 0.0

    out["reclamos_imputado"] = out["Tasa_Reclamos_10k"].isna()
    out["motivo_imputacion_reclamos"] = np.where(
        out["reclamos_imputado"],
        "sin_tasa_reclamos",
        "",
    )
    out["Tasa_Reclamos_10k"] = out["Tasa_Reclamos_10k"].fillna(float(median_tasa))
    out["Score_Reclamos"] = _percentile_score_low_is_better(out["Tasa_Reclamos_10k"])
    return out.sort_values("EPS").reset_index(drop=True)


def compute_cxp_revenue_score(
    eps_eeff_df: pd.DataFrame,
    eps_obj: Sequence[str] | None = None,
) -> pd.DataFrame:
    eps_obj = list(eps_obj or EPS_OBJ_DEFAULT)
    universe = pd.DataFrame({"EPS": eps_obj})
    if eps_eeff_df.empty:
        out = universe.copy()
        out["CxP_Comercial"] = np.nan
        out["Ingresos"] = np.nan
        out["CxP_over_REV"] = np.nan
        out["Score_CxP_REV"] = np.nan
        out["cxp_rev_imputado"] = True
        out["motivo_imputacion_cxp_rev"] = "sin_datos_eeff"
        return out

    eeff_wide = _build_eps_eeff_wide(eps_eeff_df=eps_eeff_df, eps_obj=eps_obj)
    if eeff_wide.empty:
        out = universe.copy()
        out["CxP_Comercial"] = np.nan
        out["Ingresos"] = np.nan
        out["CxP_over_REV"] = np.nan
        out["Score_CxP_REV"] = np.nan
        out["cxp_rev_imputado"] = True
        out["motivo_imputacion_cxp_rev"] = "sin_eps_obj_en_eeff"
        return out

    year_col = _resolve_year_column(eeff_wide)
    anchor_year = int(pd.to_numeric(eeff_wide[year_col], errors="coerce").max())
    anchor = eeff_wide[pd.to_numeric(eeff_wide[year_col], errors="coerce") == anchor_year].copy()
    anchor = anchor.sort_values(["EPS", year_col]).drop_duplicates(subset=["EPS"], keep="last")

    ingresos = _numeric_series(anchor, ["TotalIngresoOperativo", "Ingresosnetosporventas"])
    cxp_comercial = _commercial_payables_series(anchor)
    ratio = _safe_div(cxp_comercial.to_numpy(dtype=float), ingresos.to_numpy(dtype=float))

    scored = pd.DataFrame(
        {
            "EPS": anchor["EPS"].astype(str),
            "CxP_Comercial": pd.to_numeric(cxp_comercial, errors="coerce"),
            "Ingresos": pd.to_numeric(ingresos, errors="coerce"),
            "CxP_over_REV": pd.to_numeric(ratio, errors="coerce"),
        }
    )
    out = universe.merge(scored, on="EPS", how="left")
    median_ratio = pd.to_numeric(out["CxP_over_REV"], errors="coerce").median()
    if pd.isna(median_ratio):
        median_ratio = 0.0

    out["cxp_rev_imputado"] = out["CxP_over_REV"].isna()
    out["motivo_imputacion_cxp_rev"] = np.where(
        out["cxp_rev_imputado"],
        "sin_cuentas_cxp_o_ingresos",
        "",
    )
    out["CxP_over_REV"] = out["CxP_over_REV"].fillna(float(median_ratio))
    out["Score_CxP_REV"] = _percentile_score_low_is_better(out["CxP_over_REV"])
    return out.sort_values("EPS").reset_index(drop=True)


def impute_missing_probabilities(
    results_df: pd.DataFrame,
    eps_obj: Sequence[str] | None = None,
    scenarios: Sequence[str] | None = None,
    probability_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    eps_obj = list(eps_obj or EPS_OBJ_DEFAULT)
    probability_columns = list(probability_columns or [c for c in results_df.columns if c.startswith("P_avg_")])
    scenario_values = list(scenarios or sorted(results_df["Escenario"].dropna().unique().tolist()))

    grid = pd.MultiIndex.from_product(
        [eps_obj, scenario_values],
        names=["EPS", "Escenario"],
    ).to_frame(index=False)
    merged = grid.merge(results_df, on=["EPS", "Escenario"], how="left")

    for col in probability_columns:
        if col not in merged.columns:
            merged[col] = np.nan

    means_by_scenario = merged.groupby("Escenario")[probability_columns].mean()
    global_means = merged[probability_columns].mean()

    missing_mask = merged[probability_columns].isna().any(axis=1)
    for idx in merged.index[missing_mask]:
        scenario = merged.at[idx, "Escenario"]
        fill_vals = means_by_scenario.loc[scenario]
        for col in probability_columns:
            val = fill_vals[col]
            if pd.isna(val):
                val = global_means[col]
            if pd.isna(val):
                val = 0.0
            merged.at[idx, col] = float(val)

    merged["prob_imputada"] = missing_mask
    merged["motivo_imputacion"] = np.where(
        merged["prob_imputada"],
        "sin_eeff_historica_suficiente",
        "",
    )
    return merged.sort_values(["Escenario", "EPS"]).reset_index(drop=True)


def score_risk_percentiles(
    df: pd.DataFrame,
    probability_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    probability_columns = list(probability_columns or [c for c in df.columns if c.startswith("P_avg_")])
    out = df.copy()
    for col in probability_columns:
        score_col = f"Score_{col}"
        out[score_col] = out.groupby("Escenario")[col].transform(_percentile_score_low_is_better)
    score_cols = [f"Score_{col}" for col in probability_columns]
    out["Score_Riesgo"] = out[score_cols].mean(axis=1, skipna=True)
    out["Ranking_Escenario_Riesgo"] = (
        out.groupby("Escenario")["Score_Riesgo"].rank(method="dense", ascending=False).astype(int)
    )
    return out


def build_composite_ranking(
    scored_df: pd.DataFrame,
    market_share_df: pd.DataFrame,
    reclamos_score_df: pd.DataFrame | None = None,
    cxp_score_df: pd.DataFrame | None = None,
    risk_weight: float = 0.55,
    market_weight: float = 0.05,
    complaints_weight: float = 0.2,
    cxp_rev_weight: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_weight = (
        float(risk_weight)
        + float(market_weight)
        + float(complaints_weight)
        + float(cxp_rev_weight)
    )
    if not np.isclose(total_weight, 1.0, atol=1e-9):
        raise ValueError(
            "La suma de pesos debe ser 1.0 "
            f"(risk_weight + market_weight + complaints_weight + cxp_rev_weight = {total_weight})."
        )

    out = scored_df.merge(
        market_share_df[["EPS", "MarketShare_Valle", "Afiliados_Valle"]],
        on="EPS",
        how="left",
    )
    out["MarketShare_Valle"] = out["MarketShare_Valle"].fillna(0.0)
    out["Afiliados_Valle"] = out["Afiliados_Valle"].fillna(0.0)

    market_scores = (
        out[["EPS", "MarketShare_Valle"]]
        .drop_duplicates(subset=["EPS"])
        .assign(Score_Mercado=lambda d: _percentile_score_high_is_better(d["MarketShare_Valle"]))
    )
    out = out.merge(market_scores[["EPS", "Score_Mercado"]], on="EPS", how="left")

    if reclamos_score_df is None:
        reclamos_cols = [
            "EPS",
            "Tasa_Reclamos_10k",
            "Reclamos",
            "Score_Reclamos",
            "reclamos_imputado",
            "motivo_imputacion_reclamos",
        ]
        reclamos_scores = pd.DataFrame({"EPS": out["EPS"].drop_duplicates()})
        reclamos_scores["Tasa_Reclamos_10k"] = np.nan
        reclamos_scores["Reclamos"] = np.nan
        reclamos_scores["Score_Reclamos"] = 0.0
        reclamos_scores["reclamos_imputado"] = True
        reclamos_scores["motivo_imputacion_reclamos"] = "reclamos_score_df_no_provisto"
        reclamos_scores = reclamos_scores[reclamos_cols]
    else:
        reclamos_scores = reclamos_score_df[
            [
                "EPS",
                "Tasa_Reclamos_10k",
                "Reclamos",
                "Score_Reclamos",
                "reclamos_imputado",
                "motivo_imputacion_reclamos",
            ]
        ].copy()
        reclamos_scores["Score_Reclamos"] = pd.to_numeric(
            reclamos_scores["Score_Reclamos"], errors="coerce"
        ).fillna(0.0)
        reclamos_scores["reclamos_imputado"] = reclamos_scores["reclamos_imputado"].fillna(True)
        reclamos_scores["motivo_imputacion_reclamos"] = (
            reclamos_scores["motivo_imputacion_reclamos"].fillna("")
        )

    out = out.merge(reclamos_scores, on="EPS", how="left")
    out["Score_Reclamos"] = pd.to_numeric(out["Score_Reclamos"], errors="coerce").fillna(0.0)
    out["reclamos_imputado"] = out["reclamos_imputado"].fillna(True)
    out["motivo_imputacion_reclamos"] = out["motivo_imputacion_reclamos"].fillna("")

    if cxp_score_df is None:
        cxp_scores = pd.DataFrame({"EPS": out["EPS"].drop_duplicates()})
        cxp_scores["CxP_Comercial"] = np.nan
        cxp_scores["Ingresos"] = np.nan
        cxp_scores["CxP_over_REV"] = np.nan
        cxp_scores["Score_CxP_REV"] = 0.0
        cxp_scores["cxp_rev_imputado"] = True
        cxp_scores["motivo_imputacion_cxp_rev"] = "cxp_score_df_no_provisto"
    else:
        cxp_scores = cxp_score_df[
            [
                "EPS",
                "CxP_Comercial",
                "Ingresos",
                "CxP_over_REV",
                "Score_CxP_REV",
                "cxp_rev_imputado",
                "motivo_imputacion_cxp_rev",
            ]
        ].copy()
        cxp_scores["Score_CxP_REV"] = pd.to_numeric(
            cxp_scores["Score_CxP_REV"], errors="coerce"
        ).fillna(0.0)
        cxp_scores["cxp_rev_imputado"] = cxp_scores["cxp_rev_imputado"].fillna(True)
        cxp_scores["motivo_imputacion_cxp_rev"] = cxp_scores[
            "motivo_imputacion_cxp_rev"
        ].fillna("")

    out = out.merge(cxp_scores, on="EPS", how="left")
    out["Score_CxP_REV"] = pd.to_numeric(out["Score_CxP_REV"], errors="coerce").fillna(0.0)
    out["cxp_rev_imputado"] = out["cxp_rev_imputado"].fillna(True)
    out["motivo_imputacion_cxp_rev"] = out["motivo_imputacion_cxp_rev"].fillna("")

    out["Score_Final"] = (
        risk_weight * out["Score_Riesgo"]
        + market_weight * out["Score_Mercado"]
        + complaints_weight * out["Score_Reclamos"]
        + cxp_rev_weight * out["Score_CxP_REV"]
    )
    out["Ranking_Escenario_Final"] = (
        out.groupby("Escenario")["Score_Final"].rank(method="dense", ascending=False).astype(int)
    )
    ranking_escenario = out.sort_values(
        ["Escenario", "Ranking_Escenario_Final", "EPS"]
    ).reset_index(drop=True)

    ranking_global = (
        out.groupby("EPS", as_index=False)
        .agg(
            Score_Final=("Score_Final", "mean"),
            Score_Riesgo=("Score_Riesgo", "mean"),
            Score_Mercado=("Score_Mercado", "mean"),
            Score_Reclamos=("Score_Reclamos", "mean"),
            Score_CxP_REV=("Score_CxP_REV", "mean"),
            MarketShare_Valle=("MarketShare_Valle", "mean"),
            Afiliados_Valle=("Afiliados_Valle", "mean"),
            Tasa_Reclamos_10k=("Tasa_Reclamos_10k", "mean"),
            Reclamos=("Reclamos", "mean"),
            CxP_over_REV=("CxP_over_REV", "mean"),
            CxP_Comercial=("CxP_Comercial", "mean"),
            prob_imputada=("prob_imputada", "max"),
            reclamos_imputado=("reclamos_imputado", "max"),
            cxp_rev_imputado=("cxp_rev_imputado", "max"),
        )
        .sort_values("Score_Final", ascending=False)
        .reset_index(drop=True)
    )
    ranking_global["Ranking_Global_Final"] = (
        ranking_global["Score_Final"].rank(method="dense", ascending=False).astype(int)
    )
    ranking_global = ranking_global.sort_values(["Ranking_Global_Final", "EPS"]).reset_index(drop=True)
    return out, ranking_escenario, ranking_global


def build_income_statement_view(
    base_eps: pd.DataFrame,
    eps_name: str,
) -> pd.DataFrame:
    df = base_eps.copy()
    work = df[df["EPS"] == eps_name].copy()
    if work.empty:
        return pd.DataFrame()

    year_col = _resolve_year_column(work)
    work = work.sort_values(year_col).reset_index(drop=True)

    ingresos = _numeric_series(work, ["TotalIngresoOperativo", "Ingresosnetosporventas"])
    costo = _numeric_series(work, ["Otroscostospornaturaleza"], default=0.0).fillna(0.0)
    gadmin = _numeric_series(work, ["Gastosadministrativos"], default=0.0).fillna(0.0)
    g_personal = _numeric_series(
        work,
        ["Gastosporbeneficiosdelosempleado", "Gastosporbeneficiosdelosempleados"],
        default=0.0,
    ).fillna(0.0)
    g_transporte = _numeric_series(work, ["Costosdetransporte"], default=0.0).fillna(0.0)
    g_impuestos = _numeric_series(work, ["Impuestoycontribuciones"], default=0.0).fillna(0.0)
    g_otros = _numeric_series(work, ["Otrosgastos"], default=0.0).fillna(0.0)
    opex_cash = costo + gadmin + g_personal + g_transporte + g_impuestos + g_otros

    ebitda = _numeric_series(work, ["EBITDA"])
    ebit = _numeric_series(work, ["Gananciaoperativa(EBIT)"])
    utilidad = _numeric_series(
        work,
        [
            "Ganancia(P??rdida)Neta",
            "Gananciasdespu??sdeimpuestos",
            "GananciaoP??rdidadelPeriodo",
        ],
    )

    out = pd.DataFrame(
        {
            "A??o": work[year_col].astype(int),
            "Ingresos": ingresos.astype(float),
            "OPEX_caja": opex_cash.astype(float),
            "EBITDA": ebitda.astype(float),
            "EBIT": ebit.astype(float),
            "Utilidad_Neta": utilidad.astype(float),
        }
    )
    out["Margen_EBITDA"] = _safe_div(out["EBITDA"].to_numpy(dtype=float), out["Ingresos"].to_numpy(dtype=float))
    out["Margen_EBIT"] = _safe_div(out["EBIT"].to_numpy(dtype=float), out["Ingresos"].to_numpy(dtype=float))
    out["Margen_Neto"] = _safe_div(
        out["Utilidad_Neta"].to_numpy(dtype=float),
        out["Ingresos"].to_numpy(dtype=float),
    )
    return out


def build_eps_historical_compliance(
    base_eps: pd.DataFrame,
    eps_name: str,
) -> pd.DataFrame:
    df = base_eps.copy()
    work = df[df["EPS"] == eps_name].copy()
    if work.empty:
        return pd.DataFrame()

    year_col = _resolve_year_column(work)
    work = work.sort_values(year_col).reset_index(drop=True)
    ingresos = _numeric_series(work, ["TotalIngresoOperativo", "Ingresosnetosporventas"])
    costo = _numeric_series(work, ["Otroscostospornaturaleza"])
    lr = _safe_div(costo.to_numpy(dtype=float), ingresos.to_numpy(dtype=float))
    equity = _numeric_series(work, ["Totaldepatrimonio"])
    cash = _numeric_series(work, ["EfectivooEquivalentes"], default=0.0).fillna(0.0)
    inv = _numeric_series(work, ["Activosfinancierosdecortoplazo"], default=0.0).fillna(0.0)
    reservas = _numeric_series(work, ["Provisionesparaotrospasivosygastos"])

    capmin_map = _build_capmin_map(df["EPS"].dropna().astype(str).unique().tolist())
    capmin_req = float(capmin_map.get(eps_name, 19500.0))

    cm_ratio = _safe_div(equity.to_numpy(dtype=float), np.full(len(work), capmin_req, dtype=float))
    pa_req = 0.08 * ingresos.to_numpy(dtype=float) * lr
    pa_ratio = _safe_div(equity.to_numpy(dtype=float), pa_req)
    ri_ratio = _safe_div((cash + inv).to_numpy(dtype=float), reservas.to_numpy(dtype=float))

    out = pd.DataFrame(
        {
            "A??o": work[year_col].astype(int),
            "CM_ratio": cm_ratio,
            "PA_ratio": pa_ratio,
            "RI_ratio": ri_ratio,
        }
    )
    out["Cumple_CM"] = pd.Series(out["CM_ratio"] >= 1.0, index=out.index).fillna(False)
    out["Cumple_PA"] = pd.Series(out["PA_ratio"] >= 1.0, index=out.index).fillna(False)
    out["Cumple_RI"] = pd.Series(out["RI_ratio"] >= 1.0, index=out.index).fillna(False)
    out["Cumple_3_de_3"] = out["Cumple_CM"] & out["Cumple_PA"] & out["Cumple_RI"]
    return out
