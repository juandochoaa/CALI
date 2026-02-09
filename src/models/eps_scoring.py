from __future__ import annotations

from typing import Dict, Iterable, List, Tuple
import unicodedata

import numpy as np
import pandas as pd


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = " ".join(normalized.lower().split())
    return normalized


BLOCK_DEFS: List[Tuple[str, List[str], bool, bool]] = [
    ("REV", ["Ingresos netos por ventas", "Total Ingreso Operativo"], False, False),
    ("AC", ["Activos Corrientes"], False, False),
    ("PC", ["Pasivos Corrientes"], False, False),
    ("CASH", ["Efectivo o Equivalentes"], False, False),
    ("CxC", ["Cuentas comerciales por cobrar", "Comerciales y otras cuentas a cobrar"], False, False),
    ("INV", ["Inventarios", "Otros inventarios"], False, False),
    ("CxP", ["Cuentas Comerciales por pagar", "Comerciales y otras cuentas a pagar"], False, False),
    ("STD", ["Creditos y prestamos corrientes"], False, False),
    ("TL", ["Pasivos Totales"], False, False),
    ("EQ", ["Total de patrimonio"], False, False),
    ("TA", ["Activos Totales"], False, False),
    ("EBIT", ["Ganancia operativa (EBIT)"], False, False),
    ("EBITDA", ["EBITDA"], False, False),
    ("INT", ["Gastos por intereses", "Gastos financieros"], False, False),
    (
        "OPEX_cash",
        [
            "Gastos administrativos",
            "Gastos por beneficios de los empleado",
            "Gastos por beneficios de los empleados",
            "Costos de transporte",
            "Impuesto y contribuciones",
            "Otros costos por naturaleza",
        ],
        True,
        True,
    ),
    (
        "PAGOS_ANT",
        [
            "Pagos anticipados, ingresos devengados y otros activos circulantes diferidos",
            "Pagos anticipados",
        ],
        False,
        False,
    ),
    ("DEPR_AMORT", ["Gastos de depreciacion, amortizacion y deterioro"], False, True),
    (
        "NET_INCOME",
        ["Ganancia (Perdida) Neta", "Ganancias despues de impuestos", "Ganancia o Perdida del Periodo"],
        False,
        False,
    ),
]


RATIO_SPECS: Dict[str, Dict[str, str]] = {
    "current_ratio": {"category": "liquidity", "direction": "higher"},
    "cash_ratio": {"category": "liquidity", "direction": "higher"},
    "wc_to_rev": {"category": "liquidity", "direction": "higher"},
    "days_cash": {"category": "liquidity", "direction": "higher"},
    "current_assets_ratio": {"category": "liquidity", "direction": "higher"},
    "debt_to_assets": {"category": "solvency", "direction": "lower"},
    "equity_ratio": {"category": "solvency", "direction": "higher"},
    "assets_to_liabilities": {"category": "solvency", "direction": "higher"},
    "current_liab_share": {"category": "solvency", "direction": "lower"},
    "net_debt_to_ebitda": {"category": "solvency", "direction": "lower"},
    "gross_margin": {"category": "profitability", "direction": "higher"},
    "ebitda_margin": {"category": "profitability", "direction": "higher"},
    "ebit_margin": {"category": "profitability", "direction": "higher"},
    "net_margin": {"category": "profitability", "direction": "higher"},
    "roa": {"category": "profitability", "direction": "higher"},
    "asset_turnover": {"category": "efficiency", "direction": "higher"},
    "dso": {"category": "efficiency", "direction": "lower"},
    "dpo": {"category": "efficiency", "direction": "range"},
    "opex_cash_ratio": {"category": "efficiency", "direction": "lower"},
    "da_intensity": {"category": "efficiency", "direction": "lower"},
}


def build_blocks_long(
    df: pd.DataFrame,
    year_cols: Iterable[object],
    entity_col: str = "EPS_clean",
    account_col: str = "CUENTA",
) -> pd.DataFrame:
    frame = df[[entity_col, account_col] + list(year_cols)].copy()
    frame[account_col] = frame[account_col].astype(str)
    frame["account_norm"] = frame[account_col].map(_normalize_text)

    long_df = frame.melt(
        id_vars=[entity_col, "account_norm"],
        value_vars=list(year_cols),
        var_name="year",
        value_name="value",
    )
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df["year"] = long_df["year"].astype(str).str.strip().astype(int)

    base = long_df[[entity_col, "year"]].drop_duplicates().rename(columns={entity_col: "entity"})

    blocks_df = base.copy()
    for block, accounts, _, abs_value in BLOCK_DEFS:
        accounts_norm = [_normalize_text(acc) for acc in accounts]
        subset = long_df[long_df["account_norm"].isin(accounts_norm)]
        if subset.empty:
            blocks_df[block] = np.nan
            continue
        if block == "REV":
            order_map = {acc: idx for idx, acc in enumerate(accounts_norm)}
            grouped_accounts = (
                subset.groupby([entity_col, "year", "account_norm"], dropna=False)["value"]
                .sum(min_count=1)
                .reset_index()
            )
            grouped_accounts["account_order"] = grouped_accounts["account_norm"].map(order_map)
            grouped = (
                grouped_accounts.sort_values("account_order")
                .groupby([entity_col, "year"], dropna=False, as_index=False)
                .first()
            )
            grouped = grouped[[entity_col, "year", "value"]]
        else:
            grouped = (
                subset.groupby([entity_col, "year"], dropna=False)["value"]
                .sum(min_count=1)
                .reset_index()
            )
        if abs_value:
            grouped["value"] = grouped["value"].abs()
        grouped = grouped.rename(columns={entity_col: "entity", "value": block})
        blocks_df = blocks_df.merge(grouped, on=["entity", "year"], how="left")

    return blocks_df


def compute_ratios(blocks_df: pd.DataFrame) -> pd.DataFrame:
    df = blocks_df.copy()

    def safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
        return numer.where(denom > 0).div(denom.where(denom > 0))

    df["current_ratio"] = safe_div(df["AC"], df["PC"])
    df["cash_ratio"] = safe_div(df["CASH"], df["PC"])
    df["wc_to_rev"] = safe_div(df["AC"] - df["PC"], df["REV"])
    df["days_cash"] = safe_div(df["CASH"] * 365, df["OPEX_cash"])
    df["current_assets_ratio"] = safe_div(df["AC"], df["TA"])

    df["debt_to_assets"] = safe_div(df["TL"], df["TA"])
    df["equity_ratio"] = safe_div(df["EQ"], df["TA"])
    df["assets_to_liabilities"] = safe_div(df["TA"], df["TL"])
    df["current_liab_share"] = safe_div(df["PC"], df["TL"])
    df["net_debt_to_ebitda"] = safe_div(df["STD"] - df["CASH"], df["EBITDA"])

    df["gross_margin"] = safe_div(df["REV"] - df["OPEX_cash"], df["REV"])
    df["ebitda_margin"] = safe_div(df["EBITDA"], df["REV"])
    df["ebit_margin"] = safe_div(df["EBIT"], df["REV"])
    df["net_margin"] = safe_div(df["NET_INCOME"], df["REV"])
    df["roa"] = safe_div(df["NET_INCOME"], df["TA"])

    df["asset_turnover"] = safe_div(df["REV"], df["TA"])
    df["dso"] = safe_div(df["CxC"] * 365, df["REV"])
    df["dpo"] = safe_div(df["CxP"] * 365, df["OPEX_cash"])
    df["opex_cash_ratio"] = safe_div(df["OPEX_cash"], df["REV"])
    df["da_intensity"] = safe_div(df["DEPR_AMORT"], df["REV"])

    return df


def winsorize_ratios(
    df: pd.DataFrame, ratio_cols: Iterable[str], lower: float = 0.02, upper: float = 0.98
) -> pd.DataFrame:
    result = df.copy()
    for col in ratio_cols:
        series = result[col]
        valid = series.dropna()
        if valid.empty:
            continue
        lo = valid.quantile(lower)
        hi = valid.quantile(upper)
        result[col] = series.clip(lower=lo, upper=hi)
    return result


def _compute_size_factor(
    df: pd.DataFrame,
    target_col: str,
    clip_range: Tuple[float, float] = (0.7, 1.3),
) -> pd.Series:
    if target_col not in df.columns:
        return pd.Series(1.0, index=df.index)
    values = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0)
    values = values.clip(lower=0)
    log_vals = np.log1p(values)
    valid = log_vals[log_vals > 0]
    median = valid.median() if not valid.empty else np.nan
    if pd.isna(median) or median <= 0:
        return pd.Series(1.0, index=df.index)
    factor = log_vals / median
    low, high = clip_range
    factor = factor.clip(lower=low, upper=high)
    factor = factor.fillna(1.0)
    return factor


def compute_net_income_factor(
    df: pd.DataFrame,
    income_col: str = "NET_INCOME",
    clip_range: Tuple[float, float] = (0.7, 1.3),
) -> pd.Series:
    return _compute_size_factor(df, income_col, clip_range)


def compute_revenue_factor(
    df: pd.DataFrame,
    revenue_col: str = "REV",
    clip_range: Tuple[float, float] = (0.7, 1.3),
) -> pd.Series:
    return _compute_size_factor(df, revenue_col, clip_range)


def _percent_rank(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    n = len(valid)
    if n == 0:
        return series * np.nan
    if n == 1:
        return pd.Series([0.5] * n, index=valid.index).reindex(series.index)
    ranks = valid.rank(method="average")
    pct = (ranks - 1) / (n - 1)
    return pct.reindex(series.index)


def score_ratios(
    df: pd.DataFrame,
    ratio_specs: Dict[str, Dict[str, str]],
    dpo_range: Tuple[float, float] = (20, 60),
    dpo_zero: Tuple[float, float] = (0, 120),
) -> pd.DataFrame:
    scored = df.copy()
    for ratio, spec in ratio_specs.items():
        if ratio not in scored.columns:
            continue
        if spec["direction"] == "range":
            low, high = dpo_range
            zero_low, zero_high = dpo_zero
            values = scored[ratio]
            score = pd.Series(np.nan, index=values.index)
            in_range = values.between(low, high, inclusive="both")
            below = values < low
            above = values > high
            score[in_range] = 100
            score[below] = 100 * (values[below] - zero_low) / (low - zero_low)
            score[above] = 100 * (zero_high - values[above]) / (zero_high - high)
            scored[f"{ratio}_score"] = score.clip(lower=0, upper=100)
            continue

        pct = scored.groupby("year")[ratio].transform(_percent_rank)
        if spec["direction"] == "higher":
            scored[f"{ratio}_score"] = 100 * pct
        else:
            scored[f"{ratio}_score"] = 100 * (1 - pct)

    return scored


def aggregate_scores(
    df: pd.DataFrame,
    ratio_specs: Dict[str, Dict[str, str]],
    weights: Dict[str, float],
) -> pd.DataFrame:
    scored = df.copy()
    category_map: Dict[str, List[str]] = {}
    for ratio, spec in ratio_specs.items():
        category = spec["category"]
        category_map.setdefault(category, []).append(f"{ratio}_score")

    for category, cols in category_map.items():
        existing = [c for c in cols if c in scored.columns]
        if not existing:
            scored[f"{category}_score"] = np.nan
        else:
            scored[f"{category}_score"] = scored[existing].mean(axis=1, skipna=True)

    total = pd.Series(0.0, index=scored.index)
    weight_sum = pd.Series(0.0, index=scored.index)
    for category, weight in weights.items():
        col = f"{category}_score"
        if col not in scored.columns:
            continue
        valid = scored[col].notna()
        total[valid] = total[valid] + scored.loc[valid, col] * weight
        weight_sum[valid] = weight_sum[valid] + weight

    scored["score_financiero"] = total.where(weight_sum > 0).div(weight_sum.where(weight_sum > 0))
    return scored


def apply_size_factors(
    scored_df: pd.DataFrame,
    weights: Dict[str, float],
    factor_cols: Iterable[str],
) -> pd.DataFrame:
    adjusted = scored_df.copy()
    factor = pd.Series(1.0, index=adjusted.index)
    for col in factor_cols:
        if col not in adjusted.columns:
            adjusted[col] = 1.0
        factor = factor * pd.to_numeric(adjusted[col], errors="coerce").fillna(1.0)
    for category in weights.keys():
        col = f"{category}_score"
        if col in adjusted.columns:
            adjusted[col] = adjusted[col] * factor

    total = pd.Series(0.0, index=adjusted.index)
    weight_sum = pd.Series(0.0, index=adjusted.index)
    for category, weight in weights.items():
        col = f"{category}_score"
        if col not in adjusted.columns:
            continue
        valid = adjusted[col].notna()
        total[valid] = total[valid] + adjusted.loc[valid, col] * weight
        weight_sum[valid] = weight_sum[valid] + weight

    adjusted["score_financiero"] = total.where(weight_sum > 0).div(weight_sum.where(weight_sum > 0))
    return adjusted


def apply_profit_factor(
    scored_df: pd.DataFrame,
    weights: Dict[str, float],
    factor_col: str = "net_income_factor",
) -> pd.DataFrame:
    return apply_size_factors(scored_df, weights, [factor_col])
