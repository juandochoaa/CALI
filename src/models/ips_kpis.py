from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def compute_market_share_by_year(ing_oper_df: pd.DataFrame) -> pd.DataFrame:
    if ing_oper_df.empty:
        return pd.DataFrame(columns=["entity", "year", "REV", "MarketShare"])

    work = ing_oper_df.copy()
    needed = {"entity", "year", "REV"}
    if not needed.issubset(set(work.columns)):
        return pd.DataFrame(columns=["entity", "year", "REV", "MarketShare"])

    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work["REV"] = pd.to_numeric(work["REV"], errors="coerce")
    work = work.dropna(subset=["entity", "year", "REV"]).copy()
    if work.empty:
        return pd.DataFrame(columns=["entity", "year", "REV", "MarketShare"])

    work["year"] = work["year"].astype(int)
    totals = work.groupby("year", as_index=False)["REV"].sum().rename(columns={"REV": "REV_total"})
    out = work.merge(totals, on="year", how="left")
    out["MarketShare"] = np.divide(
        out["REV"],
        out["REV_total"],
        out=np.full(len(out), np.nan, dtype=float),
        where=out["REV_total"].to_numpy(dtype=float) > 0,
    )
    return out[["entity", "year", "REV", "MarketShare"]].sort_values(["entity", "year"]).reset_index(drop=True)


def build_ips_historical_kpis(
    ratios_df: pd.DataFrame,
    ing_oper_df: pd.DataFrame,
    selected_ips: str,
    years_universe: Sequence[int] | None = None,
) -> pd.DataFrame:
    years = (
        pd.Series(list(years_universe), dtype="Int64").dropna().astype(int).tolist()
        if years_universe is not None
        else []
    )

    margins = pd.DataFrame(columns=["year", "MargenBruto", "MargenNeto"])
    if not ratios_df.empty and {"entity", "year", "gross_margin", "net_margin"}.issubset(set(ratios_df.columns)):
        margins = ratios_df[ratios_df["entity"] == selected_ips][["year", "gross_margin", "net_margin"]].copy()
        margins["year"] = pd.to_numeric(margins["year"], errors="coerce")
        margins = margins.dropna(subset=["year"]).copy()
        margins["year"] = margins["year"].astype(int)
        margins = margins.rename(columns={"gross_margin": "MargenBruto", "net_margin": "MargenNeto"})

    share_long = compute_market_share_by_year(ing_oper_df=ing_oper_df)
    share = pd.DataFrame(columns=["year", "TamanioMercado"])
    if not share_long.empty:
        share = share_long[share_long["entity"] == selected_ips][["year", "MarketShare"]].copy()
        share = share.rename(columns={"MarketShare": "TamanioMercado"})

    if years:
        base = pd.DataFrame({"year": sorted(set(years))})
    else:
        base = (
            pd.concat(
                [
                    margins[["year"]] if not margins.empty else pd.DataFrame(columns=["year"]),
                    share[["year"]] if not share.empty else pd.DataFrame(columns=["year"]),
                ],
                ignore_index=True,
            )
            .drop_duplicates()
            .sort_values("year")
            .reset_index(drop=True)
        )

    out = (
        base.merge(margins, on="year", how="left")
        .merge(share, on="year", how="left")
        .sort_values("year")
        .reset_index(drop=True)
    )
    out = out.rename(columns={"year": "Año"})
    return out
