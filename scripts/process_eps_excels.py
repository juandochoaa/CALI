from __future__ import annotations

import csv
from pathlib import Path

from src.data.eps_excel import (
    age_group_share,
    financials_long,
    load_caracterizacion02,
    load_eps_financials,
)


ROOT = Path(".")
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    financials_path = RAW / "ESTADOS FINANCIEROS EPS Cali.xlsx"
    afiliados_path = RAW / "CIFRAS EPS DICIEMBRE-2025.xlsx"
    if not financials_path.exists():
        financials_path = ROOT / "ESTADOS FINANCIEROS EPS Cali.xlsx"
    if not afiliados_path.exists():
        afiliados_path = ROOT / "CIFRAS EPS DICIEMBRE-2025.xlsx"

    # EPS financials
    fin_records = load_eps_financials(financials_path)
    fin_long = financials_long(fin_records)
    write_csv(OUT / "eps_financials_long.csv", fin_long)

    # Affiliates by age group for Valle del Cauca
    afiliados = load_caracterizacion02(
        afiliados_path,
        department="VALLE DEL CAUCA",
    )
    share_within_age = age_group_share(afiliados, mode="within_age")
    share_within_eps = age_group_share(afiliados, mode="within_eps")
    write_csv(OUT / "eps_age_share_within_age.csv", share_within_age)
    write_csv(OUT / "eps_age_share_within_eps.csv", share_within_eps)


if __name__ == "__main__":
    main()
