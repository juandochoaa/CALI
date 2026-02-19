from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd


PAGE_PATH = (
    Path(__file__).resolve().parents[1]
    / "dashboards"
    / "pages"
    / "5_Proyecciones_EEFF.py"
)


def _load_constructor_helpers() -> dict[str, object]:
    source = PAGE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {
        "build_constructor_scenarios",
        "perpetuity_pv",
        "_format_es_number",
        "_format_mm_es",
        "build_rent_reference_table",
        "compute_fixed_rent_base",
        "calibrate_constructor_percentages",
    }
    selected: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            selected.append(node)
    module = ast.Module(body=selected, type_ignores=[])
    namespace: dict[str, object] = {"np": np, "pd": pd}
    exec(compile(module, str(PAGE_PATH), "exec"), namespace)
    return namespace


def test_rent_reference_table_values() -> None:
    helpers = _load_constructor_helpers()
    build_rent_reference_table = helpers["build_rent_reference_table"]

    table = build_rent_reference_table(4_619.2)
    min_monthly = float(
        table.loc[table["Escenario"] == "Minimo", "Arriendo total mensual (COP)"].iloc[0]
    )
    avg_monthly = float(
        table.loc[table["Escenario"] == "Promedio", "Arriendo total mensual (COP)"].iloc[0]
    )
    max_monthly = float(
        table.loc[table["Escenario"] == "Maximo", "Arriendo total mensual (COP)"].iloc[0]
    )

    assert np.isclose(min_monthly, 118_279_235.2, rtol=0.0, atol=1e-6)
    assert np.isclose(avg_monthly, 200_237_700.8, rtol=0.0, atol=1e-6)
    assert np.isclose(max_monthly, 347_992_051.2, rtol=0.0, atol=1e-6)


def test_fixed_rent_monthly_to_annual_conversion() -> None:
    helpers = _load_constructor_helpers()
    compute_fixed_rent_base = helpers["compute_fixed_rent_base"]

    monthly, annual = compute_fixed_rent_base(4_619.2, 43_349.0)
    assert np.isclose(monthly, 200_237_700.8, rtol=0.0, atol=1e-6)
    assert np.isclose(annual, 2_402_852_409.6, rtol=0.0, atol=1e-6)


def test_calibration_matches_vp_target_for_revenue_and_utility() -> None:
    helpers = _load_constructor_helpers()
    build_constructor_scenarios = helpers["build_constructor_scenarios"]
    perpetuity_pv = helpers["perpetuity_pv"]
    calibrate_constructor_percentages = helpers["calibrate_constructor_percentages"]

    years = [2026, 2027, 2028, 2029, 2030]
    proj_statement = pd.DataFrame(
        {
            2026: {"INGRESOS": 12_000.0, "UTILIDADNETA": 2_000.0},
            2027: {"INGRESOS": 12_500.0, "UTILIDADNETA": 2_200.0},
            2028: {"INGRESOS": 13_000.0, "UTILIDADNETA": 2_400.0},
            2029: {"INGRESOS": 13_700.0, "UTILIDADNETA": 2_600.0},
            2030: {"INGRESOS": 14_500.0, "UTILIDADNETA": 2_900.0},
        }
    )
    fixed_annual_payment = 1_000.0
    discount_rate = 0.15
    growth_rate = 0.03

    base_df = build_constructor_scenarios(
        proj_statement=proj_statement,
        years=years,
        fixed_payment=fixed_annual_payment,
        pct_revenue=0.0,
        pct_utility=0.0,
    )
    calib = calibrate_constructor_percentages(
        constructor_df=base_df,
        discount_rate=discount_rate,
        perpetuity_growth=growth_rate,
        fixed_annual_payment=fixed_annual_payment,
    )

    pct_revenue_auto = float(calib["pct_revenue_auto"])
    pct_utility_auto = float(calib["pct_utility_auto"])
    assert np.isfinite(pct_revenue_auto)
    assert np.isfinite(pct_utility_auto)

    calibrated_df = build_constructor_scenarios(
        proj_statement=proj_statement,
        years=years,
        fixed_payment=fixed_annual_payment,
        pct_revenue=pct_revenue_auto,
        pct_utility=pct_utility_auto,
    )
    year_index = calibrated_df["Ano"].astype(int)
    pay_revenue = pd.Series(calibrated_df["Pago_PctIngresos"].to_numpy(), index=year_index)
    pay_utility = pd.Series(calibrated_df["Pago_PctUtilidad"].to_numpy(), index=year_index)

    _, _, vp_revenue = perpetuity_pv(pay_revenue, discount_rate, growth_rate)
    _, _, vp_utility = perpetuity_pv(pay_utility, discount_rate, growth_rate)
    vp_target = float(calib["vp_target"])

    assert np.isclose(vp_revenue, vp_target, rtol=1e-9, atol=1e-6)
    assert np.isclose(vp_utility, vp_target, rtol=1e-9, atol=1e-6)


def test_calibration_returns_nan_when_discount_not_above_growth() -> None:
    helpers = _load_constructor_helpers()
    build_constructor_scenarios = helpers["build_constructor_scenarios"]
    calibrate_constructor_percentages = helpers["calibrate_constructor_percentages"]

    years = [2026, 2027]
    proj_statement = pd.DataFrame(
        {
            2026: {"INGRESOS": 1_000.0, "UTILIDADNETA": 200.0},
            2027: {"INGRESOS": 1_050.0, "UTILIDADNETA": 220.0},
        }
    )
    base_df = build_constructor_scenarios(
        proj_statement=proj_statement,
        years=years,
        fixed_payment=100.0,
        pct_revenue=0.0,
        pct_utility=0.0,
    )
    calib = calibrate_constructor_percentages(
        constructor_df=base_df,
        discount_rate=0.03,
        perpetuity_growth=0.03,
        fixed_annual_payment=100.0,
    )

    assert np.isnan(calib["vp_target"])
    warnings = calib.get("warnings", [])
    assert any("descuento" in str(msg).lower() for msg in warnings)


def test_calibration_allows_pct_utility_over_100_percent() -> None:
    helpers = _load_constructor_helpers()
    build_constructor_scenarios = helpers["build_constructor_scenarios"]
    calibrate_constructor_percentages = helpers["calibrate_constructor_percentages"]

    years = [2026, 2027, 2028]
    proj_statement = pd.DataFrame(
        {
            2026: {"INGRESOS": 10_000.0, "UTILIDADNETA": 10.0},
            2027: {"INGRESOS": 10_200.0, "UTILIDADNETA": 8.0},
            2028: {"INGRESOS": 10_500.0, "UTILIDADNETA": 12.0},
        }
    )
    base_df = build_constructor_scenarios(
        proj_statement=proj_statement,
        years=years,
        fixed_payment=5_000.0,
        pct_revenue=0.0,
        pct_utility=0.0,
    )
    calib = calibrate_constructor_percentages(
        constructor_df=base_df,
        discount_rate=0.14,
        perpetuity_growth=0.03,
        fixed_annual_payment=5_000.0,
    )

    pct_utility_auto = float(calib["pct_utility_auto"])
    assert pct_utility_auto > 1.0
    warnings = calib.get("warnings", [])
    assert any("100%" in str(msg) for msg in warnings)
