from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.eps_montecarlo import PROBABILITY_COLUMNS, run_eps_montecarlo


def _build_minimal_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]

    upc_df = pd.DataFrame(
        {
            "Regimen": ["CONTRIBUTIVO"],
            "GrupoEdad": ["1-4 años"],
            **{year: [1000.0 + (year - 2019) * 50.0] for year in years},
        }
    )

    eps_edad_df = pd.DataFrame(
        {
            "EPS": ["ASMETSALUD EPS"],
            "REGIMEN": ["CONTRIBUTIVO"],
            "GRUPOEDAD": ["De 0 a 4 años"],
            "TOTAL AFILIADOS": [100000.0],
        }
    )

    afiliados_series = [100.0, 140.0, 196.0, 274.4, 274.4, 274.4, 274.4, 274.4]
    eps_anos_df = pd.DataFrame({"EPS": ["ASMETSALUD EPS"], **{y: [v] for y, v in zip(years, afiliados_series)}})

    cuentas = []
    ingresos = [100000.0] * len(years)
    lr_series = [0.40, 0.40, 0.40, 0.40, 0.70, 0.70, 0.70, 0.70]
    costos = [ingresos[i] * lr_series[i] for i in range(len(years))]
    gadmin = [10000.0] * len(years)
    efectivo = [15000.0] * len(years)
    patrimonio = [25000.0] * len(years)
    reservas = [12000.0] * len(years)
    inversiones = [5000.0] * len(years)
    personal = [4000.0] * len(years)
    transporte = [1500.0] * len(years)
    impuestos = [1200.0] * len(years)
    otros = [800.0] * len(years)

    def add_row(cuenta: str, values: list[float]) -> None:
        cuentas.append({"EPS": "ASMETSALUD EPS", "CUENTA": cuenta, **{y: values[i] for i, y in enumerate(years)}})

    add_row("TotalIngresoOperativo", ingresos)
    add_row("Otroscostospornaturaleza", costos)
    add_row("Gastosadministrativos", gadmin)
    add_row("EfectivooEquivalentes", efectivo)
    add_row("Totaldepatrimonio", patrimonio)
    add_row("Provisionesparaotrospasivosygastos", reservas)
    add_row("Activosfinancierosdecortoplazo", inversiones)
    add_row("Gastosporbeneficiosdelosempleado", personal)
    add_row("Costosdetransporte", transporte)
    add_row("Impuestoycontribuciones", impuestos)
    add_row("Otrosgastos", otros)
    add_row("Comercialesyotrascuentasapagar", [12000.0] * len(years))
    eps_eeff_df = pd.DataFrame(cuentas)

    return upc_df, eps_eeff_df, eps_edad_df, eps_anos_df


def test_mean_reversion_params_and_removed_cashdays_zero() -> None:
    upc_df, eps_eeff_df, eps_edad_df, eps_anos_df = _build_minimal_inputs()
    results_df, diagnostics = run_eps_montecarlo(
        upc_df=upc_df,
        eps_eeff_df=eps_eeff_df,
        eps_edad_df=eps_edad_df,
        eps_afiliados_hist_df=eps_anos_df,
        eps_obj=["ASMETSALUD EPS"],
        n_sim=200,
        horizon_end=2028,
        random_seed=123,
    )

    assert "P_avg_CashDays_lt_0" not in results_df.columns
    assert "P_avg_CashDays_lt_0" not in PROBABILITY_COLUMNS
    assert "P_avg_CashDays_lt_15" in results_df.columns
    assert "P_avg_PayDays_out_20_70" in results_df.columns

    params_df = diagnostics["mean_reversion_params"]
    assert not params_df.empty

    row = params_df.iloc[0]
    assert 0.05 <= float(row["lambda_g"]) <= 0.25
    assert 0.05 <= float(row["lambda_lr"]) <= 0.25

    # g: reciente menor que media larga => el promedio simulado debe moverse hacia arriba.
    target_g = float(row["mu_long_g"])
    m0_g = float(row["mu_recent_g"])
    m1_g = m0_g + float(row["lambda_g"]) * (target_g - m0_g)
    assert abs(m1_g - target_g) < abs(m0_g - target_g) + 1e-12

    # LR: reciente mayor que media larga => el promedio simulado debe moverse hacia abajo.
    target_lr = float(row["mu_long_lr"])
    m0_lr = float(row["mu_recent_lr"])
    m1_lr = m0_lr + float(row["lambda_lr"]) * (target_lr - m0_lr)
    assert abs(m1_lr - target_lr) < abs(m0_lr - target_lr) + 1e-12

    assert np.isfinite(results_df["P_avg_CM_ratio_lt_1"]).all()
    assert np.isfinite(results_df["P_avg_PayDays_out_20_70"]).all()
