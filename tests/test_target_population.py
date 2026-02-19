from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.target_population import compute_target_population


def test_compute_target_population_from_comparacion_desglose_sum_column() -> None:
    comparacion_df = pd.DataFrame(
        {
            "Grupo Edad": ["0-19", "20-39", "TOTAL"],
            "Pacientes por edad": [10, 20, 999],
        }
    )

    result = compute_target_population(
        comparacion_df=comparacion_df,
        eps_edad_df=pd.DataFrame(),
        prevalencia_df_raw=pd.DataFrame([["GrupoEdad", "Prevalencia"], ["0-19", 0.1]]),
    )

    summary = result["summary_metrics"]
    assert np.isclose(summary["posibles_atendidos_valle"], 30.0)
    assert result["metadata"]["method"] == "comparacion_desglose"
    assert "pacientes" in str(result["metadata"]["objective_source_column"]).lower()

    edad_view = result["edad_view"]
    total_row = edad_view[edad_view["GrupoEdad"] == "TOTAL"]
    assert not total_row.empty
    assert np.isclose(pd.to_numeric(total_row["PacientesPorEdad"], errors="coerce").iloc[0], 30.0)


def test_compute_target_population_fallback_legacy_when_missing_pacientes_por_edad() -> None:
    comparacion_df = pd.DataFrame(
        {
            "ENTIDAD": ["EPS A", "EPS B", "TOTAL"],
            "SANTANDER AFILIADOS": [40, 60, 100],
            "VALLE DEL CAUCA AFILIADOS": [80, 120, 200],
            "ICB ATENDIDOS": [4, 6, 10],
            "GRUPO FOSCAL ATENDIDOS": [2, 3, 5],
        }
    )
    eps_edad_df = pd.DataFrame(
        {
            "DEPARTAMENTO": ["VALLE DEL CAUCA", "VALLE DEL CAUCA"],
            "GRUPOEDAD": ["De 0 a 4 anos", "De 20 a 24 anos"],
            "TOTAL AFILIADOS": [80, 120],
        }
    )
    prevalencia_raw = pd.DataFrame([["GrupoEdad", "Prevalencia"], ["0-19", 0.1], ["20-39", 0.2]])

    result = compute_target_population(
        comparacion_df=comparacion_df,
        eps_edad_df=eps_edad_df,
        prevalencia_df_raw=prevalencia_raw,
    )

    assert result["metadata"]["method"] == "legacy"
    assert any("Pacientes por edad" in msg for msg in result["warnings"])


def test_compute_target_population_base_formula() -> None:
    comparacion_df = pd.DataFrame(
        {
            "ENTIDAD": ["EPS A", "EPS B", "TOTAL"],
            "SANTANDER AFILIADOS": [40, 60, 100],
            "VALLE DEL CAUCA AFILIADOS": [80, 120, 200],
            "ICB ATENDIDOS": [4, 6, 10],
            "GRUPO FOSCAL ATENDIDOS": [2, 3, 5],
        }
    )

    eps_edad_df = pd.DataFrame(
        {
            "DEPARTAMENTO": ["VALLE DEL CAUCA", "VALLE DEL CAUCA", "VALLE DEL CAUCA"],
            "GRUPOEDAD": ["De 0 a 4 anos", "De 20 a 24 anos", "De 65 a 69 anos"],
            "TOTAL AFILIADOS": [80, 70, 50],
        }
    )

    prevalencia_raw = pd.DataFrame(
        [
            ["nota", "nota"],
            ["GrupoEdad", "Prevalencia"],
            ["0-19", 10],
            ["20-39", 20],
            ["60-79", 30],
        ]
    )

    result = compute_target_population(
        comparacion_df=comparacion_df,
        eps_edad_df=eps_edad_df,
        prevalencia_df_raw=prevalencia_raw,
    )

    summary = result["summary_metrics"]
    assert np.isclose(summary["pct_atendido_santander"], 0.15)
    assert np.isclose(summary["atendidos_santander"], 15.0)
    assert np.isclose(summary["posibles_atendidos_valle"], 30.0)
    assert np.isclose(summary["afiliados_valle_total"], 200.0)

    edad_view = result["edad_view"]
    total_row = edad_view[edad_view["GrupoEdad"] == "TOTAL"]
    assert not total_row.empty
    total_pacientes = pd.to_numeric(total_row["PacientesPorEdad"], errors="coerce").iloc[0]
    assert np.isclose(total_pacientes, summary["posibles_atendidos_valle"])


def test_compute_target_population_missing_columns_returns_warnings() -> None:
    result = compute_target_population(
        comparacion_df=pd.DataFrame({"A": [1]}),
        eps_edad_df=pd.DataFrame(),
        prevalencia_df_raw=pd.DataFrame(),
    )

    assert result["warnings"]
    assert pd.isna(result["summary_metrics"]["posibles_atendidos_valle"])
    assert isinstance(result["formula_view"], pd.DataFrame)
