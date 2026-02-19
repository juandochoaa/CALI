from __future__ import annotations

import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd

AGE_GROUP_ORDER = ["0-19", "20-39", "40-59", "60-79", "80+"]


def normalize_text(text: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return " ".join(normalized.lower().split())


def find_col(columns: list[str], includes: list[str]) -> str | None:
    for col in columns:
        norm = normalize_text(col)
        if all(token in norm for token in includes):
            return col
    return None


def parse_prevalencia(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["GrupoEdad", "Prevalencia"])

    header_idx = None
    for idx, row in df.iterrows():
        row_vals = [normalize_text(x) for x in row.tolist()]
        if "grupoedad" in row_vals and "prevalencia" in row_vals:
            header_idx = idx
            break

    if header_idx is None:
        return pd.DataFrame(columns=["GrupoEdad", "Prevalencia"])

    records: list[dict[str, Any]] = []
    for _, row in df.iloc[header_idx + 1 :].iterrows():
        group = row.iloc[0]
        prev = row.iloc[1] if len(row) > 1 else None
        if pd.isna(group) or str(group).strip() == "":
            break
        records.append({"GrupoEdad": str(group).strip(), "Prevalencia": prev})

    out = pd.DataFrame(records)
    out["Prevalencia"] = pd.to_numeric(out["Prevalencia"], errors="coerce")
    if not out["Prevalencia"].dropna().empty and out["Prevalencia"].max() > 1:
        out["Prevalencia"] = out["Prevalencia"] / 100.0
    out = out.dropna(subset=["Prevalencia"]).copy()
    out["GrupoEdad"] = out["GrupoEdad"].astype(str).str.strip()
    return out


def map_quinquenio_to_group(text: object) -> str | None:
    if not isinstance(text, str):
        return None
    norm = normalize_text(text)

    if "80" in norm and "mas" in norm:
        return "80+"

    nums = re.findall(r"\d+", norm)
    if len(nums) >= 2:
        start = int(nums[0])
        if start <= 19:
            return "0-19"
        if start <= 39:
            return "20-39"
        if start <= 59:
            return "40-59"
        if start <= 79:
            return "60-79"

    if len(nums) == 1:
        start = int(nums[0])
        if start >= 80:
            return "80+"

    return None


def _append_total_row(df: pd.DataFrame, label_col: str, numeric_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    total_row: dict[str, Any] = {label_col: "TOTAL"}
    for col in df.columns:
        if col == label_col:
            continue
        if col in numeric_cols:
            total_row[col] = pd.to_numeric(df[col], errors="coerce").sum(min_count=1)
        else:
            total_row[col] = np.nan
    return pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)


def _empty_snapshot(
    warnings: list[str],
    prevalencia_df: pd.DataFrame | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "summary_metrics": {
            "pct_atendido_santander": np.nan,
            "atendidos_santander": np.nan,
            "posibles_atendidos_valle": np.nan,
            "sam_pacientes_valle": np.nan,
            "afiliados_valle_total": np.nan,
            "afiliados_santander_total": np.nan,
        },
        "eps_view": pd.DataFrame(),
        "edad_view": pd.DataFrame(),
        "formula_view": _build_formula_view(),
        "edad_chart_df": pd.DataFrame(),
        "warnings": warnings,
        "metadata": {
            "status": "warning" if warnings else "ok",
            "prevalencia_df": prevalencia_df if prevalencia_df is not None else pd.DataFrame(),
        },
    }
    if metadata:
        payload["metadata"].update(metadata)
    return payload


def _build_formula_view(method: str = "legacy") -> pd.DataFrame:
    if method == "comparacion_desglose":
        return pd.DataFrame(
            {
                "Campo": [
                    "TAM",
                    "SAM por grupo de edad",
                    "SAM total",
                    "Factor de captura Santander",
                    "SOM por grupo de edad",
                    "SOM (poblacion objetivo)",
                    "Nota de implementacion",
                ],
                "Formula": [
                    "TAM = Total afiliados en Valle del Cauca",
                    "SAM_g = Afiliados_g * Prev_g",
                    "SAM = sum_g(SAM_g)",
                    "alpha = Atendidos_Santander / Afiliados_Santander",
                    "SOM_g = SAM_g * alpha",
                    "Objetivo = sum_g(PacientesPorEdad_g)",
                    "Pacientes por edad ya incorpora el factor de captura alpha.",
                ],
            }
        )

    return pd.DataFrame(
        {
            "Campo": [
                "% atendido Santander",
                "Posibles atendidos Valle",
                "% poblacion Valle",
                "Pacientes estimados",
                "Ponderacion",
                "Pacientes por edad",
                "Atendidos (EPS)",
            ],
            "Formula": [
                "(ICB atendidos + Grupo Foscal atendidos) / Afiliados Santander (TOTAL)",
                "% atendido Santander * Afiliados Valle (TOTAL)",
                "Afiliados grupo / Total afiliados Valle",
                "Prevalencia * % poblacion Valle * Total afiliados Valle",
                "Pacientes estimados grupo / Pacientes estimados totales",
                "Ponderacion * Posibles atendidos Valle",
                "ICB atendidos + Grupo Foscal atendidos",
            ],
        }
    )


def _compute_from_comparacion_desglose(
    comparacion_df: pd.DataFrame,
    prevalencia_df: pd.DataFrame,
    warnings: list[str],
) -> dict[str, Any] | None:
    comp_cols = [str(c) for c in comparacion_df.columns]
    pacientes_edad_col = find_col(comp_cols, ["pacientes", "edad"])
    if not pacientes_edad_col:
        return None

    grupo_edad_col = find_col(comp_cols, ["grupo", "edad"])
    if grupo_edad_col is None:
        grupo_edad_col = find_col(comp_cols, ["quinquenio"])
    if grupo_edad_col is None:
        grupo_edad_col = find_col(comp_cols, ["rango", "edad"])
    if grupo_edad_col is None:
        for col in comp_cols:
            norm = normalize_text(col)
            if "edad" in norm and "pacientes" not in norm:
                grupo_edad_col = col
                break

    pacientes_totales_col = find_col(comp_cols, ["pacientes", "totales"]) or find_col(
        comp_cols, ["pacientes", "total"]
    )

    afiliados_valle_col = find_col(comp_cols, ["valle", "afiliados"])
    afiliados_sant_col = find_col(comp_cols, ["santander", "afiliados"])
    icb_col = find_col(comp_cols, ["icb", "atendidos"])
    foscal_col = find_col(comp_cols, ["foscal", "atendidos"])

    work = comparacion_df.copy()
    if grupo_edad_col:
        work["GrupoEdad"] = work[grupo_edad_col].astype(str).str.strip()
        is_total = work["GrupoEdad"].map(normalize_text) == "total"
        if is_total.any():
            work = work[~is_total].copy()
    else:
        work["GrupoEdad"] = [f"Grupo {idx + 1}" for idx in range(len(work))]

    work["PacientesPorEdad"] = pd.to_numeric(work[pacientes_edad_col], errors="coerce")
    work = work[work["PacientesPorEdad"].notna()].copy()
    if work.empty:
        warnings.append(
            "La hoja Comparacion_Desglose no tiene valores numericos validos en 'Pacientes por edad'."
        )
        return None

    edad_chart_df = (
        work.groupby("GrupoEdad", as_index=False, sort=False)["PacientesPorEdad"]
        .sum(min_count=1)
        .reset_index(drop=True)
    )
    objetivo = pd.to_numeric(edad_chart_df["PacientesPorEdad"], errors="coerce").sum(min_count=1)
    sam_total = (
        pd.to_numeric(work[pacientes_totales_col], errors="coerce").sum(min_count=1)
        if pacientes_totales_col
        else np.nan
    )

    edad_view = edad_chart_df.copy()
    total_row = {
        "GrupoEdad": "TOTAL",
        "PacientesPorEdad": objetivo,
    }
    edad_view = pd.concat([edad_view, pd.DataFrame([total_row])], ignore_index=True)

    sant_total = np.nan
    valle_total = np.nan
    atendidos_total = np.nan
    pct_atendido = np.nan

    if afiliados_sant_col and afiliados_valle_col and icb_col and foscal_col:
        agg = comparacion_df.copy()
        if grupo_edad_col:
            agg["_grupo_norm"] = agg[grupo_edad_col].astype(str).map(normalize_text)
            total_row_comp = agg[agg["_grupo_norm"] == "total"]
            if not total_row_comp.empty:
                row = total_row_comp.iloc[0]
                sant_total = pd.to_numeric(row[afiliados_sant_col], errors="coerce")
                valle_total = pd.to_numeric(row[afiliados_valle_col], errors="coerce")
                atendidos_total = pd.to_numeric(row[icb_col], errors="coerce") + pd.to_numeric(
                    row[foscal_col], errors="coerce"
                )
            else:
                sant_total = pd.to_numeric(agg[afiliados_sant_col], errors="coerce").sum(min_count=1)
                valle_total = pd.to_numeric(agg[afiliados_valle_col], errors="coerce").sum(min_count=1)
                atendidos_total = pd.to_numeric(agg[icb_col], errors="coerce").sum(min_count=1) + pd.to_numeric(
                    agg[foscal_col], errors="coerce"
                ).sum(min_count=1)
        else:
            sant_total = pd.to_numeric(agg[afiliados_sant_col], errors="coerce").sum(min_count=1)
            valle_total = pd.to_numeric(agg[afiliados_valle_col], errors="coerce").sum(min_count=1)
            atendidos_total = pd.to_numeric(agg[icb_col], errors="coerce").sum(min_count=1) + pd.to_numeric(
                agg[foscal_col], errors="coerce"
            ).sum(min_count=1)

        if pd.notna(atendidos_total) and pd.notna(sant_total) and sant_total > 0:
            pct_atendido = atendidos_total / sant_total

    summary_metrics = {
        "pct_atendido_santander": pct_atendido,
        "atendidos_santander": atendidos_total,
        "posibles_atendidos_valle": objetivo,
        "sam_pacientes_valle": sam_total,
        "afiliados_valle_total": valle_total,
        "afiliados_santander_total": sant_total,
    }

    metadata = {
        "status": "warning" if warnings else "ok",
        "method": "comparacion_desglose",
        "objective_source_column": pacientes_edad_col,
        "comparacion_columns": comp_cols,
        "prevalencia_df": prevalencia_df,
    }

    return {
        "summary_metrics": summary_metrics,
        "eps_view": pd.DataFrame(),
        "edad_view": edad_view,
        "formula_view": _build_formula_view(method="comparacion_desglose"),
        "edad_chart_df": edad_chart_df,
        "warnings": warnings,
        "metadata": metadata,
    }


def _compute_target_population_legacy(
    comparacion_df: pd.DataFrame,
    eps_edad_df: pd.DataFrame,
    prev_df: pd.DataFrame,
    warnings: list[str],
) -> dict[str, Any]:
    if comparacion_df.empty:
        warnings.append("No se pudo leer la hoja Comparacion.")
        return _empty_snapshot(warnings, prevalencia_df=prev_df)

    comp_cols = [str(c) for c in comparacion_df.columns]
    ent_col = find_col(comp_cols, ["entidad"]) or comp_cols[0]
    sant_col = find_col(comp_cols, ["santander", "afiliados"])
    valle_col = find_col(comp_cols, ["valle", "afiliados"])
    icb_col = find_col(comp_cols, ["icb", "atendidos"])
    foscal_col = find_col(comp_cols, ["foscal", "atendidos"])

    detected_comp = {
        "ent_col": ent_col,
        "sant_col": sant_col,
        "valle_col": valle_col,
        "icb_col": icb_col,
        "foscal_col": foscal_col,
    }

    if not all([sant_col, valle_col, icb_col, foscal_col]):
        warnings.append("No se encontraron columnas requeridas en Comparacion.")
        return _empty_snapshot(
            warnings,
            prevalencia_df=prev_df,
            metadata={"comparacion_columns": comp_cols, "comparacion_detected": detected_comp},
        )

    comp_work = comparacion_df.copy()
    comp_work[ent_col] = comp_work[ent_col].astype(str).str.strip()
    comp_work["ent_norm"] = comp_work[ent_col].map(normalize_text)

    total_row = comp_work[comp_work["ent_norm"] == "total"]
    if not total_row.empty:
        total_vals = total_row.iloc[0]
        sant_total = pd.to_numeric(total_vals[sant_col], errors="coerce")
        valle_total = pd.to_numeric(total_vals[valle_col], errors="coerce")
        icb_total = pd.to_numeric(total_vals[icb_col], errors="coerce")
        foscal_total = pd.to_numeric(total_vals[foscal_col], errors="coerce")
    else:
        warnings.append("No se encontro fila TOTAL en Comparacion; se usaran sumas por EPS.")
        sant_total = pd.to_numeric(comp_work[sant_col], errors="coerce").sum(min_count=1)
        valle_total = pd.to_numeric(comp_work[valle_col], errors="coerce").sum(min_count=1)
        icb_total = pd.to_numeric(comp_work[icb_col], errors="coerce").sum(min_count=1)
        foscal_total = pd.to_numeric(comp_work[foscal_col], errors="coerce").sum(min_count=1)

    atendidos_total = icb_total + foscal_total
    pct_atendido = (
        atendidos_total / sant_total
        if pd.notna(atendidos_total) and pd.notna(sant_total) and sant_total > 0
        else np.nan
    )
    posibles_valle = (
        pct_atendido * valle_total if pd.notna(pct_atendido) and pd.notna(valle_total) else np.nan
    )

    eps_table = comp_work[~comp_work["ent_norm"].isin(["total"])].copy()
    if eps_table.empty:
        eps_table = comp_work.copy()
    eps_table.loc[eps_table["ent_norm"].isin(["otras", "otros"]), ent_col] = "OTROS"
    eps_table["Atendidos"] = pd.to_numeric(eps_table[icb_col], errors="coerce") + pd.to_numeric(
        eps_table[foscal_col], errors="coerce"
    )
    eps_view = eps_table[[ent_col, sant_col, valle_col, icb_col, foscal_col, "Atendidos"]].rename(
        columns={
            ent_col: "EPS",
            sant_col: "Santander afiliados",
            valle_col: "Valle afiliados",
            icb_col: "ICB atendidos",
            foscal_col: "Grupo Foscal atendidos",
        }
    )
    eps_view = _append_total_row(
        eps_view,
        "EPS",
        [
            "Santander afiliados",
            "Valle afiliados",
            "ICB atendidos",
            "Grupo Foscal atendidos",
            "Atendidos",
        ],
    )

    edad_view = pd.DataFrame()
    edad_chart_df = pd.DataFrame()
    sam_total = np.nan

    detected_edad_work: dict[str, Any] = {}
    if eps_edad_df.empty:
        warnings.append("No se pudo leer la hoja EPS_Edad.")
    else:
        edad_cols = [str(c) for c in eps_edad_df.columns]
        dept_col = find_col(edad_cols, ["departamento"])
        age_col = find_col(edad_cols, ["grupoedad"]) or find_col(edad_cols, ["quinquenio"])
        total_col = find_col(edad_cols, ["total", "afiliados"]) or find_col(edad_cols, ["afiliados"])

        detected_edad = {
            "dept_col": dept_col,
            "age_col": age_col,
            "total_col": total_col,
        }

        if not age_col or not total_col:
            warnings.append("No se encontraron columnas de edad/afiliados en EPS_Edad.")
            detected_edad_work = detected_edad
        else:
            edad_work = eps_edad_df.copy()
            if dept_col:
                edad_work[dept_col] = edad_work[dept_col].astype(str)
                edad_work = edad_work[
                    edad_work[dept_col].str.contains("valle del cauca", case=False, na=False)
                ]
            else:
                warnings.append("No se encontro columna DEPARTAMENTO en EPS_Edad; se usa toda la hoja.")

            edad_work["GrupoEdad"] = edad_work[age_col].map(map_quinquenio_to_group)
            edad_work["Afiliados"] = pd.to_numeric(edad_work[total_col], errors="coerce")
            edad_work = edad_work.dropna(subset=["GrupoEdad", "Afiliados"]).copy()

            if edad_work.empty:
                warnings.append("EPS_Edad no produjo filas validas para distribucion por edad.")
            else:
                grouped = (
                    edad_work.groupby("GrupoEdad", as_index=False)["Afiliados"]
                    .sum(min_count=1)
                    .reset_index(drop=True)
                )
                total_aff_valle = grouped["Afiliados"].sum(min_count=1)

                if (
                    pd.notna(valle_total)
                    and pd.notna(total_aff_valle)
                    and total_aff_valle > 0
                    and valle_total > 0
                ):
                    diff = abs(float(total_aff_valle) - float(valle_total)) / max(float(valle_total), 1.0)
                    if diff > 0.05:
                        warnings.append(
                            "El total de afiliados Valle en EPS_Edad difiere del TOTAL de Comparacion."
                        )

                prev_map = (
                    prev_df.drop_duplicates(subset=["GrupoEdad"])
                    .set_index("GrupoEdad")["Prevalencia"]
                    .to_dict()
                    if not prev_df.empty
                    else {}
                )
                grouped["Prevalencia"] = grouped["GrupoEdad"].map(prev_map)
                missing_prev = grouped[grouped["Prevalencia"].isna()]["GrupoEdad"].tolist()
                if missing_prev:
                    warnings.append(
                        "Falta prevalencia para grupos: " + ", ".join(sorted(set(missing_prev)))
                    )
                    grouped["Prevalencia"] = grouped["Prevalencia"].fillna(0.0)

                grouped["PctPoblacion"] = (
                    grouped["Afiliados"] / total_aff_valle
                    if pd.notna(total_aff_valle) and total_aff_valle > 0
                    else np.nan
                )
                grouped["PacientesEstimados"] = grouped["Prevalencia"] * grouped["Afiliados"]
                total_pacientes_est = grouped["PacientesEstimados"].sum(min_count=1)
                sam_total = total_pacientes_est
                grouped["Ponderacion"] = (
                    grouped["PacientesEstimados"] / total_pacientes_est
                    if pd.notna(total_pacientes_est) and total_pacientes_est > 0
                    else np.nan
                )

                if pd.notna(posibles_valle):
                    grouped["PacientesPorEdad"] = grouped["Ponderacion"] * posibles_valle
                else:
                    grouped["PacientesPorEdad"] = np.nan

                grouped["GrupoEdad"] = pd.Categorical(
                    grouped["GrupoEdad"],
                    categories=AGE_GROUP_ORDER,
                    ordered=True,
                )
                grouped = grouped.sort_values("GrupoEdad").reset_index(drop=True)

                edad_chart_df = grouped[["GrupoEdad", "PacientesPorEdad"]].copy()

                edad_view = grouped.copy()
                total_row_edad = {
                    "GrupoEdad": "TOTAL",
                    "Afiliados": edad_view["Afiliados"].sum(min_count=1),
                    "Prevalencia": np.nan,
                    "PctPoblacion": edad_view["PctPoblacion"].sum(min_count=1),
                    "PacientesEstimados": edad_view["PacientesEstimados"].sum(min_count=1),
                    "Ponderacion": edad_view["Ponderacion"].sum(min_count=1),
                    "PacientesPorEdad": edad_view["PacientesPorEdad"].sum(min_count=1),
                }
                edad_view = pd.concat([edad_view, pd.DataFrame([total_row_edad])], ignore_index=True)

            detected_edad_work = detected_edad

    summary_metrics = {
        "pct_atendido_santander": pct_atendido,
        "atendidos_santander": atendidos_total,
        "posibles_atendidos_valle": posibles_valle,
        "sam_pacientes_valle": sam_total,
        "afiliados_valle_total": valle_total,
        "afiliados_santander_total": sant_total,
    }

    metadata = {
        "status": "warning" if warnings else "ok",
        "method": "legacy",
        "objective_source_column": "posibles_atendidos_valle",
        "comparacion_columns": comp_cols,
        "comparacion_detected": detected_comp,
        "eps_edad_columns": [str(c) for c in eps_edad_df.columns],
        "eps_edad_detected": detected_edad_work,
        "prevalencia_df": prev_df,
    }

    return {
        "summary_metrics": summary_metrics,
        "eps_view": eps_view,
        "edad_view": edad_view,
        "formula_view": _build_formula_view(method="legacy"),
        "edad_chart_df": edad_chart_df,
        "warnings": warnings,
        "metadata": metadata,
    }


def compute_target_population(
    comparacion_df: pd.DataFrame,
    eps_edad_df: pd.DataFrame,
    prevalencia_df_raw: pd.DataFrame,
) -> dict[str, Any]:
    warnings: list[str] = []
    prev_df = parse_prevalencia(prevalencia_df_raw)
    if prev_df.empty:
        warnings.append("No se pudo leer la tabla de prevalencia (GrupoEdad/Prevalencia).")
    if comparacion_df.empty:
        warnings.append("No se pudo leer la hoja Comparacion_Desglose/Comparacion.")
        return _empty_snapshot(warnings, prevalencia_df=prev_df)

    result_desglose = _compute_from_comparacion_desglose(
        comparacion_df=comparacion_df,
        prevalencia_df=prev_df,
        warnings=warnings.copy(),
    )
    if result_desglose is not None:
        return result_desglose

    warnings.append(
        "No se encontro columna 'Pacientes por edad' en Comparacion_Desglose; se usa logica legacy."
    )
    return _compute_target_population_legacy(
        comparacion_df=comparacion_df,
        eps_edad_df=eps_edad_df,
        prev_df=prev_df,
        warnings=warnings,
    )
