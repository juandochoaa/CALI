import numpy as np
import pandas as pd

# Todos los datos los tengo en un excel llamado Cali ANALISIS, varias hojas
Cali_ANALISIS = pd.ExcelFile("C:\\Users\\analistagerencia\\OneDrive - 804014839_INSTITUTO DEL CORAZON DE BUCARAMANGA\\Documentos\\CALI\\data\\raw\\Cali ANALISIS.xlsx")

EPS_Afiliados = pd.read_excel(Cali_ANALISIS, sheet_name="EPS_Afiliados")
UPC = pd.read_excel(Cali_ANALISIS, sheet_name="UPC")
EPS_Edad = pd.read_excel(Cali_ANALISIS, sheet_name="EPS_Edad")
EPS_EEFF = pd.read_excel(Cali_ANALISIS, sheet_name="EPS_EEFF")
IPS_EEFF = pd.read_excel(Cali_ANALISIS, sheet_name="IPS_EEFF")
EPS_Afiliados_Historico = pd.read_excel(Cali_ANALISIS, sheet_name="EPS_Años")
EEFF_Santander = pd.read_excel(Cali_ANALISIS, sheet_name="EEFF_Santander")


import pandas as pd

# Asume que YA existen en memoria estos DataFrames:
# UPC, EPS_EEFF, EPS_Años, EPS_Edad

# =========================
# 0) EPS objetivo (filtro)
# =========================
EPS_OBJ = [
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
    "REGIMEN DE EXCEPCION",
    "SALUD TOTAL EPS",
]

def to_long_year(df, id_vars, value_name):
    year_cols = [c for c in df.columns if str(c).isdigit()]
    out = df.melt(id_vars=id_vars, value_vars=year_cols, var_name="Año", value_name=value_name)
    out["Año"] = out["Año"].astype(int)
    return out

def norm_eps(s):
    return s.astype(str).str.strip()

# ==========================================
# 1) Filtrar desde el principio (por EPS)
# ==========================================
EPS_EEFF = EPS_EEFF.copy()
EPS_Años = EPS_Afiliados_Historico.copy()
EPS_Edad = EPS_Edad.copy()

EPS_EEFF["EPS"] = norm_eps(EPS_EEFF["EPS"])
EPS_Años["EPS"] = norm_eps(EPS_Años["EPS"])
EPS_Edad["EPS"] = norm_eps(EPS_Edad["EPS"])

EPS_EEFF = EPS_EEFF[EPS_EEFF["EPS"].isin(EPS_OBJ)].copy()
EPS_Años = EPS_Años[EPS_Años["EPS"].isin(EPS_OBJ)].copy()
EPS_Edad = EPS_Edad[EPS_Edad["EPS"].isin(EPS_OBJ)].copy()

# ============================================================
# 2) UPC: wide -> long (Regimen, GrupoEdad_UPC, Año, UPC)
# ============================================================
UPC_long = to_long_year(UPC, id_vars=["Regimen", "GrupoEdad"], value_name="UPC")
UPC_long["Regimen"] = UPC_long["Regimen"].astype(str).str.strip()
UPC_long["GrupoEdad"] = UPC_long["GrupoEdad"].astype(str).str.strip()
UPC_long = UPC_long.rename(columns={"GrupoEdad": "GrupoEdad_UPC"})

# ============================================================
# 3) EPS_EEFF: wide -> long y pivot a ancho por cuenta
# ============================================================
EPS_EEFF_long = to_long_year(EPS_EEFF, id_vars=["EPS", "CUENTA"], value_name="Valor")
EPS_EEFF_long["EPS"] = norm_eps(EPS_EEFF_long["EPS"])
EPS_EEFF_long["CUENTA"] = EPS_EEFF_long["CUENTA"].astype(str).str.strip()

EEFF_wide = (
    EPS_EEFF_long
    .pivot_table(index=["EPS", "Año"], columns="CUENTA", values="Valor", aggfunc="sum")
    .reset_index()
)

# ============================================================
# 4) EPS_Años (afiliados): wide -> long
# ============================================================
EPS_Años_long = to_long_year(EPS_Años, id_vars=["EPS"], value_name="Afiliados")
EPS_Años_long["EPS"] = norm_eps(EPS_Años_long["EPS"])

# ============================================================
# 5) EPS_Edad: construir mix EPS-Regimen-GrupoEdad_UPC (sin sexo)
# ============================================================
edad_map = {
    "De 0 a 4 años": "1-4 años",
    "De 05 a 09 años": "5-14 años",
    "De 10 a 14 años": "5-14 años",
    "De 15 a 19 años": "15-18 años",  # conservador
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

eps_edad = EPS_Edad[["EPS", "REGIMEN", "GRUPOEDAD", "TOTAL AFILIADOS"]].copy()
eps_edad = eps_edad.rename(columns={
    "REGIMEN": "Regimen",
    "GRUPOEDAD": "GrupoEdad_EPS",
    "TOTAL AFILIADOS": "Afiliados",
})

eps_edad["EPS"] = norm_eps(eps_edad["EPS"])
eps_edad["Regimen"] = eps_edad["Regimen"].astype(str).str.strip()
eps_edad["GrupoEdad_EPS"] = eps_edad["GrupoEdad_EPS"].astype(str).str.strip()

eps_edad["GrupoEdad_UPC"] = eps_edad["GrupoEdad_EPS"].map(edad_map)

# Consolidar por EPS-Regimen-GrupoEdad_UPC (sumando departamentos)
eps_mix = (
    eps_edad
    .groupby(["EPS", "Regimen", "GrupoEdad_UPC"], as_index=False)["Afiliados"]
    .sum()
)

# Peso dentro de cada EPS (mix 2025)
eps_tot = eps_mix.groupby("EPS", as_index=False)["Afiliados"].sum().rename(columns={"Afiliados": "Afiliados_EPS"})
eps_mix = eps_mix.merge(eps_tot, on="EPS", how="left")
eps_mix["Peso"] = eps_mix["Afiliados"] / eps_mix["Afiliados_EPS"]

# ============================================================
# 6) UPC_mix por EPS y Año (ponderado por mix)
# ============================================================
upc_mix_eps_year = (
    eps_mix
    .merge(UPC_long, on=["Regimen", "GrupoEdad_UPC"], how="left")
    .assign(UPC_pond=lambda d: d["Peso"] * d["UPC"])
    .groupby(["EPS", "Año"], as_index=False)["UPC_pond"].sum()
    .rename(columns={"UPC_pond": "UPC_mix"})
    .sort_values(["EPS", "Año"])
    .reset_index(drop=True)
)

# ============================================================
# 7) Base final: EEFF + Afiliados + UPC_mix (solo EPS objetivo)
# ============================================================
base_eps = (
    EEFF_wide
    .merge(EPS_Años_long, on=["EPS", "Año"], how="left")
    .merge(upc_mix_eps_year, on=["EPS", "Año"], how="left")
    .sort_values(["EPS", "Año"])
    .reset_index(drop=True)
)

# Objetos finales listos:
# - UPC_long
# - EPS_EEFF_long, EEFF_wide
# - EPS_Años_long
# - eps_mix
# - upc_mix_eps_year
# - base_eps

import pandas as pd

df = base_eps.copy()

# ---- Ancla: último año disponible en EEFF ----
anchor_year = int(df["Año"].max())

# ---- Variables base ----
df["Ingresos"] = df["TotalIngresoOperativo"]
df["Costo"] = df["Otroscostospornaturaleza"]
df["GAdmin"] = df["Gastosadministrativos"]
df["Cash"] = df["EfectivooEquivalentes"]
df["Equity"] = df["Totaldepatrimonio"]

# ---- Reservas proxy (provisiones corrientes + no corrientes)
df["ReservasProxy"] = df["Provisionesparaotrospasivosygastos"]

# ---- Ratios core a simular ----
df["LR"] = df["Costo"] / df["Ingresos"]
df["AR"] = df["GAdmin"] / df["Ingresos"]

# ---- OPEX other (reemplazo de SR) con faltantes a 0 ----
opex_cols = [
    "Gastosporbeneficiosdelosempleado",
    "Costosdetransporte",
    "Impuestoycontribuciones",
    "Otrosgastos",
]

for col in opex_cols:
    if col not in df.columns:
        df[col] = 0.0
    df[col] = df[col].fillna(0.0)

df["OPEX_other"] = df[opex_cols].sum(axis=1)

# Evitar NaN por ingresos NaN (si existiera) y división normal
df["OER"] = df["OPEX_other"] / df["Ingresos"]

# ---- OpExCash y CashDays (con efectivo) ----
df["OpExCash"] = df["Costo"] + df["GAdmin"] + df["OPEX_other"]
df["CashDays"] = 365 * (df["Cash"] / df["OpExCash"])

# ---- Reservas ratio ----
df["rho_res"] = df["ReservasProxy"] / df["Ingresos"]

# ---- Crecimiento afiliados ----
df = df.sort_values(["EPS", "Año"]).reset_index(drop=True)
df["g"] = df.groupby("EPS")["Afiliados"].pct_change()

# ---- k_e anclado en anchor_year ----
anchor = df[df["Año"] == anchor_year][
    ["EPS", "Ingresos", "Afiliados", "UPC_mix", "Cash", "Equity", "ReservasProxy"]
].copy()

anchor["RevPerAff"] = anchor["Ingresos"] / anchor["Afiliados"]
anchor["k_e"] = anchor["RevPerAff"] / anchor["UPC_mix"]

baseline = anchor.rename(columns={
    "Afiliados": "Afiliados_0",
    "UPC_mix": "UPC_mix_0",
    "Cash": "Cash_0",
    "Equity": "Equity_0",
    "ReservasProxy": "ReservasProxy_0",
    "Ingresos": "Ingresos_0",
})

# ---- Parámetros (mean/std) 2019..anchor_year ----
hist = df[(df["Año"] >= 2019) & (df["Año"] <= anchor_year)].copy()

params_eps = (
    hist.groupby("EPS")[["g", "LR", "AR", "OER", "rho_res"]]
    .agg(["mean", "std"])
    .reset_index()
)
params_eps.columns = ["EPS"] + [f"{v}_{stat}" for (v, stat) in params_eps.columns[1:]]

baseline, params_eps.head()









import pandas as pd

# ============================================================
# Configuración
# ============================================================
N_SIM = 10_000
HORIZON_END = 2030
CASH_THRESHOLDS = [15, 0]
UPC_OPTIMISM_FACTOR = 1.2
UPC_GROWTH_START_YEAR = 2019
UPC_GROWTH_END_YEAR = 2026

SCENARIOS = {
    "BASE": {"LR_shift": 0.0, "g_shift": 0.0},
    "STRESS_LR": {"LR_shift": 0.05, "g_shift": 0.0},
    "STRESS_MIX": {"LR_shift": 0.05, "g_shift": -0.03},
}

# ============================================================
# 0) Preparar df histórico para estimar drivers
# ============================================================
df = base_eps.copy().sort_values(["EPS", "Año"]).reset_index(drop=True)

anchor_year = int(df["Año"].max())  # ancla al último año con EEFF
years_sim = list(range(anchor_year + 1, HORIZON_END + 1))

df["Ingresos"] = df["TotalIngresoOperativo"]
df["Costo"] = df["Otroscostospornaturaleza"]
df["GAdmin"] = df["Gastosadministrativos"]
df["Cash"] = df["EfectivooEquivalentes"]
df["Equity"] = df["Totaldepatrimonio"]

# Provisiones: si quedaron duplicadas en EEFF, el pivot_table ya las sumó en una sola columna
df["ReservasProxy"] = df["Provisionesparaotrospasivosygastos"]

# OPEX_other (faltantes -> 0; NaN -> 0)
opex_cols = ["Gastosporbeneficiosdelosempleado", "Costosdetransporte", "Impuestoycontribuciones", "Otrosgastos"]
for c in opex_cols:
    if c not in df.columns:
        df[c] = 0.0
    df[c] = df[c].fillna(0.0)

df["OPEX_other"] = df[opex_cols].sum(axis=1)
df["OpExCash"] = df["Costo"] + df["GAdmin"] + df["OPEX_other"]

# Ratios (drivers)
df["LR"] = df["Costo"] / df["Ingresos"]
df["AR"] = df["GAdmin"] / df["Ingresos"]
df["OER"] = df["OPEX_other"] / df["Ingresos"]
df["rho_res"] = df["ReservasProxy"] / df["Ingresos"]

# g de afiliados
df["g"] = df.groupby("EPS")["Afiliados"].pct_change()

# CashDays (referencia)
df["CashDays"] = 365 * (df["Cash"] / df["OpExCash"])

# ============================================================
# 1) FORZAR alpha y beta fijos en 1 (como pediste)
# ============================================================
alpha = 1.0
beta = 1.0

# ============================================================
# 2) baseline (anclado a anchor_year)
# ============================================================
baseline = df[df["Año"] == anchor_year][
    ["EPS", "Afiliados", "Ingresos", "UPC_mix", "Cash", "Equity", "ReservasProxy"]
].copy()

baseline["RevPerAff"] = baseline["Ingresos"] / baseline["Afiliados"]
baseline["k_e"] = baseline["RevPerAff"] / baseline["UPC_mix"]

baseline = baseline.rename(columns={
    "Afiliados": "Afiliados_0",
    "Cash": "Cash_0",
    "Equity": "Equity_0",
    "ReservasProxy": "ReservasProxy_0",
    "Ingresos": "Ingresos_0",
})

# ============================================================
# 3) params_eps (mean/std) 2019..anchor_year
# ============================================================
hist = df[(df["Año"] >= 2019) & (df["Año"] <= anchor_year)].copy()

params_eps = (
    hist.groupby("EPS")[["g", "LR", "AR", "OER", "rho_res"]]
    .agg(["mean", "std"])
    .reset_index()
)
params_eps.columns = ["EPS"] + [f"{v}_{stat}" for (v, stat) in params_eps.columns[1:]]

# ============================================================
# 4) Capital mínimo requerido (fijo) por EPS
# ============================================================
eps_list = baseline["EPS"].unique().tolist()

capmin_map = {eps: 19500 for eps in eps_list}
capmin_map["ASMETSALUD EPS"] = 17800
capmin_map["EMSSANAR EPS"] = 17800

capmin_2025 = pd.DataFrame({"EPS": eps_list})
capmin_2025["CapMinReq"] = capmin_2025["EPS"].map(capmin_map)

# ============================================================
# 5) UPC_mix hasta 2030 (usa tabla; extrapola si faltan años)
# ============================================================
upc = upc_mix_eps_year.copy().sort_values(["EPS", "Año"]).reset_index(drop=True)

def build_upc_projection(upc_df, end_year):
    out_rows = []
    for eps, gdf in upc_df.groupby("EPS"):
        gdf = gdf.sort_values("Año").copy()
        years = gdf["Año"].to_list()
        vals = gdf["UPC_mix"].to_list()

        last_year = int(years[-1])
        last_val = float(vals[-1])

        # growth factor base: prefer 2024->2026 if exists, else last step
        if 2024 in years and 2026 in years:
            v0 = float(gdf.loc[gdf["Año"] == 2024, "UPC_mix"].iloc[0])
            v1 = float(gdf.loc[gdf["Año"] == 2026, "UPC_mix"].iloc[0])
            gf = (v1 / v0) ** (1/2) if (v0 > 0) else 1.0
        elif len(years) >= 2:
            v0 = float(vals[-2])
            v1 = float(vals[-1])
            gf = (v1 / v0) if (v0 > 0) else 1.0
        else:
            gf = 1.0

        # Optimism factor sobre la tasa de crecimiento anual de UPC:
        # g_upc = gf - 1  ->  g_upc_ajustado = g_upc * UPC_OPTIMISM_FACTOR
        g_upc = gf - 1.0
        g_upc_adj = g_upc * UPC_OPTIMISM_FACTOR
        gf = max(0.01, 1.0 + g_upc_adj)

        out_rows.extend(gdf[["EPS", "Año", "UPC_mix"]].to_dict("records"))

        for y in range(last_year + 1, end_year + 1):
            last_val = last_val * gf
            out_rows.append({"EPS": eps, "Año": y, "UPC_mix": last_val})

    out = pd.DataFrame(out_rows).drop_duplicates(["EPS", "Año"], keep="last")
    return out.sort_values(["EPS", "Año"]).reset_index(drop=True)

def build_upc_projection_by_segment(upc_long_df, eps_mix_df, end_year):
    year_candidates = [c for c in upc_long_df.columns if c not in ["Regimen", "GrupoEdad_UPC", "UPC"]]
    if not year_candidates:
        raise ValueError("No se encontro columna de ano en UPC_long.")
    year_col = year_candidates[0]
    seg_hist = upc_long_df[["Regimen", "GrupoEdad_UPC", year_col, "UPC"]].copy()
    seg_hist[year_col] = pd.to_numeric(seg_hist[year_col], errors="coerce")
    seg_hist["UPC"] = pd.to_numeric(seg_hist["UPC"], errors="coerce")
    seg_hist = seg_hist.dropna(subset=[year_col, "UPC"]).copy()
    seg_hist[year_col] = seg_hist[year_col].astype(int)
    seg_hist = seg_hist.sort_values(["Regimen", "GrupoEdad_UPC", year_col]).reset_index(drop=True)

    seg_rows = []
    growth_rows = []

    for (regimen, grupo_edad), gdf in seg_hist.groupby(["Regimen", "GrupoEdad_UPC"], dropna=False):
        gdf = gdf.sort_values(year_col).copy()
        years = sorted(gdf[year_col].unique().tolist())
        if not years:
            continue

        y0 = UPC_GROWTH_START_YEAR if UPC_GROWTH_START_YEAR in years else int(min(years))
        y1 = UPC_GROWTH_END_YEAR if UPC_GROWTH_END_YEAR in years else int(max(years))

        if y1 > y0:
            v0 = float(gdf.loc[gdf[year_col] == y0, "UPC"].iloc[0])
            v1 = float(gdf.loc[gdf[year_col] == y1, "UPC"].iloc[0])
            g_base = (v1 / v0) ** (1.0 / (y1 - y0)) - 1.0 if v0 > 0 else 0.0
        else:
            g_base = 0.0

        g_adj = g_base * UPC_OPTIMISM_FACTOR
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

        hist_rows = gdf[["Regimen", "GrupoEdad_UPC", year_col, "UPC"]].rename(columns={"UPC": "UPC_segmento"})
        seg_rows.extend(hist_rows.to_dict("records"))

        last_year = int(max(years))
        last_val = float(gdf.loc[gdf[year_col] == last_year, "UPC"].iloc[-1])
        for y in range(last_year + 1, end_year + 1):
            last_val = last_val * gf
            seg_rows.append(
                {
                    "Regimen": regimen,
                    "GrupoEdad_UPC": grupo_edad,
                    year_col: y,
                    "UPC_segmento": last_val,
                }
            )

    upc_segment_full = (
        pd.DataFrame(seg_rows)
        .drop_duplicates(["Regimen", "GrupoEdad_UPC", year_col], keep="last")
        .sort_values(["Regimen", "GrupoEdad_UPC", year_col])
        .reset_index(drop=True)
    )

    eps_weights = eps_mix_df[["EPS", "Regimen", "GrupoEdad_UPC", "Peso"]].copy()
    upc_eps_full = eps_weights.merge(
        upc_segment_full,
        on=["Regimen", "GrupoEdad_UPC"],
        how="left",
    )

    missing = int(upc_eps_full["UPC_segmento"].isna().sum())
    if missing > 0:
        print(f"[WARN] Segmentos EPS sin UPC proyectado: {missing} filas.")

    upc_eps_full = upc_eps_full[upc_eps_full["UPC_segmento"].notna()].copy()
    upc_eps_full["UPC_pond"] = upc_eps_full["Peso"] * upc_eps_full["UPC_segmento"]

    upc_mix_full = (
        upc_eps_full.groupby(["EPS", year_col], as_index=False)["UPC_pond"]
        .sum()
        .rename(columns={"UPC_pond": "UPC_mix"})
        .sort_values(["EPS", year_col])
        .reset_index(drop=True)
    )

    growth_df = pd.DataFrame(growth_rows).sort_values(["Regimen", "GrupoEdad_UPC"]).reset_index(drop=True)
    return upc_mix_full, growth_df, year_col


upc_full, upc_growth_segments, upc_year_col = build_upc_projection_by_segment(
    upc_long_df=UPC_long,
    eps_mix_df=eps_mix,
    end_year=HORIZON_END,
)
upc_dict = {
    (row["EPS"], int(row[upc_year_col])): float(row["UPC_mix"])
    for row in upc_full.to_dict("records")
}

# ============================================================
# 6) Inputs finales
# ============================================================
inputs = (
    baseline[["EPS", "Afiliados_0", "Cash_0", "Equity_0", "k_e", "Ingresos_0"]]
    .merge(params_eps, on="EPS", how="left")
    .merge(capmin_2025, on="EPS", how="left")
)

# ============================================================
# 7) Utilidad: normal truncada
# ============================================================
def rnorm_trunc(mean, std, low, high, size):
    if std is None or np.isnan(std) or std == 0:
        x = np.full(size, mean, dtype=float)
    else:
        x = np.random.normal(mean, std, size=size)
    return np.clip(x, low, high)

# ============================================================
# 8) Inversiones: proxy para computables del RI
#    Usamos Activosfinancierosdecortoplazo como "inversiones líquidas".
#    Para proyectarlas simple: mantener ratio inv/Ingresos del anchor_year.
# ============================================================
if "Activosfinancierosdecortoplazo" not in df.columns:
    df["Activosfinancierosdecortoplazo"] = 0.0

inv_ratio_map = {}
for eps in eps_list:
    inv0 = df[(df["EPS"] == eps) & (df["Año"] == anchor_year)]["Activosfinancierosdecortoplazo"]
    ing0 = df[(df["EPS"] == eps) & (df["Año"] == anchor_year)]["Ingresos"]
    if len(inv0) and len(ing0) and float(ing0.iloc[0]) != 0:
        inv_ratio_map[eps] = float(inv0.iloc[0]) / float(ing0.iloc[0])
    else:
        inv_ratio_map[eps] = 0.0

# ============================================================
# 9) Monte Carlo (RI incluye efectivo + inversiones)
# ============================================================
results = []

for scenario_name, shock in SCENARIOS.items():
    LR_shift = shock["LR_shift"]
    g_shift = shock["g_shift"]

    for row in inputs.itertuples(index=False):
        eps = row.EPS

        afiliados = np.full(N_SIM, float(row.Afiliados_0), dtype=float)
        cash = np.full(N_SIM, float(row.Cash_0), dtype=float)
        equity = np.full(N_SIM, float(row.Equity_0), dtype=float)

        k_e = float(row.k_e)
        capmin = float(row.CapMinReq)
        inv_ratio0 = float(inv_ratio_map.get(eps, 0.0))

        g_m, g_s = float(row.g_mean), float(row.g_std)
        LR_m, LR_s = float(row.LR_mean), float(row.LR_std)
        AR_m, AR_s = float(row.AR_mean), float(row.AR_std)
        OER_m, OER_s = float(row.OER_mean), float(row.OER_std)
        rr_m, rr_s = float(row.rho_res_mean), float(row.rho_res_std)

        avg_cash_counts = {thr: np.zeros(N_SIM, dtype=float) for thr in CASH_THRESHOLDS}
        avg_cm_count = np.zeros(N_SIM, dtype=float)
        avg_pa_count = np.zeros(N_SIM, dtype=float)
        avg_ri_count = np.zeros(N_SIM, dtype=float)

        for y in years_sim:
            upc_mix_y = upc_dict.get((eps, y), upc_dict.get((eps, y - 1), 0.0))

            g = rnorm_trunc(g_m + g_shift, g_s, -0.50, 0.50, N_SIM)
            LR = rnorm_trunc(LR_m + LR_shift, LR_s, 0.01, 2.00, N_SIM)
            AR = rnorm_trunc(AR_m, AR_s, 0.00, 1.00, N_SIM)
            OER = rnorm_trunc(OER_m, OER_s, 0.00, 1.00, N_SIM)
            rho_res = rnorm_trunc(rr_m, rr_s, 0.00, 2.00, N_SIM)

            afiliados = afiliados * (1.0 + g)
            ingresos = afiliados * upc_mix_y * k_e

            opex_cash = ingresos * (LR + AR + OER)
            margin = ingresos - opex_cash

            cash = np.maximum(0.0, cash + alpha * margin)
            equity = equity + beta * margin

            reservas = ingresos * rho_res

            # Inversiones computables (proxy): crecen proporcional a ingresos
            inversiones = ingresos * inv_ratio0

            cashdays = 365.0 * np.divide(
                cash,
                opex_cash,
                out=np.full_like(cash, np.nan, dtype=float),
                where=opex_cash != 0,
            )
            cm_ratio = equity / capmin
            pa_req = 0.08 * ingresos * LR
            pa_ratio = np.divide(
                equity,
                pa_req,
                out=np.full_like(equity, np.nan, dtype=float),
                where=pa_req != 0,
            )

            # RI incluye efectivo + inversiones
            ri_ratio = np.divide(
                (cash + inversiones),
                reservas,
                out=np.full_like(cash, np.nan, dtype=float),
                where=reservas != 0,
            )

            for thr in CASH_THRESHOLDS:
                cash_breach = cashdays < thr
                avg_cash_counts[thr] += cash_breach.astype(float)

            cm_breach = cm_ratio < 1.0
            pa_breach = pa_ratio < 1.0
            ri_breach = ri_ratio < 1.0

            avg_cm_count += cm_breach.astype(float)
            avg_pa_count += pa_breach.astype(float)
            avg_ri_count += ri_breach.astype(float)

        n_years = max(len(years_sim), 1)
        row_out = {
            "EPS": eps,
            "Escenario": scenario_name,
            "anchor_year": anchor_year,
            "alpha_fijo": alpha,
            "beta_fijo": beta,
            "P_avg_CM_ratio_lt_1": (avg_cm_count / n_years).mean(),
            "P_avg_PA_ratio_lt_1": (avg_pa_count / n_years).mean(),
            "P_avg_RI_ratio_lt_1": (avg_ri_count / n_years).mean(),
        }
        for thr in CASH_THRESHOLDS:
            row_out[f"P_avg_CashDays_lt_{thr}"] = (avg_cash_counts[thr] / n_years).mean()
        results.append(row_out)

results_df = pd.DataFrame(results).sort_values(["Escenario", "EPS"]).reset_index(drop=True)

def percentile_score_low_is_better(series: pd.Series) -> pd.Series:
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


# ============================================================
# 10) Scoring 0-100 por percentiles (menor probabilidad = mejor)
# ============================================================
def build_rankings(df: pd.DataFrame, prob_prefix: str, score_suffix: str):
    prob_cols = [c for c in df.columns if c.startswith(prob_prefix)]
    scored = df.copy()
    for col in prob_cols:
        metric = col.replace(prob_prefix, "")
        score_col = f"Score_{score_suffix}_{metric}"
        scored[score_col] = scored.groupby("Escenario")[col].transform(
            percentile_score_low_is_better
        )

    score_cols = [c for c in scored.columns if c.startswith(f"Score_{score_suffix}_")]
    score_mean_col = f"Score_Promedio_{score_suffix}"
    rank_col = f"Ranking_Escenario_{score_suffix}"

    scored[score_mean_col] = scored[score_cols].mean(axis=1, skipna=True)
    scored[rank_col] = scored.groupby("Escenario")[score_mean_col].rank(
        method="dense",
        ascending=False,
    ).astype(int)

    ranking_escenario = scored.sort_values(["Escenario", rank_col, "EPS"]).reset_index(drop=True)

    ranking_global = (
        scored.groupby("EPS", as_index=False)[score_mean_col]
        .mean()
        .rename(columns={score_mean_col: f"Score_Promedio_Global_{score_suffix}"})
    )
    ranking_global[f"Ranking_Global_{score_suffix}"] = ranking_global[
        f"Score_Promedio_Global_{score_suffix}"
    ].rank(method="dense", ascending=False).astype(int)
    ranking_global = ranking_global.sort_values(
        [f"Ranking_Global_{score_suffix}", "EPS"]
    ).reset_index(drop=True)

    return scored, ranking_escenario, ranking_global, prob_cols, score_cols, score_mean_col, rank_col


(
    results_scored_avg,
    ranking_escenario_avg,
    ranking_global_avg,
    prob_cols_avg,
    score_cols_avg,
    score_mean_avg,
    rank_avg,
) = build_rankings(results_df, prob_prefix="P_avg_", score_suffix="avg")

print("\nProbabilidades base (PROMEDIO):")
print(results_df.to_string(index=False))

print("\nRanking por escenario (score 0-100) - enfoque PROMEDIO:")
print(
    ranking_escenario_avg[
        ["Escenario", rank_avg, "EPS", score_mean_avg] + score_cols_avg
    ].to_string(index=False)
)

print("\nRanking global de EPS (promedio de escenarios) - enfoque PROMEDIO:")
print(ranking_global_avg.to_string(index=False))


