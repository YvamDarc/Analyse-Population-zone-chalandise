import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Analyse & Comparaison", layout="wide")
st.title("📊 Analyse & comparaison — Zone A vs Zone B")

# =========================
# Helpers (âge / buckets)
# =========================
def sort_years(vals):
    def k(v):
        s = str(v)
        return int(s) if s.isdigit() else s
    return sorted([str(v) for v in vals], key=k)

def safe_int_series(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce").fillna(0.0)
    return out.round().astype(int)

def parse_age_macro(age_code: str):
    """Supporte Y0T4, Y15T29, Y_LT15, Y_GE80"""
    if age_code is None:
        return None, None
    s = str(age_code).strip()

    m = re.fullmatch(r"Y(\d{1,3})T(\d{1,3})", s)
    if m:
        return float(m.group(1)), float(m.group(2))

    m = re.fullmatch(r"Y_LT(\d{1,3})", s)
    if m:
        x = float(m.group(1))
        return 0.0, x - 1.0

    m = re.fullmatch(r"Y_GE(\d{1,3})", s)
    if m:
        x = float(m.group(1))
        return x, None

    return None, None

def midpoint(a, b):
    if a is None and b is None:
        return None
    if a is not None and b is not None:
        return (a + b) / 2.0
    if a is not None and b is None:
        return a + 2.5
    return None

def bucket_from_mid(mid):
    if mid is None or (isinstance(mid, float) and np.isnan(mid)):
        return "Inconnu"
    if mid < 15:
        return "0-14"
    if mid < 30:
        return "15-29"
    if mid < 45:
        return "30-44"
    if mid < 60:
        return "45-59"
    if mid < 75:
        return "60-74"
    return "75+"

def detect_total_age_value(age_values):
    """Cherche une modalité AGE = total population si elle existe"""
    priorities = ["Y_TOT", "Y_TOTAL", "TOTAL", "ALL", "Y_ALL", "Y_T", "_T", "T"]
    sset = set(age_values)
    for p in priorities:
        if p in sset:
            return p
    for v in age_values:
        if "TOT" in str(v).upper() or "TOTAL" in str(v).upper():
            return v
    return None

def pick_age_base(age_values):
    """
    Prend une base non-recouvrante si possible :
    - priorité aux classes YxTy (souvent 5 ans)
    - sinon prend des macro-tranches (Y_LT.., Y..T.. larges, Y_GE..)
    """
    vals = [str(v) for v in age_values]
    yxtys = [v for v in vals if re.fullmatch(r"Y\d{1,3}T\d{1,3}", v)]
    if len(yxtys) >= 10:
        return yxtys

    # fallback macro “larges”
    parsed = []
    for v in vals:
        a, b = parse_age_macro(v)
        if a is None and b is None:
            continue
        parsed.append((v, a, b))

    if not parsed:
        return []

    chosen = []
    for v, a, b in parsed:
        if v.startswith("Y_LT") or v.startswith("Y_GE"):
            chosen.append(v)
        elif b is not None and (b - a) >= 10:
            chosen.append(v)

    # si trop peu, on garde tout ce qui est parseable
    if len(chosen) < 6:
        chosen = [v for v, _, _ in parsed]

    return sorted(list(set(chosen)))

def filter_base_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    """Filtre recensement : communes + POP + total sexe (si colonnes présentes)"""
    d = df.copy()

    # colonnes attendues
    needed = [c for c in ["AGE","GEO","GEO_OBJECT","RP_MEASURE","SEX","TIME_PERIOD","OBS_VALUE"] if c in d.columns]
    d = d[needed].copy()

    if "GEO" not in d.columns or "AGE" not in d.columns or "TIME_PERIOD" not in d.columns or "OBS_VALUE" not in d.columns:
        return pd.DataFrame()

    d["GEO"] = d["GEO"].astype(str).str.zfill(5)
    d["TIME_PERIOD"] = d["TIME_PERIOD"].astype(str)
    d["AGE"] = d["AGE"].astype(str)

    if "GEO_OBJECT" in d.columns:
        d = d[d["GEO_OBJECT"].astype(str).str.upper().eq("COM")].copy()

    if "RP_MEASURE" in d.columns:
        d = d[d["RP_MEASURE"].astype(str).str.upper().eq("POP")].copy()

    if "SEX" in d.columns:
        # typique : "_T"
        d = d[d["SEX"].astype(str).isin(["_T","T","TOTAL"])].copy()

    d["OBS_VALUE"] = safe_int_series(d["OBS_VALUE"])
    return d

def zone_total_population_by_year(df_zone: pd.DataFrame, total_age_value, age_base_for_sum):
    """Total zone par année : priorité AGE total sinon somme base non recouvrante"""
    d = df_zone.copy()
    if total_age_value is not None:
        dd = d[d["AGE"] == str(total_age_value)]
        if not dd.empty:
            s = dd.groupby("TIME_PERIOD")["OBS_VALUE"].sum()
            s = s.loc[sort_years(s.index.tolist())]
            return s.astype(int)

    if age_base_for_sum:
        dd = d[d["AGE"].isin(age_base_for_sum)]
        s = dd.groupby("TIME_PERIOD")["OBS_VALUE"].sum()
        s = s.loc[sort_years(s.index.tolist())]
        return s.astype(int)

    # dernier recours
    s = d.groupby("TIME_PERIOD")["OBS_VALUE"].sum()
    s = s.loc[sort_years(s.index.tolist())]
    return s.astype(int)

def zone_age_buckets_by_year(df_zone: pd.DataFrame, age_base):
    """Aire empilée par buckets (0-14, 15-29, ...)"""
    if not age_base:
        return pd.DataFrame()

    d = df_zone[df_zone["AGE"].isin(age_base)].copy()
    if d.empty:
        return pd.DataFrame()

    # AGE -> bucket
    map_bucket = {}
    for a in d["AGE"].unique().tolist():
        amin, amax = parse_age_macro(a)
        mid = midpoint(amin, amax)
        map_bucket[a] = bucket_from_mid(mid)

    d["bucket"] = d["AGE"].map(map_bucket).fillna("Inconnu")
    g = d.groupby(["TIME_PERIOD","bucket"])["OBS_VALUE"].sum().reset_index()
    pivot = g.pivot(index="TIME_PERIOD", columns="bucket", values="OBS_VALUE").fillna(0).astype(int)
    pivot = pivot.loc[sort_years(pivot.index.tolist())]

    bucket_order = ["0-14","15-29","30-44","45-59","60-74","75+","Inconnu"]
    cols = [c for c in bucket_order if c in pivot.columns] + [c for c in pivot.columns if c not in bucket_order]
    return pivot[cols]


# =========================
# Récupération état global
# =========================
datasets = st.session_state.get("datasets", {})
zone_a = st.session_state.get("zone_a", None)
zone_b = st.session_state.get("zone_b", None)

if not datasets:
    st.error("Aucun dataset importé. Va sur la page **📥 Import données**.")
    st.stop()

if zone_a is None or len(zone_a) == 0:
    st.error("Zone A non définie. Va sur la page **🗺️ Zone A**.")
    st.stop()

# Zone B optionnelle
use_zone_b = (zone_b is not None and len(zone_b) > 0)
codes_a = zone_a["code_insee"].astype(str).tolist()
codes_b = zone_b["code_insee"].astype(str).tolist() if use_zone_b else []

st.caption(f"Zone A: {len(codes_a)} communes" + (f" | Zone B: {len(codes_b)} communes" if use_zone_b else " | Zone B: non définie"))

# =========================
# Sélection des datasets
# =========================
def guess_age_dataset_name(datasets_dict):
    for name, obj in datasets_dict.items():
        df = obj["df"]
        cols = set([c.upper() for c in df.columns])
        if {"AGE","GEO","TIME_PERIOD","OBS_VALUE"}.issubset(cols):
            return name
    return None

def guess_revenue_dataset_name(datasets_dict):
    # heuristique grossière : présence de "revenu" / "median" / "nivvie"
    for name, obj in datasets_dict.items():
        cols = " ".join([c.lower() for c in obj["df"].columns])
        if ("revenu" in cols) or ("niv" in cols and "vie" in cols) or ("median" in cols and "uc" in cols):
            return name
    return None

def guess_comp_dataset_name(datasets_dict):
    for name, obj in datasets_dict.items():
        cols = " ".join([c.lower() for c in obj["df"].columns])
        if ("naf" in cols) or ("ape" in cols) or ("concurr" in cols) or ("etabl" in cols) or ("siret" in cols):
            return name
    return None

age_guess = guess_age_dataset_name(datasets)
rev_guess = guess_revenue_dataset_name(datasets)
comp_guess = guess_comp_dataset_name(datasets)

names = list(datasets.keys())

with st.sidebar:
    st.header("Données utilisées")
    age_ds_name = st.selectbox("Dataset Population/Ages (INSEE)", options=names, index=(names.index(age_guess) if age_guess in names else 0))
    rev_ds_name = st.selectbox("Dataset Revenus (optionnel)", options=["(Aucun)"] + names, index=(1 + names.index(rev_guess) if rev_guess in names else 0))
    comp_ds_name = st.selectbox("Dataset Concurrence (optionnel)", options=["(Aucun)"] + names, index=(1 + names.index(comp_guess) if comp_guess in names else 0))

    st.divider()
    st.header("Scoring (optionnel)")
    w_age = st.slider("Poids âge", 0, 70, 40, 5)
    w_rev = st.slider("Poids revenus", 0, 70, 25, 5)
    w_acc = st.slider("Poids accessibilité (proxy densité)", 0, 70, 20, 5)
    w_comp = st.slider("Poids concurrence", 0, 70, 15, 5)

# normaliser poids
w_sum = w_age + w_rev + w_acc + w_comp
if w_sum == 0:
    w_age = 40; w_rev = 25; w_acc = 20; w_comp = 15
    w_sum = 100

W = {"age": w_age / w_sum, "rev": w_rev / w_sum, "acc": w_acc / w_sum, "comp": w_comp / w_sum}


# =========================
# 1) Population totale par année — A vs B
# =========================
st.markdown("## 1) Population totale (zone) — par année")

df_age_raw = datasets[age_ds_name]["df"]
df_age = filter_base_dimensions(df_age_raw)
if df_age.empty:
    st.error("Le dataset sélectionné ne ressemble pas à un fichier INSEE AGE/GEO/TIME_PERIOD/OBS_VALUE exploitable.")
    st.stop()

dfA = df_age[df_age["GEO"].isin(codes_a)].copy()
dfB = df_age[df_age["GEO"].isin(codes_b)].copy() if use_zone_b else pd.DataFrame(columns=dfA.columns)

age_values_A = sorted(dfA["AGE"].unique().tolist())
total_age_auto = detect_total_age_value(age_values_A)
age_base_auto = pick_age_base(age_values_A)

c1, c2 = st.columns([0.6, 0.4], gap="large")
with c2:
    st.caption("Réglages anti-doublons")
    total_age_choice = st.selectbox("AGE total (Zone A)", options=["(Auto)"] + age_values_A, index=0)
    age_base_mode = st.radio("Base tranches", ["Auto", "Manuel"], index=0, horizontal=True)
    if age_base_mode == "Manuel":
        age_base_manual = st.multiselect("Modalités AGE pour tranches", options=age_values_A, default=age_base_auto[:min(30, len(age_base_auto))])
        age_base = age_base_manual
    else:
        age_base = age_base_auto

total_age_value = total_age_auto if total_age_choice == "(Auto)" else total_age_choice

popA = zone_total_population_by_year(dfA, total_age_value, age_base)
pop_df = pd.DataFrame({"annee": popA.index, "Zone A": popA.values})

if use_zone_b:
    # pour B, on applique la même règle (AGE total choisi) — ça reste cohérent
    popB = zone_total_population_by_year(dfB, total_age_value, age_base)
    pop_df["Zone B"] = popB.reindex(popA.index).fillna(0).astype(int).values

pop_df = pop_df.sort_values("annee")

with c1:
    st.dataframe(pop_df, use_container_width=True, height=260)
    st.line_chart(pop_df.set_index("annee"), use_container_width=True)


# =========================
# 2) Tranches d’âge — A vs B
# =========================
st.markdown("## 2) Tranches d’âge (zone) — évolution")

bucketA = zone_age_buckets_by_year(dfA, age_base)

if bucketA.empty:
    st.warning(
        "Impossible de construire des tranches d’âge avec la base actuelle.\n"
        "👉 Passe en **Base tranches = Manuel** et sélectionne les modalités AGE qui sont vraiment des tranches."
    )
    st.stop()

st.subheader("Zone A")
st.dataframe(bucketA.reset_index().rename(columns={"TIME_PERIOD": "annee"}), use_container_width=True, height=240)
st.area_chart(bucketA, use_container_width=True)

if use_zone_b:
    bucketB = zone_age_buckets_by_year(dfB, age_base)
    st.subheader("Zone B")
    st.dataframe(bucketB.reset_index().rename(columns={"TIME_PERIOD": "annee"}), use_container_width=True, height=240)
    st.area_chart(bucketB, use_container_width=True)
else:
    bucketB = None
    st.info("Zone B non définie : la comparaison tranches d’âge est affichée uniquement pour Zone A.")


# =========================
# 3) Synthèse comparée (dernière année) + indices
# =========================
st.markdown("## 3) Synthèse comparée (dernière année) — indices base 100")

latest_year = pop_df["annee"].astype(str).max()
st.caption(f"Dernière année utilisée : {latest_year}")

def latest_bucket_share(bucket_pivot: pd.DataFrame) -> pd.Series:
    if bucket_pivot is None or bucket_pivot.empty:
        return pd.Series(dtype=float)
    yr = str(latest_year)
    if yr not in bucket_pivot.index.astype(str):
        return pd.Series(dtype=float)
    row = bucket_pivot.loc[yr].astype(float)
    tot = float(row.sum()) if row.sum() != 0 else 1.0
    return (row / tot) * 100.0

shareA = latest_bucket_share(bucketA).rename("Zone A (%)")
shareB = latest_bucket_share(bucketB).rename("Zone B (%)") if use_zone_b else pd.Series(dtype=float)

synth = pd.concat([shareA, shareB], axis=1)
st.dataframe(synth, use_container_width=True)

# indices âge : on prend "60-74 + 75+" comme proxy audition (modulable)
def age_index_from_shares(shares: pd.Series) -> float:
    if shares.empty:
        return 100.0
    return float(shares.get("60-74", 0.0) + shares.get("75+", 0.0))

age_idx_A = age_index_from_shares(shareA)
age_idx_B = age_index_from_shares(shareB) if use_zone_b else np.nan

# =========================
# 4) Revenus (optionnel)
# =========================
rev_idx_A = 100.0
rev_idx_B = np.nan

if rev_ds_name != "(Aucun)":
    df_rev = datasets[rev_ds_name]["df"].copy()
    st.markdown("### Revenus (optionnel) — mapping colonnes")

    # mapping colonne code insee + valeur revenu
    cols = df_rev.columns.tolist()
    c1, c2, c3 = st.columns(3)
    with c1:
        rev_col_geo = st.selectbox("Colonne code INSEE", options=cols, index=0)
    with c2:
        rev_col_value = st.selectbox("Colonne revenu/niveau de vie", options=cols, index=min(1, len(cols)-1))
    with c3:
        rev_agg = st.selectbox("Agrégation zone", options=["moyenne", "médiane"], index=1)

    df_rev["code_insee"] = df_rev[rev_col_geo].astype(str).str.zfill(5)
    df_rev["value"] = pd.to_numeric(df_rev[rev_col_value], errors="coerce")

    def zone_rev_index(codes, reference_value):
        z = df_rev[df_rev["code_insee"].isin(codes)]["value"].dropna()
        if z.empty or reference_value is None or np.isnan(reference_value) or reference_value == 0:
            return np.nan, np.nan
        val = float(z.median()) if rev_agg == "médiane" else float(z.mean())
        idx = 100.0 * (val / reference_value)
        return val, idx

    # référence = Zone A (défendable et simple)
    zA = df_rev[df_rev["code_insee"].isin(codes_a)]["value"].dropna()
    refA = float(zA.median()) if (not zA.empty and rev_agg == "médiane") else (float(zA.mean()) if not zA.empty else np.nan)

    rev_val_A, rev_idx_A = zone_rev_index(codes_a, refA)
    rev_val_B, rev_idx_B = zone_rev_index(codes_b, refA) if use_zone_b else (np.nan, np.nan)

    st.write(f"Revenu Zone A (référence 100) : {rev_val_A:.2f} → indice {rev_idx_A:.1f}" if not np.isnan(rev_val_A) else "Revenus Zone A non calculables")
    if use_zone_b:
        st.write(f"Revenu Zone B : {rev_val_B:.2f} → indice {rev_idx_B:.1f}" if not np.isnan(rev_val_B) else "Revenus Zone B non calculables")


# =========================
# 5) Concurrence (optionnel)
# =========================
comp_idx_A = 100.0
comp_idx_B = np.nan

if comp_ds_name != "(Aucun)":
    df_comp = datasets[comp_ds_name]["df"].copy()
    st.markdown("### Concurrence (optionnel) — mapping colonnes")

    cols = df_comp.columns.tolist()
    c1, c2, c3 = st.columns(3)
    with c1:
        comp_col_geo = st.selectbox("Colonne code INSEE (concurrence)", options=cols, index=0)
    with c2:
        comp_col_count = st.selectbox("Colonne nb concurrents", options=cols, index=min(1, len(cols)-1))
    with c3:
        comp_mode = st.selectbox("Score concurrence", options=["moins = mieux", "plus = mieux"], index=0)

    df_comp["code_insee"] = df_comp[comp_col_geo].astype(str).str.zfill(5)
    df_comp["count"] = pd.to_numeric(df_comp[comp_col_count], errors="coerce").fillna(0)

    def zone_comp_metric(codes):
        return float(df_comp[df_comp["code_insee"].isin(codes)]["count"].sum())

    compA = zone_comp_metric(codes_a)
    compB = zone_comp_metric(codes_b) if use_zone_b else np.nan

    # référence = Zone A
    if compA == 0:
        comp_idx_A = 100.0
        comp_idx_B = np.nan
    else:
        if comp_mode == "moins = mieux":
            # moins de concurrence => indice > 100 si B a moins de concurrence
            comp_idx_A = 100.0
            comp_idx_B = 100.0 * (compA / compB) if (use_zone_b and compB and compB != 0) else np.nan
        else:
            comp_idx_A = 100.0
            comp_idx_B = 100.0 * (compB / compA) if use_zone_b else np.nan

    st.write(f"Concurrents Zone A : {compA:.0f} (référence 100)")
    if use_zone_b:
        st.write(f"Concurrents Zone B : {compB:.0f} → indice {comp_idx_B:.1f}" if not np.isnan(compB) else "Concurrents Zone B non calculables")


# =========================
# 6) Scoring (simple, défendable)
# =========================
st.markdown("## 4) Scoring synthétique (comparaison)")

# Accessibilité proxy: ici on prend densité proxy = population / nb communes (faute de surface)
# (quand tu auras surfaces/isochrones, on remplacera)
acc_idx_A = 100.0
acc_idx_B = np.nan

# proxy “accessibilité” : population totale dernière année / nb communes
def acc_proxy(pop_series: pd.Series, n_communes: int) -> float:
    if pop_series.empty or n_communes == 0:
        return np.nan
    last = int(pop_series.loc[str(latest_year)]) if str(latest_year) in pop_series.index.astype(str) else int(pop_series.iloc[-1])
    return last / n_communes

accA = acc_proxy(popA, len(codes_a))
accB = acc_proxy(popB, len(codes_b)) if use_zone_b else np.nan
acc_idx_A = 100.0
acc_idx_B = 100.0 * (accB / accA) if (use_zone_b and accA and not np.isnan(accB)) else np.nan

# Indice âge : référence Zone A = 100
age_idx_ref_A = 100.0
age_idx_ref_B = 100.0 * (age_idx_B / age_idx_A) if (use_zone_b and age_idx_A and not np.isnan(age_idx_B)) else np.nan

# Revenus/Concurrence déjà base ZoneA=100 via indices calculés (rev_idx_A=100 par design)
rev_idx_ref_A = 100.0
rev_idx_ref_B = rev_idx_B

comp_idx_ref_A = 100.0
comp_idx_ref_B = comp_idx_B

def score(idx_age, idx_rev, idx_acc, idx_comp):
    return (W["age"]*idx_age + W["rev"]*idx_rev + W["acc"]*idx_acc + W["comp"]*idx_comp)

scoreA = score(100.0, 100.0, 100.0, 100.0)
scoreB = score(age_idx_ref_B, rev_idx_ref_B, acc_idx_B, comp_idx_ref_B) if use_zone_b else np.nan

score_table = pd.DataFrame([
    {"Zone": "Zone A", "Indice âge": 100.0, "Indice revenu": 100.0, "Indice accessibilité": 100.0, "Indice concurrence": 100.0, "Score": scoreA},
] + ([
    {"Zone": "Zone B", "Indice âge": age_idx_ref_B, "Indice revenu": rev_idx_ref_B, "Indice accessibilité": acc_idx_B, "Indice concurrence": comp_idx_ref_B, "Score": scoreB},
] if use_zone_b else []))

st.dataframe(score_table, use_container_width=True)

if use_zone_b:
    # Barres empilées (contributions)
    contrib = pd.DataFrame([
        {"Zone":"Zone A", "Âge": W["age"]*100, "Revenu": W["rev"]*100, "Accessibilité": W["acc"]*100, "Concurrence": W["comp"]*100},
        {"Zone":"Zone B",
         "Âge": W["age"]*age_idx_ref_B,
         "Revenu": W["rev"]*rev_idx_ref_B if not np.isnan(rev_idx_ref_B) else 0,
         "Accessibilité": W["acc"]*acc_idx_B if not np.isnan(acc_idx_B) else 0,
         "Concurrence": W["comp"]*comp_idx_ref_B if not np.isnan(comp_idx_ref_B) else 0},
    ]).set_index("Zone")

    st.caption("Lecture : Zone A est la référence (=100). Les contributions Zone B montrent ce qui tire le score.")
    st.bar_chart(contrib, use_container_width=True)

st.info(
    "Notes :\n"
    "- La Zone A est utilisée comme **référence 100** (simple et défendable).\n"
    "- L’accessibilité ici est un **proxy** (pop/nb communes). Quand tu ajoutes surface/isochrones, on remplacera.\n"
    "- Les modules Revenus/Concurrence sont optionnels et nécessitent un mapping de colonnes."
)
