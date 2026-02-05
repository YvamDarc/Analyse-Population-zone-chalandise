import math
import pandas as pd
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium

st.title("🗺️ Zone A — sélection communes (rayon)")

UA = {"User-Agent": "zone-demographie/1.0"}
GEO_COMMUNES_URL = "https://geo.api.gouv.fr/communes"
GEOCODE_URL = "https://data.geopf.fr/geocodage/search/"

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def geocode_search(q: str, limit: int = 8):
    r = requests.get(GEOCODE_URL, params={"q": q, "limit": limit}, timeout=25, headers=UA)
    r.raise_for_status()
    feats = r.json().get("features", [])
    out = []
    for f in feats:
        props = f.get("properties", {}) or {}
        coords = (f.get("geometry") or {}).get("coordinates", None)
        if not coords:
            continue
        lon, lat = coords[0], coords[1]
        out.append({"label": props.get("label", ""), "lat": float(lat), "lon": float(lon)})
    return out

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def geo_commune_by_latlon(lat: float, lon: float):
    r = requests.get(
        GEO_COMMUNES_URL,
        params={"lat": lat, "lon": lon, "fields": "nom,code,codeDepartement", "format": "json"},
        timeout=25,
        headers=UA,
    )
    r.raise_for_status()
    data = r.json()
    return data[0] if isinstance(data, list) and data else None

@st.cache_data(ttl=7 * 24 * 3600, show_spinner=True)
def geo_communes_dept(code_dept: str) -> pd.DataFrame:
    r = requests.get(
        GEO_COMMUNES_URL,
        params={
            "codeDepartement": code_dept,
            "fields": "nom,code,codeDepartement,codesPostaux,population,centre",
            "format": "geojson",
            "geometry": "centre",
        },
        timeout=60,
        headers=UA,
    )
    r.raise_for_status()
    feats = r.json().get("features", [])
    rows = []
    for f in feats:
        props = f.get("properties", {}) or {}
        coords = (f.get("geometry") or {}).get("coordinates", None)
        if not coords:
            continue
        lon, lat = coords[0], coords[1]
        rows.append(
            {
                "code_insee": str(props.get("code")).zfill(5),
                "nom": props.get("nom"),
                "code_dept": props.get("codeDepartement"),
                "codes_postaux": ", ".join(props.get("codesPostaux") or []),
                "population_geoapi": props.get("population"),
                "lat": float(lat),
                "lon": float(lon),
            }
        )
    return pd.DataFrame(rows)

# ---------------- UI ----------------
q = st.text_input("Adresse / code postal / ville", value="")
radius_km = st.slider("Rayon (km)", 1, 80, 15, 1)

if st.button("🔎 Rechercher"):
    if q.strip():
        st.session_state["zone_a_geo"] = geocode_search(q.strip(), limit=8)
    else:
        st.warning("Entre une ville/adresse.")

results = st.session_state.get("zone_a_geo", [])
if not results:
    st.info("Tape une adresse puis clique **Rechercher**.")
    st.stop()

idx = st.selectbox("Choisir le point", list(range(len(results))), format_func=lambda i: results[i]["label"])
center = results[idx]
st.caption(f"Centre: {center['label']}")

if st.button("📍 Charger communes du rayon"):
    com = geo_commune_by_latlon(center["lat"], center["lon"])
    if not com:
        st.error("Commune introuvable.")
        st.stop()
    df_communes = geo_communes_dept(com["codeDepartement"])
    df_communes["dist_km"] = df_communes.apply(
        lambda r: haversine_km(center["lat"], center["lon"], r["lat"], r["lon"]), axis=1
    )
    in_radius = df_communes[df_communes["dist_km"] <= radius_km].copy()
    in_radius = in_radius.sort_values(["dist_km", "population_geoapi"], ascending=[True, False])
    st.session_state["zone_a_in_radius"] = in_radius

in_radius = st.session_state.get("zone_a_in_radius", None)
if in_radius is None:
    st.info("Clique **Charger communes du rayon**.")
    st.stop()

st.write(f"Communes dans {radius_km} km : **{len(in_radius):,}**")

# sélection
if "zone_a_selected_codes" not in st.session_state:
    st.session_state["zone_a_selected_codes"] = set()

view = in_radius[["code_insee", "nom", "codes_postaux", "code_dept", "population_geoapi", "dist_km"]].copy()
view["ajouter"] = view["code_insee"].isin(st.session_state["zone_a_selected_codes"])

edited = st.data_editor(
    view,
    hide_index=True,
    use_container_width=True,
    column_config={
        "ajouter": st.column_config.CheckboxColumn("Ajouter"),
        "dist_km": st.column_config.NumberColumn("Distance (km)", format="%.1f"),
    },
    disabled=["code_insee", "nom", "codes_postaux", "code_dept", "population_geoapi", "dist_km"],
    key="zone_a_editor",
)

st.session_state["zone_a_selected_codes"] = set(
    edited.loc[edited["ajouter"] == True, "code_insee"].astype(str).tolist()
)

sel = sorted(st.session_state["zone_a_selected_codes"])
st.info(f"Zone A — communes sélectionnées : **{len(sel)}**")

# carte
m = folium.Map(location=[center["lat"], center["lon"]], zoom_start=10, control_scale=True)
folium.Marker([center["lat"], center["lon"]], tooltip="Centre", popup=center["label"]).add_to(m)
folium.Circle([center["lat"], center["lon"]], radius=radius_km * 1000, fill=False).add_to(m)
for _, r in in_radius.head(250).iterrows():
    folium.CircleMarker([r["lat"], r["lon"]], radius=4, tooltip=f"{r['nom']} ({r['code_insee']})").add_to(m)
st_folium(m, width=900, height=420)

# sortie DF zone A
df_zone_a = in_radius[in_radius["code_insee"].isin(sel)].copy()
st.session_state["zone_a"] = df_zone_a

st.markdown("### DataFrame Zone A")
st.dataframe(df_zone_a, use_container_width=True, height=260)

st.download_button(
    "⬇️ Télécharger Zone A (CSV)",
    data=df_zone_a.to_csv(index=False).encode("utf-8"),
    file_name="zone_A_communes.csv",
    mime="text/csv",
)
