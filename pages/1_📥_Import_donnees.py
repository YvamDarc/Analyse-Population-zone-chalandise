import io
import csv
import time
import pandas as pd
import streamlit as st

st.title("📥 Import des données (une fois)")

# ---------- Reader robuste ----------
@st.cache_data(show_spinner=False)
def read_uploaded_csv_smart(file_bytes: bytes):
    size_mb = round(len(file_bytes) / (1024 * 1024), 2)
    sample = file_bytes[:300_000].decode("utf-8", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        sep = dialect.delimiter
    except Exception:
        sep = ";"

    def _try(enc: str):
        t0 = time.time()
        df = pd.read_csv(
            io.BytesIO(file_bytes),
            sep=sep,
            engine="python",
            quotechar='"',
            encoding=enc,
            on_bad_lines="skip",
        )
        diag = {
            "size_mb": size_mb,
            "sep_used": sep,
            "encoding": enc,
            "seconds": round(time.time() - t0, 2),
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "columns": df.columns.tolist(),
        }
        return df, diag

    try:
        return _try("utf-8")
    except Exception:
        return _try("latin-1")


# ---------- UI ----------
st.caption("Tu peux importer plusieurs fichiers : âges, revenus, logements, entreprises, etc.")
uploaded = st.file_uploader(
    "Dépose tes fichiers CSV (tu peux en sélectionner plusieurs)",
    type=["csv"],
    accept_multiple_files=True
)

if "datasets" not in st.session_state:
    st.session_state["datasets"] = {}

datasets = st.session_state["datasets"]

if uploaded:
    for f in uploaded:
        key = f.name
        with st.spinner(f"Lecture {key}..."):
            df, diag = read_uploaded_csv_smart(f.getvalue())
        datasets[key] = {"df": df, "diag": diag}
        st.success(f"Importé : {key} ({diag['rows']:,} lignes, {diag['size_mb']} Mo)")

st.markdown("## Datasets en mémoire")
if not datasets:
    st.info("Aucun dataset importé pour l’instant.")
else:
    for name, obj in datasets.items():
        st.markdown(f"### {name}")
        st.json(obj["diag"])
        st.dataframe(obj["df"].head(20), use_container_width=True, height=220)

if st.button("🧹 Vider tous les datasets"):
    st.session_state["datasets"] = {}
    st.experimental_rerun()
