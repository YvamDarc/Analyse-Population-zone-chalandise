# pages/1_📥_Import_donnees.py
import io, csv, time
import pandas as pd
import streamlit as st

st.title("📥 Import des données")

@st.cache_data(show_spinner=False)
def read_uploaded_csv_smart(file_bytes: bytes):
    sample = file_bytes[:300_000].decode("utf-8", errors="replace")
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=[",",";","\t","|"]).delimiter
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

# ✅ état
if "datasets_files" not in st.session_state:
    st.session_state["datasets_files"] = {}  # {name: bytes}
if "datasets_diag" not in st.session_state:
    st.session_state["datasets_diag"] = {}   # {name: diag}

uploaded = st.file_uploader("Import CSV (multi)", type=["csv"], accept_multiple_files=True)

if uploaded:
    for f in uploaded:
        b = f.getvalue()
        st.session_state["datasets_files"][f.name] = b
        # on calcule diag / aperçu via cache (rapide à la relance)
        df, diag = read_uploaded_csv_smart(b)
        st.session_state["datasets_diag"][f.name] = diag
        st.success(f"Importé: {f.name} ({diag['rows']:,} lignes)")

st.markdown("## Datasets disponibles")
if not st.session_state["datasets_files"]:
    st.info("Aucun dataset importé.")
else:
    for name, diag in st.session_state["datasets_diag"].items():
        st.write(f"**{name}** — {diag['rows']:,} lignes, sep={diag['sep_used']} enc={diag['encoding']}")
