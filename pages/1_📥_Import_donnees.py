# pages/1_📥_Import_donnees.py
import io
import csv
import time
import os
import pandas as pd
import streamlit as st

st.title("📥 Import des données")

# =========================
# CONFIG PERSISTENCE
# =========================
SAVE_DIR = "/tmp/zone_chalandise_datasets"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# CSV reader robuste
# =========================
@st.cache_data(show_spinner=False)
def read_uploaded_csv_smart(file_bytes: bytes):
    sample = file_bytes[:300_000].decode("utf-8", errors="replace")
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"]).delimiter
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


# =========================
# INIT SESSION STATE
# =========================
if "datasets_files" not in st.session_state:
    st.session_state["datasets_files"] = {}   # {name: bytes}
if "datasets_diag" not in st.session_state:
    st.session_state["datasets_diag"] = {}    # {name: diag}


# =========================
# 🔁 AUTO-RELOAD depuis /tmp
# =========================
if not st.session_state["datasets_files"]:
    for fname in os.listdir(SAVE_DIR):
        fpath = os.path.join(SAVE_DIR, fname)
        try:
            with open(fpath, "rb") as f:
                b = f.read()
            df, diag = read_uploaded_csv_smart(b)
            st.session_state["datasets_files"][fname] = b
            st.session_state["datasets_diag"][fname] = diag
        except Exception:
            pass


# =========================
# UPLOAD UI
# =========================
uploaded = st.file_uploader(
    "Import CSV (multi-fichiers)",
    type=["csv"],
    accept_multiple_files=True
)

if uploaded:
    for f in uploaded:
        b = f.getvalue()

        # mémoire session
        st.session_state["datasets_files"][f.name] = b

        # persistance locale
        with open(os.path.join(SAVE_DIR, f.name), "wb") as out:
            out.write(b)

        # diagnostic
        df, diag = read_uploaded_csv_smart(b)
        st.session_state["datasets_diag"][f.name] = diag

        st.success(f"Importé : {f.name} ({diag['rows']:,} lignes)")


# =========================
# AFFICHAGE
# =========================
st.markdown("## 📚 Datasets disponibles")

if not st.session_state["datasets_files"]:
    st.info("Aucun dataset importé.")
else:
    for name, diag in st.session_state["datasets_diag"].items():
        st.write(
            f"**{name}** — "
            f"{diag['rows']:,} lignes | "
            f"sep={diag['sep_used']} | "
            f"enc={diag['encoding']}"
        )

# =========================
# OUTILS
# =========================
with st.expander("🧹 Maintenance"):
    if st.button("Supprimer tous les datasets"):
        st.session_state["datasets_files"].clear()
        st.session_state["datasets_diag"].clear()
        for f in os.listdir(SAVE_DIR):
            try:
                os.remove(os.path.join(SAVE_DIR, f))
            except Exception:
                pass
        st.experimental_rerun()
