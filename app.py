import streamlit as st

st.set_page_config(page_title="Chalandise — multi-pages", layout="wide")

st.title("Chalandise — Multi-pages (Import → Zone A → Zone B)")

# État global minimal
datasets = st.session_state.get("datasets", {})
zone_a = st.session_state.get("zone_a", None)
zone_b = st.session_state.get("zone_b", None)

st.subheader("État du projet")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Datasets importés", len(datasets))
with c2:
    st.metric("Zone A sélectionnée", "Oui" if zone_a is not None and len(zone_a) else "Non")
with c3:
    st.metric("Zone B sélectionnée", "Oui" if zone_b is not None and len(zone_b) else "Non")

st.markdown("---")
st.info(
    "1) Va sur **📥 Import données** (page 1)\n"
    "2) Va sur **🗺️ Zone A** (page 2)\n"
    "3) Va sur **🗺️ Zone B** (page 3, optionnelle)\n"
)
