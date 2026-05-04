import streamlit as st
from pathlib import Path 

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"

def render_sidebar():
    with st.sidebar:
        logo_path = ASSETS_DIR / "logo_tj.png"
        st.image(str(logo_path), width=100)
        
        st.title("Segmentasi Pelanggan Transjakarta")
        st.markdown("")
        st.page_link("app.py", label="Dashboard", icon="🚌")
        st.page_link("pages/1_Dataset.py", label="Dataset", icon="📁")
        st.page_link("pages/2_Klasterisasi.py", label="Klasterisasi", icon="📊")
        st.page_link("pages/3_Interpretasi.py", label="Interpretasi Hasil", icon="🧩")
        st.page_link("pages/4_Klasifikasi.py", label="Klasifikasi", icon="📌")
        st.page_link("pages/5_Prediksi.py", label="Prediksi", icon="🔎")

        st.divider()

        st.caption("Safira Aulia Rahma")
        st.caption("NIM: 4611422125")