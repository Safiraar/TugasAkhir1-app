import streamlit as st
from utils.sidebar import render_sidebar, ASSETS_DIR

st.set_page_config(
    page_title="Segmentasi Pelanggan Transjakarta", 
    layout="wide",
    initial_sidebar_state="expanded"
)

render_sidebar()

st.title("Dashboard")

st.write("")
col1, col2, col3 = st.columns([1, 1.2, 1])

with col2: 
    logo_univ = ASSETS_DIR / "logo_univ.png"
    st.image(str(logo_univ), width=180)

st.subheader(
    "OPTIMASI K-MEANS MENGGUNAKAN *PARTICLE SWARM OPTIMIZATION* "
    "PADA SEGMENTASI PELANGGAN TRANSJAKARTA BERBASIS MODEL "
    "*RECENCY, FREQUENCY* DAN *MONETARY* DENGAN *SUPPORT VECTOR MACHINE*"
)


st.write("**Nama:** Safira Aulia Rahma")
st.write("**NIM:** 4611422125")
st.write("**Program Studi:** Teknik Informatika")
st.write("**Fakultas:** Fakultas Matematika dan Ilmu Pengetahuan Alam")
st.write("**Universitas:** Universitas Negeri Semarang")
st.write("**Tahun:** 2026")
st.write("**Dosen Pembimbing** : Endang Sugiharti S.Si., M.Kom.")
