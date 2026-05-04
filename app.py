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

st.write("")
st.write("")

kiri, kanan = st.columns([1,1], gap="medium")

with kiri:
    with st.container(border=True):
        st.subheader("Rumusan Masalah")
        st.markdown(
            """
            1. Bagaimana penerapan algoritma K-Means dengan optimasi *Particle 
            Swarm Optimization* (PSO) dalam melakukan segmentasi pelanggan Transjakarta 
            dengan model *Recency, Frequency,* dan *Monetary* (RFM) yang selanjutnya diikuti 
            dengan proses klasifikasi menggunakan *Support Vector Machine* (SVM)? 
            2. Berapa akurasi, presisi, *recall*, dan *F1-Score* yang diperoleh algoritma 
            *Support Vector Machine* (SVM)?
             """
        )

with kanan: 
    with st.container(border=True):
        st.subheader("Tujuan Penelitian")
        st.markdown(
            """
            1. Menerapkan algoritma K-Means dengan optimasi *Particle Swarm Optimization* 
            (PSO) dalam melakukan segmentasi pelanggan Transjakarta dengan model 
            *Recency, Frequency,* dan *Monetary* (RFM) yang selanjutnya diikuti dengan 
            proses klasifikasi menggunakan *Support Vector Machine* (SVM).  
            2. Mendapatkan hasil akurasi, presisi, *recall*, dan *F1-Score* yang 
            diperolah algoritma *Support Vector Machine* (SVM).
            """
        )

st.write("")
with st.container(border=True):
    st.subheader("Alur Penelitian")
    alur = ASSETS_DIR / "alur_penelitian.png"
    st.image(str(alur), use_container_width=True)

st.info(
    "Dosen Pembimbing : Endang Sugiharti, S.Si., M.Kom.  \n"
    "Dosen Penguji    : Ir. Much Aziz Muslim, S.Kom., M.Kom., Ph.D."
)
