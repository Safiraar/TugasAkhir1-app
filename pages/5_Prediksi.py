import streamlit as st
import pandas as pd
import joblib

from utils.sidebar import render_sidebar
from utils.interpretasi import interpretasi

st.set_page_config(
    page_title="Segmentasi Pelanggan Transjakarta", 
    layout="wide",
    initial_sidebar_state="expanded"
)

render_sidebar()



# Memuat model dan scaler
scaler_svm = joblib.load("model/scaler_svm.pkl")
svm = joblib.load("model/svm_model.pkl")

# Inisialisasi session state
if "hasil_prediksi" not in st.session_state:
    st.session_state.hasil_prediksi = None
if "recency" not in st.session_state:
    st.session_state.recency = 0
if "frequency" not in st.session_state:
    st.session_state.frequency = 0
if "monetary" not in st.session_state:
    st.session_state.monetary = 0

# Fungsi reset — langsung set nilai ke 0
def mulai_ulang():
    st.session_state.hasil_prediksi = None
    st.session_state.recency   = 0
    st.session_state.frequency = 0
    st.session_state.monetary  = 0


st.title("Prediksi Klaster Pelanggan")

st.markdown("""
            Halaman ini digunakan untuk **memprediksi klaster pelanggan** dari data 
            seorang pelanggan Transjakarta berdasarkan model RFM. 
            Masukan data pelanggan pada kolom di bawah ini, lalu tekan tombol **prediksi**
            untuk mengetahui klaster dari pelanggan tersebut
            """)

#Input User
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Recency (hari)**")
    st.caption("Jumlah hari sejak transaksi terakhir pelanggan")
    recency = st.number_input("Recency (hari)", min_value=0, key="recency", label_visibility="collapsed")

with col2:
    st.markdown("**Frequency (transaksi)**")
    st.caption("Total jumlah transaksi yang pernah dilakukan")
    frequency = st.number_input("Frequency (transaksi)", min_value=0, key="frequency", label_visibility="collapsed")

with col3:
    st.markdown("**Monetary (Rp)**")
    st.caption("Total keseluruhan uang yang telah dikeluarkan (dalam Rupiah)")
    monetary = st.number_input("Monetary (Rp)", min_value=0, key="monetary", label_visibility="collapsed")

st.divider()

# # Input user — key sama dengan nama session_state
# recency   = st.number_input("Recency (hari)",        min_value=0, key="recency")
# frequency = st.number_input("Frequency (transaksi)", min_value=0, key="frequency")
# monetary  = st.number_input("Monetary (Rp)",         min_value=0, key="monetary")

# Tombol prediksi
if st.session_state.hasil_prediksi is None:
    if st.button("Prediksi"):
        if frequency == 0:
            st.error("⚠️ Frequency tidak boleh 0, minimal 1")
            st.stop()

        input_df = pd.DataFrame(
            [[recency, frequency, monetary]],
            columns=['Recency', 'Frequency', 'Monetary']
        )

        input_scaled = scaler_svm.transform(input_df)
        hasil = svm.predict(input_scaled)
        klaster = hasil[0] + 1

        st.session_state.hasil_prediksi = klaster
        st.rerun()

# Tampilkan hasil + tombol mulai ulang
if st.session_state.hasil_prediksi is not None:
    klaster = st.session_state.hasil_prediksi
    info = interpretasi[klaster]
    
    st.success(f"Pelanggan ini termasuk kedalam **Klaster {klaster} - {info['tipe']}**")
    
    st.subheader("📊 Informasi Klaster")
    st.markdown(f"**Tipe:** {info['tipe']}")
    st.markdown(f"**Deskripsi:** {info['deskripsi']}")
    
    st.button("🔄 Mulai Ulang", on_click=mulai_ulang)