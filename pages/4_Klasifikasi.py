import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

from utils.sidebar import render_sidebar

st.set_page_config(
    page_title="Segmentasi Pelanggan Transjakarta", 
    layout="wide",
    initial_sidebar_state="expanded"
)

render_sidebar()

# Memuat metriks dan cm
metrics = joblib.load("model/metrics.pkl")
cm = np.load("model/confusion_matrix.npy")

st.title("Klasifikasi")

#Informasi
with st.container():
    st.subheader("_Support Vector Machine_")
    st.info("Menggunakan label data hasil klasterisasi K-Means optimasi PSO sebagai target klasifikasi.")

    st.info(f"""**Parameter klasifikasi yang digunakan:**  \n
    - Data split = 70:30  \n
    - Kernel = RBF (Radial Basis Function)  \n
    - C = 0,1  \n
    - Gamma = 0,1
    """)

#Hasil klasifikasi 
with st.container():
    st.subheader("Hasil Klasifikasi")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Akurasi",  f"{metrics['akurasi']:.4f}")
    col2.metric("Presisi",  f"{metrics['presisi']:.4f}")
    col3.metric("Recall",   f"{metrics['recall']:.4f}")
    col4.metric("F1 Score", f"{metrics['f1']:.4f}")

    kiri, kanan = st.columns(2, gap="large")
    with kiri: 
        st.write("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix SVM')
        st.pyplot(fig)
    
    with kanan:
        st.write("**Classification Report**")
        report_df = pd.DataFrame(metrics['classification_report']).transpose()
        report_df = report_df.round(3)
        st.dataframe(report_df)
    

    st.subheader("**Perhitungan Akurasi**")
    benar = np.diag(cm).sum()
    total = cm.sum()

    # Hitung TP, TN, FP, FN dari confusion matrix
    TP = np.diag(cm).sum()           # semua diagonal = benar
    FP = cm.sum(axis=0) - np.diag(cm)  # kolom - diagonal
    FN = cm.sum(axis=1) - np.diag(cm)  # baris - diagonal
    TN = cm.sum() - (FP + FN + np.diag(cm)).sum() + np.diag(cm).sum()

    TP_total = int(np.diag(cm).sum())
    FP_total = int(FP.sum())
    FN_total = int(FN.sum())
    TN_total = int(total - TP_total - FP_total - FN_total)

    # Rumus
    st.write("**Rumus Akurasi:**")
    st.latex(r"Akurasi = \frac{TP + TN}{TP + TN + FP + FN}")

    # Perhitungan
    st.write("**Perhitungan:**")
    st.latex(rf"Akurasi = \frac{{{benar}}}{{{total}}} \times 100\%")

    # Hasil
    hasil = (benar / total) * 100
    st.write("**Hasil:**")
    st.latex(rf"Akurasi = {hasil:.2f}\%")

    # Kesimpulan
    st.success(f"Model SVM menghasilkan akurasi sebesar **{hasil:.2f}%** dari total **{total}** data uji, dengan **{benar}** prediksi benar.")