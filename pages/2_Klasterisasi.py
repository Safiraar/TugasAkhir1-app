import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

from utils.sidebar import render_sidebar, ASSETS_DIR

st.set_page_config(
    page_title="Segmentasi Pelanggan Transjakarta", 
    layout="wide",
    initial_sidebar_state="expanded"
)

render_sidebar()

# ──────────────────────────────────────────
# LOAD DATA DARI JOBLIB
# ──────────────────────────────────────────
hasil_kmeans     = joblib.load("model/hasil_kmeans.pkl")
hasil_kmeans_pso = joblib.load("model/hasil_kmeans_pso.pkl")

# K-Means
silhouette_kmeans        = hasil_kmeans['silhouette_score']
dbi_kmeans               = hasil_kmeans['dbi']
cluster_counts_kmeans    = hasil_kmeans['jumlah_anggota']
centroid_scaled_kmeans   = hasil_kmeans['centroid_scaled']
centroid_asli_kmeans     = hasil_kmeans['centroid_asli']
df_kmeans                = hasil_kmeans['hasil_klaster']
rfm_scaled_kmeans        = hasil_kmeans['rfm_scaled']

# K-Means PSO
silhouette_kmeans_pso        = hasil_kmeans_pso['silhouette_score']
dbi_kmeans_pso               = hasil_kmeans_pso['dbi']
cluster_counts_kmeans_pso    = hasil_kmeans_pso['jumlah_anggota']
centroid_scaled_kmeans_pso   = hasil_kmeans_pso['centroid_scaled']
centroid_asli_kmeans_pso     = hasil_kmeans_pso['centroid_asli']
df_kmeans_pso                = hasil_kmeans_pso['hasil_klaster']
rfm_scaled_kmeans_pso        = hasil_kmeans_pso['rfm_scaled']

# ──────────────────────────────────────────
# DATA PENENTUAN JUMLAH KLASTER
# ──────────────────────────────────────────
silhouette_scores = {
    "Klaster": [2, 3, 4, 5, 6, 7, 8],
    "Silhouette Score": [0.555, 0.651, 0.705, 0.639, 0.724, 0.729, 0.613]
}
silhouette_df = pd.DataFrame(silhouette_scores)

# ──────────────────────────────────────────
# JUDUL
# ──────────────────────────────────────────
st.title("Klasterisasi")

# ──────────────────────────────────────────
# PENENTUAN JUMLAH KLASTER
# ──────────────────────────────────────────
with st.container(border=True):
    st.success("Jumlah klaster yang digunakan adalah 7")

    with st.expander("Lihat selengkapnya"):
        st.write("")
        kiri, kanan = st.columns([1, 1.4], gap="medium")

        with kiri:
            st.write("Hasil perhitungan Silhouette Score")
            st.dataframe(silhouette_df)

        with kanan:
            jumlah_klaster = ASSETS_DIR / "jumlah_klaster.png"
            st.image(str(jumlah_klaster), use_container_width=True)

        st.info("Gunakan jumlah klaster yang memiliki nilai Silhouette Score terbesar")

# ──────────────────────────────────────────
# PILIHAN METODE
# ──────────────────────────────────────────
st.write("")
metode = st.selectbox(
    "**Pilih metode klasterisasi**",
    ["K-Means", "K-Means Optimasi PSO", "Perbandingan"],
    index=0
)

# ──────────────────────────────────────────
# K-MEANS
# ──────────────────────────────────────────
if metode == "K-Means":
    st.subheader("K-Means")

    m1, m2 = st.columns(2)
    m1.metric("Silhouette Score", f"{silhouette_kmeans:.3f}")
    m2.metric("DBI", f"{dbi_kmeans:.3f}")

    kiri, kanan = st.columns([1, 1], gap="medium")

    with kiri:
        st.write("**Jumlah Anggota Setiap Klaster K-Means:**")
        st.dataframe(cluster_counts_kmeans, use_container_width=True)

    with kanan:
        st.write("**Visualisasi 3D Hasil Klasterisasi K-Means**")

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(
            rfm_scaled_kmeans['Recency'],
            rfm_scaled_kmeans['Frequency'],
            rfm_scaled_kmeans['Monetary'],
            c=rfm_scaled_kmeans['Klaster'],
            cmap='viridis'
        )
        ax.set_xlabel('Recency')
        ax.set_ylabel('Frequency')
        ax.set_zlabel('Monetary')
        ax.set_title('Visualisasi 3D K-Means')

        handles, labels_legend = scatter.legend_elements()
        labels_legend = [str(i + 1) for i in range(len(handles))]
        ax.legend(handles, labels_legend, title="Klaster", loc='upper left', bbox_to_anchor=(1, 1.1))

        st.pyplot(fig)

    st.write("**Posisi Centroid Setiap Klaster K-Means (Scaled):**")
    st.dataframe(centroid_scaled_kmeans.round(3), use_container_width=True)

    st.write("**Posisi Centroid Setiap Klaster K-Means (Skala Asli):**")
    st.dataframe(centroid_asli_kmeans.round(3), use_container_width=True)

    st.write("")
    st.write("**Hasil Klasterisasi K-Means**")
    jumlah_tampil_kmeans = st.selectbox(
        "Pilih jumlah baris data yang ditampilkan",
        [5, 10, 25, 50, 100],
        index=0,
        key="jumlah_tampil_kmeans"
    )
    st.dataframe(df_kmeans.head(jumlah_tampil_kmeans), use_container_width=True)

# ──────────────────────────────────────────
# K-MEANS OPTIMASI PSO
# ──────────────────────────────────────────
if metode == "K-Means Optimasi PSO":
    st.subheader("K-Means Optimasi PSO")

    m1, m2 = st.columns(2)
    m1.metric("Silhouette Score", f"{silhouette_kmeans_pso:.3f}")
    m2.metric("DBI", f"{dbi_kmeans_pso:.3f}")

    kiri, kanan = st.columns([1, 1], gap="medium")

    with kiri:
        st.write("**Jumlah Anggota Setiap Klaster K-Means Optimasi PSO:**")
        st.dataframe(cluster_counts_kmeans_pso, use_container_width=True)

    with kanan:
        st.write("**Visualisasi 3D Hasil Klasterisasi K-Means Optimasi PSO**")

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(
            rfm_scaled_kmeans_pso['Recency'],
            rfm_scaled_kmeans_pso['Frequency'],
            rfm_scaled_kmeans_pso['Monetary'],
            c=rfm_scaled_kmeans_pso['Klaster'],
            cmap='viridis'
        )
        ax.set_xlabel('Recency')
        ax.set_ylabel('Frequency')
        ax.set_zlabel('Monetary')
        ax.set_title('Visualisasi 3D K-Means Optimasi PSO')

        handles, labels_legend = scatter.legend_elements()
        labels_legend = [str(i + 1) for i in range(len(handles))]
        ax.legend(handles, labels_legend, title="Klaster", loc='upper left', bbox_to_anchor=(1, 1.1))

        st.pyplot(fig)

    st.write("**Posisi Centroid Setiap Klaster K-Means Optimasi PSO (Scaled):**")
    st.dataframe(centroid_scaled_kmeans_pso.round(3), use_container_width=True)

    st.write("**Posisi Centroid Setiap Klaster K-Means Optimasi PSO (Skala Asli):**")
    st.dataframe(centroid_asli_kmeans_pso.round(3), use_container_width=True)

    st.write("")
    st.write("**Hasil Klasterisasi K-Means Optimasi PSO**")
    jumlah_tampil_kmeans_pso = st.selectbox(
        "Pilih jumlah baris data yang ditampilkan",
        [5, 10, 25, 50, 100],
        index=0,
        key="jumlah_tampil_kmeans_pso"
    )
    st.dataframe(df_kmeans_pso.head(jumlah_tampil_kmeans_pso), use_container_width=True)

# ──────────────────────────────────────────
# PERBANDINGAN
# ──────────────────────────────────────────
if metode == "Perbandingan":
    st.subheader("Perbandingan Hasil Klasterisasi")

    kiri, kanan = st.columns(2, gap="small")

    with kiri:
        st.error(
            "**K-Means**  \n"
            f"Silhouette Score : {silhouette_kmeans:.3f}  \n"
            f"DBI : {dbi_kmeans:.3f}"
        )

    with kanan:
        st.success(
            "**K-Means Optimasi PSO**  \n"
            f"Silhouette Score : {silhouette_kmeans_pso:.3f}  \n"
            f"DBI : {dbi_kmeans_pso:.3f}"
        )

    st.write("")

    kiri, kanan = st.columns(2, gap="small")

    with kiri:
        with st.container(border=True):
            st.subheader("Perbandingan *Silhouette Score*")

            fig_sil, ax_sil = plt.subplots(figsize=(6, 5))
            bars = ax_sil.bar(
                ["K-Means", "K-Means PSO"],
                [silhouette_kmeans, silhouette_kmeans_pso],
                color=['red', 'green']
            )
            for bar in bars:
                ax_sil.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}",
                    ha='center', va='bottom', fontsize=11, fontweight='bold'
                )
            ax_sil.set_title("Perbandingan Silhouette Score")
            ax_sil.set_ylabel("Nilai")
            ax_sil.set_ylim(0, 1)
            ax_sil.set_xlabel("Klasterisasi")
            st.pyplot(fig_sil)

    with kanan:
        with st.container(border=True):
            st.subheader("Perbandingan *Davies-Bouldin Index* (DBI)")

            fig_dbi, ax_dbi = plt.subplots(figsize=(6, 5))
            bars = ax_dbi.bar(
                ["K-Means", "K-Means PSO"],
                [dbi_kmeans, dbi_kmeans_pso],
                color=['red', 'green']
            )
            for bar in bars:
                ax_dbi.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}",
                    ha='center', va='bottom', fontsize=11, fontweight='bold'
                )
            ax_dbi.set_title("Perbandingan Davies-Bouldin Index")
            ax_dbi.set_ylabel("Nilai")
            ax_dbi.set_xlabel("Klasterisasi")
            ax_dbi.set_ylim(0, 1)
            st.pyplot(fig_dbi)

    st.write("")
    with st.container(border=True):
        st.subheader("Keterangan")

        if silhouette_kmeans_pso > silhouette_kmeans and dbi_kmeans_pso < dbi_kmeans:
            st.success("**K-Means Optimasi PSO** menghasilkan kualitas klasterisasi yang lebih baik dibandingkan K-Means.")
            st.info(f"✅ Silhouette Score K-Means Optimasi PSO lebih tinggi ({silhouette_kmeans_pso:.3f} > {silhouette_kmeans:.3f})")
            st.info(f"✅ DBI K-Means Optimasi PSO lebih rendah ({dbi_kmeans_pso:.3f} < {dbi_kmeans:.3f})")
        elif silhouette_kmeans > silhouette_kmeans_pso and dbi_kmeans < dbi_kmeans_pso:
            st.success("**K-Means** menghasilkan kualitas klasterisasi yang lebih baik dibandingkan K-Means Optimasi PSO.")
            st.info(f"✅ Silhouette Score K-Means lebih tinggi ({silhouette_kmeans:.3f} > {silhouette_kmeans_pso:.3f})")
            st.info(f"✅ DBI K-Means lebih rendah ({dbi_kmeans:.3f} < {dbi_kmeans_pso:.3f})")
        else:
            st.warning("Hasil tidak konklusif, perlu analisis lebih lanjut.")
            st.info(f"Silhouette Score — K-Means: {silhouette_kmeans:.3f} | K-Means PSO: {silhouette_kmeans_pso:.3f}")
            st.info(f"DBI — K-Means: {dbi_kmeans:.3f} | K-Means PSO: {dbi_kmeans_pso:.3f}")