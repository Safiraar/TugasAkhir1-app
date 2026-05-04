import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score, davies_bouldin_score
from utils.sidebar import render_sidebar, ASSETS_DIR

st.set_page_config(
    page_title="Segmentasi Pelanggan Transjakarta", 
    layout="wide",
    initial_sidebar_state="expanded"
)

render_sidebar()

# Memuat model dan scaler yang sudah disimpan
kmeans = joblib.load("model/kmeans.pkl")
kmeans_pso = joblib.load("model/kmeans_pso.pkl")
scaler = joblib.load("model/scaler.pkl")

# Fungsi untuk klasterisasi dan evaluasi
def run_clustering(model, data, original_df, scaler):
    labels = model.predict(data)
    
    silhouette = silhouette_score(data, labels)
    dbi = davies_bouldin_score(data, labels)

    unique, counts = np.unique(labels, return_counts=True)

    cluster_counts_df = pd.DataFrame({
        "Cluster": [int(i) + 1 for i in unique],
        "Jumlah Data": [int(j) for j in counts]
    })
    centroids = model.cluster_centers_

    #Centroid 
    centroid_df = pd.DataFrame(
        centroids, 
        columns=original_df.columns
        )
    centroid_df.insert(0, "Cluster", range(1, len(centroid_df) + 1))
    
    # Centroid skala asli 
    centroid_asli_df = pd.DataFrame(
        scaler.inverse_transform(centroids),
        columns=original_df.columns
    )
    centroid_asli_df.insert(0, "Cluster", range(1, len(centroid_asli_df) + 1))

    original_df['Klaster'] = labels + 1

    return labels, silhouette, dbi, cluster_counts_df, centroid_df, centroid_asli_df

# Memuat data
df = pd.read_csv('data/rfm.csv')
data_scaled = scaler.transform(df)

# Silhouette score untuk penentuan jumlah klaster
silhouette_scores = {
    "Klaster": [2, 3, 4, 5, 6, 7, 8],
    "Silhouette Score": [0.555, 0.651, 0.705, 0.639, 0.724, 0.729, 0.613]
}
silhouette_df = pd.DataFrame(silhouette_scores)

# Jalankan clustering SEMUA metode sebelum blok if metode
df_kmeans = df.copy()
cluster_kmeans, silhouette_kmeans, dbi_kmeans, cluster_counts_kmeans, centroid_positions_kmeans, centroid_asli_kmeans = run_clustering(kmeans, data_scaled, df_kmeans, scaler)

df_kmeans_pso = df.copy()
cluster_kmeans_pso, silhouette_kmeans_pso, dbi_kmeans_pso, cluster_counts_kmeans_pso, centroid_positions_kmeans_pso, centroid_asli_kmeans_pso = run_clustering(kmeans_pso, data_scaled, df_kmeans_pso, scaler)


st.title("Klasterisasi")

# Penentuan jumlah klaster
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
            
        st.info("Gunakan jumlah klaster yang memiliki nilai silhouette score terbesar")

# Pilihan metode
st.write("")
metode = st.selectbox(
    "**Pilih metode klasterisasi**",
    ["K-Means", "K-Means Optimasi PSO", "Perbandingan"],
    index=0
)

# K-Means
if metode == "K-Means":
    st.subheader("K-Means")

    m1, m2 = st.columns(2)
    m1.metric("Silhouette Score", f"{silhouette_kmeans:.3f}")
    m2.metric("DBI", f"{dbi_kmeans:.3f}")
    
    kiri, kanan = st.columns([1, 1], gap="medium")
    with kiri:
        with st.container():
            st.write("**Jumlah Anggota Setiap Klaster K-Means:**")
            st.write(cluster_counts_kmeans)
    
    with kanan:
        with st.container():
            st.write("**Visualisasi 3D Hasil Klasterisasi K-Means**")

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')

            scatter = ax.scatter(
                df_kmeans['Recency'], df_kmeans['Frequency'], df_kmeans['Monetary'],
                c=cluster_kmeans, cmap='viridis'
            )
            ax.set_xlabel('Recency')
            ax.set_ylabel('Frequency')
            ax.set_zlabel('Monetary')
            ax.set_title('Visualisasi 3D K-Means')

            handles, labels = scatter.legend_elements()
            labels = [str(i + 1) for i in range(len(handles))]
            ax.legend(handles, labels, title="Klaster", loc='upper left', bbox_to_anchor=(1, 1.1))

            st.pyplot(fig)

    st.write("**Posisi Centroid Setiap Klaster K-Means:**")
    st.write(centroid_positions_kmeans.round(3))

    st.write("**Posisi Centroid Skala Asli K-Means:**")
    st.write(centroid_asli_kmeans.round(3))

    st.write("")
    st.write("**Hasil Klasterisasi K-Means**")
    jumlah_tampil_kmeans = st.selectbox(
        "Pilih jumlah baris data yang ditampilkan",
        [5, 10, 25, 50, 100],
        index=0,
        key="jumlah_tampil_kmeans"
    )
    st.dataframe(df_kmeans.head(jumlah_tampil_kmeans), use_container_width=True)

# K-Means PSO
if metode == "K-Means Optimasi PSO":
    st.subheader("K-Means Optimasi PSO")

    m1, m2 = st.columns(2)
    m1.metric("Silhouette Score", f"{silhouette_kmeans_pso:.3f}")
    m2.metric("DBI", f"{dbi_kmeans_pso:.3f}")
    
    kiri, kanan = st.columns([1, 1], gap="medium")
    with kiri:
        with st.container():
            st.write("**Jumlah Anggota Setiap Klaster K-Means Optimasi PSO:**")
            st.write(cluster_counts_kmeans_pso)
    
    with kanan:
        with st.container():
            st.write("**Visualisasi 3D Hasil Klasterisasi K-Means Optimasi PSO**")

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')

            scatter = ax.scatter(
                df_kmeans_pso['Recency'], df_kmeans_pso['Frequency'], df_kmeans_pso['Monetary'],
                c=cluster_kmeans_pso, cmap='viridis'
            )
            ax.set_xlabel('Recency')
            ax.set_ylabel('Frequency')
            ax.set_zlabel('Monetary')
            ax.set_title('Visualisasi 3D K-Means Optimasi PSO')

            handles, labels = scatter.legend_elements()
            labels = [str(i + 1) for i in range(len(handles))]
            ax.legend(handles, labels, title="Klaster", loc='upper left', bbox_to_anchor=(1, 1.1))

            st.pyplot(fig)

    st.write("**Posisi Centroid Setiap Klaster K-Means Optimasi PSO:**")
    st.write(centroid_positions_kmeans_pso.round(3))

    st.write("**Posisi Centroid Skala Asli K-Means Optimasi PSO:**")
    st.write(centroid_asli_kmeans_pso.round(3))

    st.write("")
    st.write("**Hasil Klasterisasi K-Means Optimasi PSO**")
    jumlah_tampil_kmeans_pso = st.selectbox(
        "Pilih jumlah baris data yang ditampilkan",
        [5, 10, 25, 50, 100],
        index=0,
        key="jumlah_tampil_kmeans_pso"
    )
    st.dataframe(df_kmeans_pso.head(jumlah_tampil_kmeans_pso), use_container_width=True)

# Perbandingan
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

    # Visualisasi
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
            # Tampilkan angka di atas setiap bar
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
            # Tampilkan angka di atas setiap bar
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

    # Keterangan
    st.write("")
    with st.container(border=True):
        st.subheader("Keterangan")

        if silhouette_kmeans_pso > silhouette_kmeans and dbi_kmeans_pso < dbi_kmeans:
            st.success("**K-Means optimasi PSO** lebih baik dari K-Means.")
            st.info("✅ Silhouette Score K-Means optimasi PSO lebih tinggi (+ 0,092)")
            st.info("✅ DBI K-Means optimasi PSO lebih rendah (- 0,142)")
        elif silhouette_kmeans > silhouette_kmeans_pso and dbi_kmeans < dbi_kmeans_pso:
            st.success("K-Means lebih baik dari K-Means PSO.")
            st.info("✅ Silhouette Score K-Means lebih tinggi")
            st.info("✅ DBI K-Means lebih rendah")
        else:
            st.warning("Hasil tidak konklusif, perlu analisis lebih lanjut.")
            st.info(f"Silhouette Score — K-Means: {silhouette_kmeans:.3f} | K-Means PSO: {silhouette_kmeans_pso:.3f}")
            st.info(f"DBI — K-Means: {dbi_kmeans:.3f} | K-Means PSO: {dbi_kmeans_pso:.3f}")