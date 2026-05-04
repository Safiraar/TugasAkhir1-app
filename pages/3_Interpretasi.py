import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

from utils.sidebar import render_sidebar
from utils.interpretasi import interpretasi

st.set_page_config(
    page_title="Segmentasi Pelanggan Transjakarta", 
    layout="wide",
    initial_sidebar_state="expanded"
)

render_sidebar()

# Memuat model dan scaler
kmeans_pso = joblib.load("model/kmeans_pso.pkl")
scaler = joblib.load("model/scaler.pkl")

# Memuat data
df = pd.read_csv('data/rfm.csv')
data_scaled = scaler.transform(df)

# Jalankan clustering
df_plot = df.copy()
labels = kmeans_pso.predict(data_scaled)
df_plot['Klaster'] = labels + 1

# Centroid dikembalikan ke skala asli
centroids_scaled = kmeans_pso.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)
centroid_df = pd.DataFrame(centroids_original, columns=df.columns)
centroid_df.insert(0, "Klaster", range(1, len(centroid_df) + 1))

# Warna viridis (sama dengan page klasterisasi)
cmap = plt.cm.viridis
n_klaster = 7
warna_klaster = {
    k: cmap((k - 1) / (n_klaster - 1)) for k in range(1, n_klaster + 1)
}

# Helper konversi warna ke hex
def to_hex(k):
    r, g, b, _ = warna_klaster[k]
    return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))

st.title("Interpretasi Hasil Klasterisasi")

# Visualisasi 3D + Centroid
with st.container():
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot sebaran data tiap klaster
    for k in sorted(df_plot['Klaster'].unique()):
        subset = df_plot[df_plot['Klaster'] == k]
        ax.scatter(
            subset['Recency'],
            subset['Frequency'],
            subset['Monetary'],
            c=[warna_klaster[k]],
            label=f"Klaster {k} - {interpretasi[k]['tipe']}",
            alpha=1,
            s=15
        )

    # Plot centroid + nomor klaster
    for i, row in centroid_df.iterrows():
        k = int(row['Klaster'])
        ax.scatter(
            row['Recency'],
            row['Frequency'],
            row['Monetary'],
            c=[warna_klaster[k]],
            marker='X',
            s=200,
            edgecolors='black',
            linewidths=0.8,
            zorder=5
        )
        ax.text(
            row['Recency'],
            row['Frequency'],
            row['Monetary'] * 1.05,
            f"  {k}",
            fontsize=11,
            fontweight='bold',
            color='black',
            bbox=dict(
                boxstyle='round,pad=0.2',
                facecolor='white',
                edgecolor=to_hex(k),
                alpha=0.8
            )
        )

    ax.set_xlabel('Recency', labelpad=8)
    ax.set_ylabel('Frequency', labelpad=8)
    ax.set_zlabel('Monetary', labelpad=8)
    ax.set_title('Visualisasi 3D Hasil Klasterisasi K-Means Optimasi PSO', pad=15)
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.05, 1.1),
        fontsize=12,
        markerscale=3,
        frameon=True
    )

    st.pyplot(fig)
    st.caption("Tanda ✕ dengan nomor menunjukkan posisi centroid setiap klaster")


# Tabel Interpretasi
st.write("")
st.subheader("Interpretasi Hasil Masing-Masing Klaster")

for k, info in interpretasi.items():
    with st.container(border=True):
        col_badge, col_konten = st.columns([0.08, 0.92], gap="small")

        with col_badge:
            st.markdown(
                f"""
                <div style='
                    background-color: {to_hex(k)};
                    color: white;
                    border-radius: 50%;
                    width: 48px;
                    height: 48px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 20px;
                    font-weight: bold;
                    margin-top: 4px;
                '>{k}</div>
                """,
                unsafe_allow_html=True
            )

        with col_konten:
            st.markdown(f"**{info['tipe']}**")
            st.markdown(f"📊 *{info['karakteristik']}*")
            st.markdown(f"📝 {info['deskripsi']}")