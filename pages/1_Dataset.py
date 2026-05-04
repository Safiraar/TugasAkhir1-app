import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt 
from pathlib import Path 
from utils.sidebar import render_sidebar

st.set_page_config(
    page_title="Segmentasi Pelanggan Transjakarta", 
    layout="wide",
    initial_sidebar_state="expanded"
)

render_sidebar()

st.title("Dataset")

#PATH 
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_PATH = DATA_DIR / "raw_data.csv"
CLEAN_PATH = DATA_DIR / "clean_data.csv"
RFM_PATH = DATA_DIR / "rfm.csv"

#Load Data
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

with st.spinner("Memuat data..."):
    raw_data = load_csv(RAW_PATH)
    clean_data = load_csv(CLEAN_PATH)
    rfm_data = load_csv(RFM_PATH)

    
#Tampilan 
tab1, tab2, tab3, tab4 = st.tabs(["Deskripsi Data", "Hasil Pre-processing", "EDA", "RFM"])

with tab1: 
    with st.container(): 
        st.subheader("Deskripsi Dataset")
        st.write("**Dataset:** Dataset yang digunakan adalah *Transjakarta - Public Transportation Transaction*")
        st.write("**Sumber Data:** Dataset bersumber dari Kaggle")
        
        m1, m2 = st.columns(2)
        m1.metric("Jumlah Baris", f"{len(raw_data):,}")
        m2.metric("Jumlah Kolom", f"{raw_data.shape[1]}")

        st.write("")
        st.write("**Nama Kolom Data**")
        kolom_list = list(raw_data.columns)
        kolom_1 = pd.DataFrame({"Kolom Data": kolom_list[:11]})
        kolom_2 = pd.DataFrame({"Kolom Data": kolom_list[11:22]})

        kiri, kanan = st.columns([1,1], gap="medium")
        with kiri : 
            st.dataframe(kolom_1, use_container_width=True, height=430)
        with kanan : 
            st.dataframe(kolom_2, use_container_width=True, height=430)
        
        st.write("")
        st.write("**Menampilkan data**")
        jumlah_tampil_raw = st.selectbox(
            "Pilih jumlah baris yang akan ditampilkan",
            [5, 10, 25, 50, 100],
            index=0,
            key="jumlah_tampil_raw",
        )
        st.dataframe(raw_data.head(jumlah_tampil_raw), use_container_width=True)

with tab2: 
    with st.container():
        st.subheader("Hasil Pre-processing")
        st.write("Setelah dilakukan tahap pre-processing data, maka ukuran datasenya menjadi :")
        
        m1, m2 = st.columns(2)
        m1.metric("Jumlah Baris", f"{len(clean_data):,}")
        m2.metric("Jumlah kolom", f"{clean_data.shape[1]}")

        st.write("")
        st.write("**Nama Kolom Data**")
        kolom_list_clean = list(clean_data.columns)
        kolom_1 = pd.DataFrame({"Kolom Data": kolom_list_clean[:4]})
        kolom_2 = pd.DataFrame({"Kolom Data": kolom_list_clean[4:8]})

        kiri, kanan = st.columns([1,1], gap="medium")
        with kiri : 
            st.dataframe(kolom_1, use_container_width=True)
        with kanan : 
            st.dataframe(kolom_2, use_container_width=True)
        
        st.write("")
        st.write("**Menampilkan data**")
        jumlah_tampil_clean = st.selectbox(
            "Pilih jumlah baris yang akan ditampilkan",
            [5, 10, 25, 50, 100],
            index=0,
            key="jumlah_tampil_clean",
        )
        st.dataframe(clean_data.head(jumlah_tampil_clean), use_container_width=True)
        
# Tab EDA
with tab3:
    st.subheader("Exploratory Data Analysis")
    
    eda_menu = st.radio(
        "Pilih analisis EDA", 
        ["Demografi", "Jenis Kartu", "Jam Sibuk", "Top 10 Koridor"], 
        horizontal=True,
    )

    st.write("")

    # Demografi
    if eda_menu == "Demografi":

        df_eda = clean_data.copy()
        df_eda['usia'] = 2026 - df_eda['payCardBirthDate'].astype(int)

        bins = [10, 20, 30, 40, 50, 60, float('inf')]
        labels_usia = ['10-19', '20-29', '30-39', '40-49', '50-59', '>60']
        df_eda['age_group'] = pd.cut(df_eda['usia'], bins=bins, labels=labels_usia, right=False)
        usia_group = df_eda['age_group'].value_counts().sort_index()

        gender_counts = df_eda['payCardSex'].value_counts()
        gender_labels = ['Perempuan' if g == 'F' else 'Laki-laki' for g in gender_counts.index]

        kiri, kanan = st.columns(2, gap="medium")

        with kiri:
            with st.container(border=True):
                st.write("**Distribusi Usia Pelanggan**")
                fig, ax = plt.subplots(figsize=(7, 5))
                bars = ax.bar(usia_group.index, usia_group.values, color='cornflowerblue')
                for bar in bars:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 300,
                        f"{int(bar.get_height()):,}",
                        ha='center', va='bottom', fontsize=9
                    )
                ax.set_title("Distribusi Usia Pelanggan")
                ax.set_xlabel("Usia")
                ax.set_ylabel("Jumlah")
                st.pyplot(fig)

            st.write("**Data Distribusi Usia**")
            usia_df = usia_group.reset_index()
            usia_df.columns = ["Kelompok Usia", "Jumlah"]
            st.dataframe(usia_df, use_container_width=True)

        with kanan:
            with st.container(border=True):
                st.write("**Distribusi Jenis Kelamin Pelanggan**")
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.pie(
                    gender_counts.values,
                    labels=gender_counts.index,
                    autopct='%1.1f%%',
                    colors=['#E07EA8', 'cornflowerblue'],
                    startangle=0
                )
                ax.legend(gender_labels, title="Kategori", loc="lower right", bbox_to_anchor=(1.15, 0))
                ax.set_title("Distribusi Jenis Kelamin Pelanggan")
                st.pyplot(fig)

            st.write("**Data Distribusi Jenis Kelamin**")
            gender_df = gender_counts.reset_index()
            gender_df.columns = ["Kode", "Jumlah"]
            gender_df.insert(1, "Jenis Kelamin", gender_df["Kode"].map({"F": "Perempuan", "M": "Laki-laki"}))
            st.dataframe(gender_df, use_container_width=True)

    # Jenis Kartu
    elif eda_menu == "Jenis Kartu":
        kartu_counts = clean_data['payCardBank'].value_counts().head(6)

        kiri, kanan = st.columns(2, gap="medium")

        with kiri:
            with st.container(border=True):
                st.write("**Distribusi Penggunaan Jenis Kartu Pelanggan**")
                fig, ax = plt.subplots(figsize=(7, 5))
                bars = ax.barh(
                    kartu_counts.index[::-1],
                    kartu_counts.values[::-1],
                    color='cornflowerblue'
                )
                for bar in bars:
                    ax.text(
                        bar.get_width() + 300,
                        bar.get_y() + bar.get_height() / 2,
                        f"{int(bar.get_width()):,}",
                        ha='left', va='center', fontsize=9
                    )
                max_val = kartu_counts.values.max()
                ax.set_xlim(0, max_val * 1.15)

                ax.set_title("Distribusi Penggunaan Jenis Kartu Pelanggan")
                ax.set_xlabel("Jumlah")
                ax.set_ylabel("Jenis Kartu")
                st.pyplot(fig)

        with kanan:
            st.write("**Data Jenis Kartu**")
            kartu_df = kartu_counts.reset_index()
            kartu_df.columns = ["Jenis Kartu", "Jumlah"]
            st.dataframe(kartu_df, use_container_width=True)

    # Jam Sibuk
    elif eda_menu == "Jam Sibuk":
        df_jam = clean_data.copy()
        df_jam['jam'] = pd.to_datetime(df_jam['tapInTime'], errors='coerce').dt.hour
        jam_counts = df_jam['jam'].value_counts().sort_index()
        all_hours = pd.Series(0, index=range(24))
        jam_counts = (all_hours + jam_counts).fillna(0).astype(int)

        kiri, kanan = st.columns(2, gap="medium")

        with kiri:
            with st.container(border=True):
                st.write("**Distribusi Jam Sibuk**")
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.bar(jam_counts.index, jam_counts.values, color='cornflowerblue')
                ax.set_title("Distribusi Jam Sibuk")
                ax.set_xlabel("Jam")
                ax.set_ylabel("Jumlah Transaksi")
                ax.set_xticks(range(24))
                st.pyplot(fig)

        with kanan:
            st.write("**Data Jam Sibuk**")
            jam_df = jam_counts.reset_index()
            jam_df.columns = ["Jam", "Jumlah Transaksi"]
            st.dataframe(jam_df, use_container_width=True)

    # Top 10 Koridor
    elif eda_menu == "Top 10 Koridor":
        koridor_counts = clean_data['tapInStopsName'].value_counts().head(10)

        kiri, kanan = st.columns(2, gap="medium")

        with kiri:
            with st.container(border=True):
                st.write("**Top 10 Koridor Transjakarta dengan Transaksi Tertinggi**")
                fig, ax = plt.subplots(figsize=(7, 6))
                bars = ax.barh(
                    koridor_counts.index[::-1],
                    koridor_counts.values[::-1],
                    color='cornflowerblue'
                )
                for bar in bars:
                    ax.text(
                        bar.get_width() + 5,
                        bar.get_y() + bar.get_height() / 2,
                        f"{int(bar.get_width()):,}",
                        ha='left', va='center', fontsize=9
                    )
                
                max_val = koridor_counts.values.max()
                ax.set_xlim(0, max_val * 1.15)

                ax.set_title("Top 10 Koridor Transjakarta dengan Transaksi Tertinggi")
                ax.set_xlabel("Jumlah Transaksi")
                ax.set_ylabel("Koridor")
                st.pyplot(fig)

        with kanan:
            st.write("**Data Top 10 Koridor**")
            koridor_df = koridor_counts.reset_index()
            koridor_df.columns = ["Koridor", "Jumlah Transaksi"]
            st.dataframe(koridor_df, use_container_width=True)
    
    
with tab4: 
    with st.container(): 
        st.subheader("RFM")

        st.write("**_Recency:_** Menunjukkan jarak waktu sejak transaksi terakhir pelanggan.")
        st.write("**_Frequency:_** Menunjukkan frekuensi atau jumlah transaksi yang dilakukan pelanggan.")
        st.write("**_Monetary:_** Menunjukkan total nilai transaksi yang dilakukan pelanggan.")

        st.write("")

        m1, m2 = st.columns(2)
        m1.metric("Jumlah Baris", f"{len(rfm_data):,}")
        m2.metric("Jumlah Kolom", f"{rfm_data.shape[1]}")

        st.write("")

        jumlah_tampil_rfm = st.selectbox(
            "Pilih jumlah baris data RFM yang ditampilkan",
            [5, 10, 25, 50, 100],
            index=0,
            key="jumlah_tampil_rfm",
        )
        st.dataframe(rfm_data.head(jumlah_tampil_rfm), use_container_width=True)
